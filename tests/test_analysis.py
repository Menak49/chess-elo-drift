import numpy as np
import pandas as pd
import pytest

from chess_elo_drift import config
from chess_elo_drift.analysis.dataset import (
    build_estimation_sample,
    coverage_of_reported_accuracy,
    _report_band,
)
from chess_elo_drift.analysis.statistics import (
    _ols_with_clustered_errors,
    compare_bands,
    estimate_all_gaps,
    estimate_gap,
    rating_equivalent_of_gap,
    validate_against_reported,
)

LEGACY = config.LEGACY_ERA.name
MODERN = config.MODERN_ERA.name


def row(**overrides):
    """One engine-output row, with every column the analysis reads."""
    payload = {
        "game_id": "g1",
        "era": LEGACY,
        "month": "2018-03",
        "time_class": "blitz",
        "time_control": "300",
        "colour": "white",
        "username": "alice",
        "rating": 1500,
        "opponent_rating": 1490,
        "result": "win",
        "ply_count": 60,
        "moves_scored": 25,
        "accuracy": 85.0,
        "acpl": 40.0,
        "inaccuracies": 3,
        "mistakes": 1,
        "blunders": 0,
        "reported_accuracy": None,
        "engine_depth": config.ENGINE_DEPTH,
    }
    payload.update(overrides)
    return payload


def frame(rows):
    return pd.DataFrame([row(**r) for r in rows])


# -- inclusion rules ---------------------------------------------------------


def test_sample_drops_sides_the_engine_could_not_score():
    sample = build_estimation_sample(frame([{"accuracy": None}, {"game_id": "g2"}]))
    assert list(sample["game_id"]) == ["g2"]


def test_sample_drops_sides_scored_on_too_few_moves():
    thin = config.MIN_MOVES_SCORED_PER_SIDE - 1
    sample = build_estimation_sample(
        frame([{"game_id": "g1", "moves_scored": thin}, {"game_id": "g2", "moves_scored": thin + 1}])
    )
    assert list(sample["game_id"]) == ["g2"]


@pytest.mark.parametrize("rating", [config.RATING_MIN - 1, config.RATING_MAX + 1])
def test_sample_drops_ratings_outside_the_study_window(rating):
    assert build_estimation_sample(frame([{"rating": rating}])).empty


def test_sample_caps_how_much_one_player_contributes():
    rows = [{"game_id": f"g{i}", "username": "prolific"} for i in range(10)]
    sample = build_estimation_sample(frame(rows), max_per_player=3)
    assert len(sample) == 3


def test_the_cap_applies_per_era_and_cadence_not_globally():
    rows = [
        {"game_id": f"{era}-{tc}-{i}", "username": "prolific", "era": era, "time_class": tc}
        for era in (LEGACY, MODERN)
        for tc in ("blitz", "rapid")
        for i in range(5)
    ]
    sample = build_estimation_sample(frame(rows), max_per_player=3)
    assert len(sample) == 12  # 3 per (era, cadence) cell, four cells
    counts = sample.groupby(["era", "time_class"], observed=True).size()
    assert set(counts) == {3}


def test_the_cap_keeps_the_same_rows_whatever_the_collection_order():
    rows = [{"game_id": f"g{i}", "username": "prolific"} for i in range(10)]
    forward = build_estimation_sample(frame(rows), max_per_player=3)
    backward = build_estimation_sample(frame(list(reversed(rows))), max_per_player=3)
    assert list(forward["game_id"]) == list(backward["game_id"])


def test_sample_labels_the_era_under_study():
    sample = build_estimation_sample(frame([{"era": LEGACY}, {"era": MODERN, "game_id": "g2"}]))
    assert dict(zip(sample["era"], sample["is_modern"])) == {LEGACY: 0, MODERN: 1}


def test_rating_is_centred_on_the_window_midpoint():
    midpoint = (config.RATING_MIN + config.RATING_MAX) / 2
    sample = build_estimation_sample(frame([{"rating": int(midpoint) + 200}]))
    assert sample["rating_centred"].iloc[0] == pytest.approx(200)


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(600, 600), (999, 600), (1000, 1000), (1799, 1400), (1800, 1800), (2200, 1800)],
)
def test_report_bands_are_400_wide_and_clip_at_the_top(rating, expected):
    assert _report_band(pd.Series([rating])).iloc[0] == expected


def test_reported_accuracy_coverage_is_measured_per_era_and_cadence():
    rows = [
        {"game_id": "g1", "era": MODERN, "reported_accuracy": 80.0},
        {"game_id": "g2", "era": MODERN, "reported_accuracy": None},
        {"game_id": "g3", "era": LEGACY, "reported_accuracy": None},
    ]
    coverage = coverage_of_reported_accuracy(frame(rows)).set_index("era")["coverage"]
    assert coverage[MODERN] == pytest.approx(0.5)
    assert coverage[LEGACY] == pytest.approx(0.0)


# -- least squares with clustered errors -------------------------------------


def test_least_squares_recovers_the_coefficients_it_was_given():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    design = np.column_stack([np.ones(400), x])
    outcome = 3.0 + 2.0 * x + rng.normal(scale=0.01, size=400)
    fit = _ols_with_clustered_errors(design, outcome, np.arange(400))
    assert fit.coefficients == pytest.approx([3.0, 2.0], abs=0.01)
    assert fit.clusters == 400


def test_clustering_widens_the_error_bar_when_rows_repeat_a_player():
    """Twenty players measured twenty times each is not four hundred observations.

    Every row of a cluster carries the same player offset, so treating them as
    independent understates the uncertainty. That is exactly the situation the
    opponent-graph walk produces, so the correction has to bite.
    """
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 20)
    offsets = rng.normal(scale=5.0, size=20)[groups]
    design = np.column_stack([np.ones(400), rng.normal(size=400)])
    outcome = 50.0 + offsets + rng.normal(scale=0.5, size=400)

    clustered = _ols_with_clustered_errors(design, outcome, groups)
    independent = _ols_with_clustered_errors(design, outcome, np.arange(400))
    assert clustered.standard_errors[0] > 3 * independent.standard_errors[0]


# -- the headline estimate ---------------------------------------------------


def planted_sample(era_effect: float, *, slope_per_100: float = 1.5, noise: float = 0.2, n: int = 240):
    """A sample where the era gap and the rating slope are known by construction."""
    rng = np.random.default_rng(7)
    rows = []
    for index in range(n):
        era = LEGACY if index % 2 else MODERN
        rating = int(rng.integers(config.RATING_MIN, config.RATING_MAX + 1))
        accuracy = (
            75.0
            + slope_per_100 * (rating - 1400) / 100.0
            + (era_effect if era == MODERN else 0.0)
            + rng.normal(scale=noise)
        )
        rows.append(
            {
                "game_id": f"g{index}",
                "username": f"player{index}",  # one row each: no clustering to model
                "era": era,
                "rating": rating,
                "accuracy": round(accuracy, 3),
            }
        )
    return build_estimation_sample(frame(rows))


def test_the_era_coefficient_recovers_a_planted_gap():
    gap = estimate_gap(planted_sample(-2.5), "blitz")
    assert gap is not None
    assert gap.estimate == pytest.approx(-2.5, abs=0.15)
    assert gap.p_value < 0.001
    assert gap.ci_low < -2.5 < gap.ci_high


def test_a_sample_with_no_era_gap_is_not_reported_as_one():
    gap = estimate_gap(planted_sample(0.0), "blitz")
    assert gap is not None
    assert gap.estimate == pytest.approx(0.0, abs=0.15)
    assert gap.p_value > 0.05


def test_the_estimate_survives_the_eras_sitting_at_different_ratings_in_a_band():
    """The artefact the continuous rating control exists to rule out.

    Modern rows are placed 150 points higher than legacy ones inside the same
    400-point band. A comparison of band means would read that gap in rating as
    a gap in skill; the regression must not.
    """
    rng = np.random.default_rng(11)
    rows = []
    for index in range(240):
        era = LEGACY if index % 2 else MODERN
        base = 1450 if era == MODERN else 1300
        rating = int(np.clip(base + rng.integers(-40, 41), config.RATING_MIN, config.RATING_MAX))
        accuracy = 75.0 + 1.5 * (rating - 1400) / 100.0 + rng.normal(scale=0.2)
        rows.append(
            {
                "game_id": f"g{index}",
                "username": f"player{index}",
                "era": era,
                "rating": rating,
                "accuracy": round(accuracy, 3),
            }
        )
    gap = estimate_gap(build_estimation_sample(frame(rows)), "blitz")
    assert gap is not None
    assert gap.estimate == pytest.approx(0.0, abs=0.2)


def test_no_estimate_is_offered_when_one_era_is_missing():
    rows = [
        {"game_id": f"g{i}", "username": f"p{i}", "era": LEGACY, "rating": 1000 + i}
        for i in range(40)
    ]
    assert estimate_gap(build_estimation_sample(frame(rows)), "blitz") is None


def test_no_estimate_is_offered_on_a_sample_too_small_to_fit():
    rows = [
        {
            "game_id": f"g{i}",
            "username": f"p{i}",
            "era": LEGACY if i % 2 else MODERN,
            "rating": 1000 + i,
        }
        for i in range(10)
    ]
    assert estimate_gap(build_estimation_sample(frame(rows)), "blitz") is None


def test_every_cadence_and_metric_gets_a_row():
    sample = planted_sample(-2.0)
    both_cadences = pd.concat([sample, sample.assign(time_class="rapid")], ignore_index=True)
    gaps = estimate_all_gaps(both_cadences)
    assert set(zip(gaps["time_class"], gaps["metric"])) == {
        ("blitz", "accuracy"),
        ("blitz", "acpl"),
        ("rapid", "accuracy"),
        ("rapid", "acpl"),
    }


def test_the_gap_converts_to_rating_points_through_the_fitted_slope():
    """A 1.5-point-per-100 slope makes a 3-point gap worth about 200 rating points."""
    equivalent = rating_equivalent_of_gap(planted_sample(-3.0, slope_per_100=1.5), "blitz")
    assert equivalent is not None
    assert equivalent == pytest.approx(-200, rel=0.15)


# -- the band-by-band shape check --------------------------------------------


def test_band_comparison_reports_the_difference_between_eras():
    rows = [
        {"game_id": f"g{i}", "username": f"p{i}", "era": LEGACY, "rating": 1500, "accuracy": 80.0 + i % 3}
        for i in range(10)
    ] + [
        {"game_id": f"h{i}", "username": f"q{i}", "era": MODERN, "rating": 1500, "accuracy": 85.0 + i % 3}
        for i in range(10)
    ]
    comparison = compare_bands(build_estimation_sample(frame(rows)))
    assert len(comparison) == 1
    assert comparison["difference"].iloc[0] == pytest.approx(5.0)
    assert comparison["band_label"].iloc[0] == "1400-1799"


def test_a_band_with_one_era_missing_is_skipped_rather_than_guessed():
    rows = [
        {"game_id": f"g{i}", "username": f"p{i}", "era": LEGACY, "rating": 700}
        for i in range(5)
    ]
    assert compare_bands(build_estimation_sample(frame(rows))).empty


# -- the cross-check against chess.com's own number --------------------------


def test_validation_needs_enough_overlap_to_say_anything():
    rows = [
        {"game_id": f"g{i}", "username": f"p{i}", "reported_accuracy": 70.0}
        for i in range(5)
    ]
    assert validate_against_reported(build_estimation_sample(frame(rows))) == {"n": 5.0}


def test_validation_reports_how_the_two_scales_move_together():
    rng = np.random.default_rng(3)
    rows = []
    for index in range(60):
        accuracy = float(rng.uniform(60, 95))
        rows.append(
            {
                "game_id": f"g{index}",
                "username": f"p{index}",
                "accuracy": round(accuracy, 3),
                # a different scale that rises with the same underlying strength
                "reported_accuracy": round(accuracy * 0.8 + 10 + rng.normal(scale=0.5), 3),
            }
        )
    result = validate_against_reported(build_estimation_sample(frame(rows)))
    assert result["n"] == 60
    assert result["pearson_r"] > 0.95
    assert result["spearman_rho"] > 0.95


def test_the_slope_behind_the_rating_conversion_is_reported_with_it():
    """The conversion's denominator has to be inspectable: it is what makes the
    rating-point figure fragile."""
    from chess_elo_drift.analysis.statistics import accuracy_slope_per_100

    slope = accuracy_slope_per_100(planted_sample(-3.0, slope_per_100=1.5), "blitz")
    assert slope is not None
    estimate, error = slope
    assert estimate == pytest.approx(1.5, abs=0.1)
    assert 0 < error < 0.1


def test_no_slope_is_offered_on_a_sample_too_small_to_fit():
    from chess_elo_drift.analysis.statistics import accuracy_slope_per_100

    rows = [
        {"game_id": f"g{i}", "username": f"p{i}", "era": LEGACY if i % 2 else MODERN, "rating": 1000 + i}
        for i in range(10)
    ]
    assert accuracy_slope_per_100(build_estimation_sample(frame(rows)), "blitz") is None
