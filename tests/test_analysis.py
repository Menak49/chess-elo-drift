"""The analysis must recover effects planted in synthetic data, and say nothing when there are none.

The generators here build engine output, rating snapshots and conversion
surveys where every effect is known by construction: a year trend, a site
offset, a clock effect, a conversion line. Each test checks one of them comes
back out, or that a confound the design is meant to remove stays removed.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chess_elo_drift import config
from chess_elo_drift.analysis import conversion, sites, yearly
from chess_elo_drift.analysis.dataset import (
    apply_inclusion_rules,
    build_estimation_sample,
    coverage_of_reported_accuracy,
    report_band,
)
from chess_elo_drift.analysis.statistics import ols_clustered, validate_against_reported
from chess_elo_drift.analysis.surveys import (
    POOLS,
    established_chesscom,
    established_lichess,
    load_conversion_inputs,
    read_jsonl,
)
from chess_elo_drift.records import estimated_duration_seconds

# -- synthetic engine output -------------------------------------------------


def row(**overrides):
    """One engine-output row, with every column the analysis reads."""
    payload = {
        "game_id": "g1",
        "platform": "chesscom",
        "year": 2026,
        "month": "2026-03",
        "time_class": "blitz",
        "time_control": "300",
        "estimated_seconds": 300,
        "colour": "white",
        "username": "alice",
        "rating": 1500,
        "provisional": None,
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


#: Time controls by site, cadence and period: chess.com blitz loses 10|0 to
#: rapid after 2020, as it did in reality, so the clock mix shifts over time.
CONTROLS = {
    ("chesscom", "blitz", "before"): ("180", "180+2", "300", "300+5", "600"),
    ("chesscom", "blitz", "after"): ("180", "180+2", "300", "300+5"),
    ("chesscom", "rapid", "before"): ("900+10", "1800", "1500+10"),
    ("chesscom", "rapid", "after"): ("600", "600+5", "900+10", "1800"),
    ("lichess", "blitz", "before"): ("180", "180+2", "300", "300+3"),
    ("lichess", "blitz", "after"): ("180", "180+2", "300", "300+3"),
    ("lichess", "rapid", "before"): ("600", "600+5", "900+10"),
    ("lichess", "rapid", "after"): ("600", "600+5", "900+10"),
}

#: How each site's rating relates to a chess.com-scale strength, per cadence.
IDENTITY = (0.0, 1.0)


def synthetic_evaluations(
    *,
    years=config.YEARS,
    platforms=config.PLATFORMS,
    time_classes=config.TIME_CLASSES,
    per_cell: int = 160,
    trend_per_year: float = 0.0,
    slope_per_100: float = 0.6,
    slope_drift_per_year: float = 0.0,
    clock_effect: float = 2.5,
    site_maps: dict | None = None,
    controls: dict | None = None,
    player_sd: float = 1.0,
    noise_sd: float = 2.5,
    players_per_pool: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Engine output with a known accuracy law.

    accuracy = 80 + slope * (strength - 1500) / 100 + trend * (year - 2026)
               + clock_effect * log(seconds / reference) + player + noise

    where strength is the rating mapped back to the chess.com scale through
    `site_maps[(platform, time_class)] = (a, b)`, i.e. site rating = a + b * strength.
    """
    rng = np.random.default_rng(seed)
    site_maps = site_maps or {}
    controls = controls or CONTROLS
    player_effects = {}
    rows = []
    reference = {"blitz": 300, "rapid": 600}
    for platform in platforms:
        for time_class in time_classes:
            a, b = site_maps.get((platform, time_class), IDENTITY)
            for year in years:
                if not config.pool_exists(platform, time_class, config.YearMonth(year, 3)):
                    continue
                period = "before" if year <= 2020 else "after"
                options = controls[(platform, time_class, period)]
                for index in range(per_cell):
                    player = f"{time_class[0]}{int(rng.integers(players_per_pool))}"
                    key = (platform, player)
                    if key not in player_effects:
                        player_effects[key] = rng.normal(scale=player_sd)
                    rating = int(rng.integers(config.RATING_MIN, config.RATING_MAX + 1))
                    strength = (rating - a) / b
                    control = options[int(rng.integers(len(options)))]
                    seconds = estimated_duration_seconds(control)
                    slope = slope_per_100 + slope_drift_per_year * (year - 2026)
                    accuracy = (
                        80.0
                        + slope * (strength - 1500) / 100
                        + trend_per_year * (year - 2026)
                        + clock_effect * np.log(seconds / reference[time_class])
                        + player_effects[key]
                        + rng.normal(scale=noise_sd)
                    )
                    month = config.SAMPLED_MONTHS[index % len(config.SAMPLED_MONTHS)]
                    reported = None
                    if platform == "chesscom" and rng.random() < 0.02 + 0.015 * (year - 2014):
                        reported = round(accuracy * 0.9 - 3 + rng.normal(scale=2), 1)
                    rows.append(
                        {
                            "game_id": f"{platform}-{time_class}-{year}-{index}",
                            "platform": platform,
                            "year": year,
                            "month": f"{year}-{month:02d}",
                            "time_class": time_class,
                            "time_control": control,
                            "estimated_seconds": seconds,
                            "username": player,
                            "rating": rating,
                            "provisional": bool(rng.random() < 0.04) if platform == "lichess" else None,
                            "moves_scored": int(rng.integers(12, 50)),
                            "accuracy": round(float(np.clip(accuracy, 0, 100)), 3),
                            "acpl": round(float(max(5.0, 150 - 1.4 * accuracy + rng.normal(scale=4))), 2),
                            "reported_accuracy": reported,
                        }
                    )
    return frame(rows)


# -- synthetic conversion inputs ---------------------------------------------

#: Each pool as a line in one underlying strength T (chess.com rapid scale).
PLANTED_POOLS = {
    "chesscom_rapid": (0.0, 1.0),
    "chesscom_blitz": (-280.0, 1.10),
    "lichess_rapid": (650.0, 0.75),
    "lichess_blitz": (450.0, 0.85),
}


def planted_line(source: str, target: str) -> tuple[float, float]:
    """The exact conversion from `source` to `target` implied by PLANTED_POOLS."""
    a_s, b_s = PLANTED_POOLS[source]
    a_t, b_t = PLANTED_POOLS[target]
    return a_t - b_t * a_s / b_s, b_t / b_s


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_conversion_inputs(
    root: Path,
    *,
    people: int = 1200,
    linked_share: float = 0.5,
    rd: float = 50.0,
    preference_sd: float = 0.0,
    seed: int = 1,
) -> None:
    """Surveys, links and snapshots of one population, with known pool relations.

    Every observed rating is its pool's line in T plus a measurement error of SD
    `rd`, matching the RD the files report, so a Deming fit with the RD ratio is
    consistent. `preference_sd` adds a genuine per-person, per-pool deviation.
    """
    rng = np.random.default_rng(seed)
    strength = np.clip(rng.normal(1350, 350, people), 450, 2450)
    fetched = "2026-09-20T12:00:00Z"
    last_date = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
    seen_at = int(pd.Timestamp("2026-09-15", tz="UTC").timestamp() * 1000)

    def observed(pool: str) -> np.ndarray:
        a, b = PLANTED_POOLS[pool]
        return np.round(a + b * strength + rng.normal(0, rd, people) + rng.normal(0, preference_sd, people))

    ratings = {pool: observed(pool) for pool in PLANTED_POOLS}
    linked = rng.random(people) < linked_share

    chesscom, lichess, links = [], [], []
    for i in range(people):
        stale = rng.random() < 0.05
        chesscom.append(
            {
                **({"source": "chesscom_archive_opponent"} if i % 3 == 0 else {}),
                "username": f"cc{i}",
                "fetched_at": fetched,
                "found": bool(rng.random() > 0.02),
                "blitz": {"rating": ratings["chesscom_blitz"][i], "rd": rd, "last_date": last_date - (400 * 86400 if stale else 0), "games": 300, "best": None},
                "rapid": {"rating": ratings["chesscom_rapid"][i], "rd": rd if rng.random() > 0.05 else 250, "last_date": last_date, "games": 150 if rng.random() > 0.05 else 4, "best": None},
            }
        )
        lichess.append(
            {
                "username": f"li{i}",
                "fetched_at": fetched,
                "created_at": 1500000000000,
                "seen_at": seen_at,
                "title": None,
                "disabled": bool(rng.random() < 0.02),
                "tos_violation": bool(rng.random() < 0.02),
                "blitz": {"rating": ratings["lichess_blitz"][i], "rd": rd, "games": 400, "provisional": False},
                "rapid": {"rating": ratings["lichess_rapid"][i], "rd": rd, "games": 120, "provisional": bool(rng.random() < 0.05)},
                "chesscom_link": f"https://www.chess.com/member/cc{i}" if linked[i] else None,
                "source": "crawl",
            }
        )
        if linked[i]:
            links.append(
                {
                    "lichess_username": f"li{i}",
                    "chesscom_username": f"cc{i}",
                    "fetched_at": fetched,
                    "chesscom_found": True,
                    "lichess": {"blitz": ratings["lichess_blitz"][i], "rapid": ratings["lichess_rapid"][i], "seen_at": seen_at},
                    "chesscom": {"blitz": ratings["chesscom_blitz"][i], "rapid": ratings["chesscom_rapid"][i]},
                }
            )
    write_jsonl(root / "conversion" / "chesscom_stats.jsonl", chesscom)
    write_jsonl(root / "conversion" / "lichess_users.jsonl", lichess)
    write_jsonl(root / "conversion" / "linked_accounts.jsonl", links)

    for platform, prefix in (("chesscom", "cc"), ("lichess", "li")):
        snapshots = []
        for i in range(0, people, 2):
            snapshots.append(
                {
                    "platform": platform,
                    "username": f"{prefix}{i}",
                    "month": "2026-06",
                    "blitz_rating": int(ratings[f"{platform}_blitz"][i] + rng.normal(0, 20)),
                    "blitz_games": int(rng.integers(0, 40)),
                    "rapid_rating": int(ratings[f"{platform}_rapid"][i] + rng.normal(0, 20)),
                    "rapid_games": int(rng.integers(0, 30)),
                }
            )
        write_jsonl(root / platform / "rating_snapshots.jsonl", snapshots)


# -- inclusion rules ---------------------------------------------------------


def test_sample_drops_sides_the_engine_could_not_score_or_barely_scored():
    thin = config.MIN_MOVES_SCORED_PER_SIDE - 1
    sample = build_estimation_sample(
        frame([{"game_id": "g1", "accuracy": None}, {"game_id": "g2", "moves_scored": thin}, {"game_id": "g3"}])
    )
    assert list(sample["game_id"]) == ["g3"]


@pytest.mark.parametrize("rating", [config.RATING_MIN - 1, config.RATING_MAX + 1])
def test_sample_drops_ratings_outside_the_study_window(rating):
    assert build_estimation_sample(frame([{"rating": rating}])).empty


def test_lichess_rapid_counts_only_once_the_pool_existed():
    rows = [
        {"game_id": "old", "platform": "lichess", "time_class": "rapid", "year": 2017, "month": "2017-09", "time_control": "600"},
        {"game_id": "new", "platform": "lichess", "time_class": "rapid", "year": 2018, "month": "2018-03", "time_control": "600"},
        {"game_id": "cc", "platform": "chesscom", "time_class": "rapid", "year": 2016, "month": "2016-03", "time_control": "900+10"},
    ]
    assert set(build_estimation_sample(frame(rows))["game_id"]) == {"new", "cc"}


def test_provisional_lichess_ratings_are_dropped_and_counted():
    rows = [
        {"game_id": "p", "platform": "lichess", "provisional": True},
        {"game_id": "s", "platform": "lichess", "provisional": False},
        {"game_id": "c", "platform": "chesscom", "provisional": None},
    ]
    sample, attrition = apply_inclusion_rules(frame(rows))
    assert set(sample["game_id"]) == {"s", "c"}
    removed = attrition.set_index("rule")["rows_removed"]
    assert removed["Lichess rating not provisional"] == 1


def test_the_cap_applies_per_site_year_and_cadence():
    rows = [
        {"game_id": f"{p}-{y}-{tc}-{i}", "username": "prolific", "platform": p, "year": y, "month": f"{y}-03", "time_class": tc}
        for p in ("chesscom", "lichess")
        for y in (2019, 2026)
        for tc in ("blitz", "rapid")
        for i in range(5)
    ]
    sample = build_estimation_sample(frame(rows), max_per_player=2)
    assert len(sample) == 16
    # The same username on two sites is two different players.
    assert sample["player_id"].nunique() == 2


def test_the_cap_keeps_the_same_rows_whatever_the_collection_order():
    rows = [{"game_id": f"g{i}", "username": "prolific"} for i in range(10)]
    forward = build_estimation_sample(frame(rows), max_per_player=3)
    backward = build_estimation_sample(frame(list(reversed(rows))), max_per_player=3)
    assert list(forward["game_id"]) == list(backward["game_id"])


def test_the_clock_is_derived_from_the_time_control_when_missing():
    sample = build_estimation_sample(frame([{"estimated_seconds": None, "time_control": "180+2"}]))
    assert sample["estimated_seconds"].iloc[0] == 260
    assert sample["log_clock"].iloc[0] == pytest.approx(np.log(260 / 300))


def test_filling_the_clock_keeps_its_values_and_raises_no_warning():
    """Nothing missing is the common case, and it must not trip pandas' dtype warning."""
    from chess_elo_drift.analysis.dataset import _normalise

    complete = frame([{"estimated_seconds": 300, "time_control": "300"}, {"estimated_seconds": 260, "time_control": "180+2"}])
    mixed = frame(
        [
            {"estimated_seconds": None, "time_control": "180+2"},
            {"estimated_seconds": 300, "time_control": "600"},
            {"estimated_seconds": None, "time_control": "1/86400"},
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        untouched, filled = _normalise(complete), _normalise(mixed)
    assert list(untouched["estimated_seconds"]) == [300, 260]
    assert filled["estimated_seconds"].iloc[0] == 260
    assert filled["estimated_seconds"].iloc[1] == 300  # a recorded value wins over the control
    assert np.isnan(filled["estimated_seconds"].iloc[2])  # a daily control has no clock


def test_attrition_accounts_for_every_row():
    evaluations = synthetic_evaluations(years=(2025, 2026), per_cell=40)
    sample, attrition = apply_inclusion_rules(evaluations)
    assert attrition["rows_left"].iloc[-1] == len(sample)
    assert attrition["rows_removed"].sum() == len(evaluations) - len(sample)


@pytest.mark.parametrize(("rating", "expected"), [(600, 600), (899, 600), (900, 900), (2099, 1800), (2100, 2100), (2400, 2100)])
def test_report_bands_are_300_wide_and_the_top_joins_the_last_band(rating, expected):
    assert report_band(pd.Series([rating])).iloc[0] == expected


def test_reported_accuracy_coverage_is_per_year_and_cadence_on_chesscom_only():
    rows = [
        {"game_id": "a", "year": 2026, "reported_accuracy": 80.0},
        {"game_id": "b", "year": 2026, "reported_accuracy": None},
        {"game_id": "c", "year": 2015, "month": "2015-03", "reported_accuracy": None},
        {"game_id": "d", "platform": "lichess", "year": 2026},
    ]
    coverage = coverage_of_reported_accuracy(frame(rows)).set_index("year")["coverage"]
    assert coverage[2026] == pytest.approx(0.5)
    assert coverage[2015] == pytest.approx(0.0)
    assert coverage_of_reported_accuracy(frame(rows))["observations"].sum() == 3


# -- least squares with clustered errors -------------------------------------


def test_least_squares_recovers_the_coefficients_it_was_given():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    design = np.column_stack([np.ones(400), x])
    outcome = 3.0 + 2.0 * x + rng.normal(scale=0.01, size=400)
    fit = ols_clustered(design, outcome, np.arange(400))
    assert fit.coefficients == pytest.approx([3.0, 2.0], abs=0.01)
    assert fit.clusters == 400


def test_clustering_widens_the_error_bar_when_rows_repeat_a_player():
    """Twenty players measured twenty times each is not four hundred observations."""
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 20)
    offsets = rng.normal(scale=5.0, size=20)[groups]
    design = np.column_stack([np.ones(400), rng.normal(size=400)])
    outcome = 50.0 + offsets + rng.normal(scale=0.5, size=400)

    clustered = ols_clustered(design, outcome, groups)
    independent = ols_clustered(design, outcome, np.arange(400))
    assert clustered.standard_errors[0] > 3 * independent.standard_errors[0]


def test_a_wald_test_rejects_a_planted_difference_and_not_an_absent_one():
    rng = np.random.default_rng(2)
    group = rng.integers(0, 3, 900)
    design = np.column_stack([np.ones(900), group == 1, group == 2]).astype(float)
    restrictions = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
    planted = ols_clustered(design, design @ [10, 0.8, 0] + rng.normal(size=900), np.arange(900))
    absent = ols_clustered(design, design @ [10, 0, 0] + rng.normal(size=900), np.arange(900))
    assert planted.wald_p_value(restrictions) < 0.001
    assert absent.wald_p_value(restrictions) > 0.01


# -- year by year -----------------------------------------------------------


@pytest.fixture(scope="module")
def trending_sample():
    """A planted decline of 1.5 accuracy points over 2014-2026 at a fixed rating."""
    evaluations = synthetic_evaluations(trend_per_year=-0.125, platforms=("chesscom",), per_cell=220, seed=3)
    return build_estimation_sample(evaluations)


def test_the_trend_recovers_a_planted_decline(trending_sample):
    trends = yearly.trends(trending_sample, (yearly.MAIN,))
    for row in trends.itertuples():
        assert row.trend_per_year == pytest.approx(-0.125, abs=0.04)
        assert row.ci_low < -0.125 < row.ci_high
        assert row.p_value < 0.001


def test_each_year_is_measured_against_the_latest_year(trending_sample):
    effects = yearly.year_effects(trending_sample)
    blitz = effects[effects["time_class"] == "blitz"].set_index("year")
    assert (blitz["reference_year"] == 2026).all()
    assert blitz.loc[2026, "effect"] == 0.0
    assert blitz.loc[2014, "effect"] == pytest.approx(1.5, abs=0.6)
    # Worth 1.5 / 0.6 * 100 = 250 rating points, as an order of magnitude.
    assert 120 < blitz.loc[2014, "rating_points_equivalent"] < 450


def test_accuracy_at_fixed_ratings_follows_the_planted_law(trending_sample):
    levels = yearly.accuracy_at_ratings(trending_sample)
    rapid_2026 = levels[(levels["time_class"] == "rapid") & (levels["year"] == 2026)].set_index("rating")
    assert rapid_2026.loc[1500, "accuracy"] == pytest.approx(80.0, abs=1.0)
    assert rapid_2026.loc[2000, "accuracy"] - rapid_2026.loc[1000, "accuracy"] == pytest.approx(6.0, abs=1.5)
    assert (rapid_2026["ci_low"] < rapid_2026["accuracy"]).all()


def test_no_trend_is_reported_when_none_was_planted():
    sample = build_estimation_sample(synthetic_evaluations(platforms=("lichess",), time_classes=("blitz",), seed=4))
    main = yearly.trends(sample, (yearly.MAIN,)).iloc[0]
    assert main.ci_low < 0 < main.ci_high
    assert yearly.verdict(main) == "no detectable change"


def test_the_clock_control_removes_a_shift_in_the_time_control_mix():
    """Longer games in later years, and no change in skill at all.

    Without the control the longer clock is read as players getting better at
    the same rating; with it the trend must vanish.
    """
    controls = {
        ("chesscom", "blitz", "before"): ("180", "180", "300"),
        ("chesscom", "blitz", "after"): ("300", "600", "600"),
    }
    evaluations = synthetic_evaluations(
        platforms=("chesscom",), time_classes=("blitz",), controls=controls, clock_effect=4.0, per_cell=200, seed=5
    )
    sample = build_estimation_sample(evaluations)
    no_clock = next(s for s in yearly.SPECIFICATIONS if s.name == "no_clock")
    trends = yearly.trends(sample, (yearly.MAIN, no_clock)).set_index("specification")
    assert abs(trends.loc["main", "trend_per_year"]) < 0.06
    assert trends.loc["no_clock", "trend_per_year"] > 0.25


def test_the_month_control_removes_a_year_sampled_from_one_season():
    """Every June is 3 points kinder, and the later years were read in June only.

    That is what a snowball stuck in one month produces: without the control
    the season is read as a trend, with it the trend must vanish.
    """
    evaluations = synthetic_evaluations(platforms=("lichess",), time_classes=("blitz",), per_cell=200, seed=9)
    late = evaluations["year"] >= 2021
    evaluations.loc[late, "month"] = evaluations.loc[late, "year"].astype(str) + "-06"
    evaluations.loc[evaluations["month"].str.endswith("-06"), "accuracy"] += 3.0
    sample = build_estimation_sample(evaluations)
    no_seasons = next(s for s in yearly.SPECIFICATIONS if s.name == "no_seasons")
    trends = yearly.trends(sample, (yearly.MAIN, no_seasons)).set_index("specification")
    assert abs(trends.loc["main", "trend_per_year"]) < 0.06
    assert trends.loc["no_seasons", "trend_per_year"] > 0.1


def test_years_each_read_in_one_month_still_give_sane_levels():
    """Month terms that only restate the years must be dropped, not solved for."""
    evaluations = synthetic_evaluations(platforms=("lichess",), time_classes=("rapid",), per_cell=200, seed=11)
    single = {year: config.SAMPLED_MONTHS[year % 3] for year in config.YEARS}
    evaluations["month"] = [f"{year}-{single[year]:02d}" for year in evaluations["year"]]
    levels = yearly.accuracy_at_ratings(build_estimation_sample(evaluations))
    at_1500 = levels[levels["rating"] == 1500]["accuracy"]
    assert at_1500.between(78.5, 81.5).all()


def test_a_few_games_from_a_second_month_do_not_switch_the_control_on():
    """Seven March games in a September year once set the September effect alone."""
    months = ["2015-09"] * 300 + ["2015-03"] * 7 + ["2016-09"] * 300 + ["2017-03"] * 300
    cell = pd.DataFrame({"month": months, "year": [int(m[:4]) for m in months]})
    assert yearly._season_columns(cell) == {}
    balanced = pd.DataFrame(
        {"month": [f"{y}-{m:02d}" for y in (2015, 2016) for m in (3, 6, 9) for _ in range(40)]}
    )
    balanced["year"] = balanced["month"].str[:4].astype(int)
    assert set(yearly._season_columns(balanced)) == {"month_06", "month_09"}


def test_the_month_control_predicts_the_average_month_not_the_first():
    evaluations = synthetic_evaluations(platforms=("chesscom",), time_classes=("rapid",), per_cell=240, seed=10)
    evaluations.loc[evaluations["month"].str.endswith("-09"), "accuracy"] += 3.0
    levels = yearly.accuracy_at_ratings(build_estimation_sample(evaluations))
    at_1500 = levels[levels["rating"] == 1500]["accuracy"]
    assert at_1500.mean() == pytest.approx(81.0, abs=0.5)


def test_a_drifting_rating_slope_is_detected_and_a_stable_one_is_not():
    drifting = build_estimation_sample(
        synthetic_evaluations(platforms=("chesscom",), time_classes=("rapid",), slope_drift_per_year=0.06, per_cell=220, seed=6)
    )
    stable = build_estimation_sample(
        synthetic_evaluations(platforms=("chesscom",), time_classes=("rapid",), per_cell=220, seed=7)
    )
    drift_row = yearly.slope_drift(drifting).iloc[0]
    assert drift_row.slope_change_per_year == pytest.approx(0.06, abs=0.03)
    assert drift_row.equal_slopes_p_value < 0.01
    assert yearly.slope_drift(stable).iloc[0].p_value > 0.01


def test_robustness_specifications_that_only_make_sense_on_chesscom_skip_lichess():
    sample = build_estimation_sample(synthetic_evaluations(per_cell=60, seed=8))
    trends = yearly.trends(sample)
    chesscom_only = {"stable_controls", "before_break", "after_break"}
    assert chesscom_only & set(trends.loc[trends["platform"] == "lichess", "specification"]) == set()
    assert chesscom_only <= set(trends.loc[trends["platform"] == "chesscom", "specification"])
    segment = trends[trends["specification"] == "after_break"].iloc[0]
    assert segment.first_year == yearly.FIRST_YEAR_AFTER_BREAK


def test_years_too_thin_to_estimate_are_left_out():
    evaluations = synthetic_evaluations(years=(2024, 2025, 2026), platforms=("chesscom",), time_classes=("blitz",), per_cell=80)
    thin = evaluations[(evaluations["year"] != 2024) | (evaluations.index % 10 == 0)]
    effects = yearly.year_effects(build_estimation_sample(thin))
    assert set(effects["year"]) == {2025, 2026}


def test_a_single_year_gives_no_year_model():
    sample = build_estimation_sample(synthetic_evaluations(years=(2026,), per_cell=80))
    assert yearly.year_effects(sample).empty
    assert yearly.accuracy_at_ratings(sample).empty


# -- the site comparison ----------------------------------------------------


def test_equating_inverts_a_known_curve():
    source = np.array([80.0, 0.6, 0.0, 0.0])
    target = np.array([80.0 - 1.8, 0.6, 0.0, 0.0])  # the same curve, 300 points later
    matched = sites.equate(source, target, np.array([1000.0, 1500.0, 2000.0]), (600.0, 2400.0))
    assert matched == pytest.approx([1300, 1800, 2300], abs=1)


def test_equating_refuses_to_extrapolate_beyond_the_sampled_ratings():
    source = np.array([80.0, 0.6, 0.0, 0.0])
    target = np.array([78.2, 0.6, 0.0, 0.0])
    matched = sites.equate(source, target, np.array([2300.0]), (600.0, 2400.0))
    assert np.isnan(matched[0])


def test_a_planted_site_offset_is_recovered_with_an_honest_interval():
    maps = {("lichess", "blitz"): (300.0, 1.0)}
    evaluations = synthetic_evaluations(
        years=(2025, 2026), time_classes=("blitz",), site_maps=maps, per_cell=400, noise_sd=2.0, seed=9
    )
    offsets = sites.site_offsets(build_estimation_sample(evaluations), replicates=120, seed=1)
    at = sites.offsets_at(offsets)
    at = at[at["year"] == 2026].set_index("rating")
    for rating in (1000, 1500, 2000):
        assert at.loc[rating, "offset"] == pytest.approx(300, abs=90)
        assert at.loc[rating, "offset_ci_low"] < 300 < at.loc[rating, "offset_ci_high"]
    reverse = offsets[(offsets["source"] == "lichess") & (offsets["year"] == 2026) & (offsets["rating"] == 1800)]
    assert reverse["equivalent"].iloc[0] == pytest.approx(1500, abs=90)


def test_a_column_where_every_replicate_failed_is_nan_without_a_warning():
    draws = np.array([[1000.0, np.nan], [1100.0, np.nan], [1200.0, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        low, high = sites._percentile_interval(draws)
    assert np.isnan(low[1]) and np.isnan(high[1])
    assert 1000 <= low[0] < high[0] <= 1200


def test_thin_site_years_are_named_with_what_each_site_had():
    evaluations = synthetic_evaluations(years=(2025, 2026), time_classes=("blitz",), per_cell=100, seed=5)
    thin_lichess = evaluations[(evaluations["platform"] != "lichess") | (evaluations["year"] != 2025) | (evaluations.index % 10 == 0)]
    sample = build_estimation_sample(thin_lichess)
    skipped = sites.skipped_cells(sample)
    assert list(skipped["year"]) == [2025] and list(skipped["time_class"]) == ["blitz"]
    assert skipped["lichess"].iloc[0] < sites.MIN_SITE_OBSERVATIONS <= skipped["chesscom"].iloc[0]
    assert sites.skipped_cells(build_estimation_sample(evaluations)).empty


def test_no_offset_is_offered_when_one_site_is_missing():
    evaluations = synthetic_evaluations(years=(2026,), platforms=("chesscom",), per_cell=100)
    assert sites.site_offsets(build_estimation_sample(evaluations), replicates=10).empty


# -- conversion: the line ---------------------------------------------------


def test_deming_recovers_a_line_that_ols_attenuates():
    rng = np.random.default_rng(10)
    truth = rng.normal(1400, 250, 3000)
    x = truth + rng.normal(0, 80, truth.size)
    y = 250 + 0.9 * truth + rng.normal(0, 80, truth.size)
    fit = conversion.deming(x, y, delta=1.0)
    assert fit.slope == pytest.approx(0.9, abs=0.03)
    assert fit.intercept == pytest.approx(250, abs=45)
    assert conversion.ols(x, y).slope < 0.85


def test_deming_is_its_own_inverse():
    rng = np.random.default_rng(11)
    x = rng.normal(1500, 300, 500)
    y = 100 + 1.1 * x + rng.normal(0, 60, 500)
    forward = conversion.deming(x, y, delta=2.0)
    backward = conversion.deming(y, x, delta=0.5)
    assert backward.slope == pytest.approx(forward.inverse().slope)
    assert backward.intercept == pytest.approx(forward.inverse().intercept)


def test_curvature_is_flagged_only_when_the_relation_bends():
    rng = np.random.default_rng(12)
    x = rng.uniform(700, 2300, 4000)
    straight = conversion.curvature(x, 250 + 0.9 * x + rng.normal(0, 40, x.size))
    bent = conversion.curvature(x, 250 + 0.9 * x + 0.0006 * (x - 1500) ** 2 + rng.normal(0, 40, x.size))
    assert straight.adequate
    assert not bent.adequate and bent.max_deviation > conversion.CURVATURE_TOLERANCE


def test_the_outlier_screen_drops_wild_pairs_and_nothing_else():
    rng = np.random.default_rng(15)
    truth = rng.normal(1400, 300, 400)
    x = truth + rng.normal(0, 60, truth.size)
    y = 300 + 0.8 * truth + rng.normal(0, 60, truth.size)
    assert conversion.screen_outliers(x, y).sum() >= 399  # clean data: at most a stray pair

    wild_x = np.concatenate([x, [900, 1000, 2100, 2200, 1200]])
    wild_y = np.concatenate([y, [2000, 2100, 1100, 1200, 2300]])  # 900+ points off the line
    keep = conversion.screen_outliers(wild_x, wild_y)
    assert not keep[-5:].any()
    assert keep[:-5].sum() >= 399


def test_a_fit_counts_and_excludes_the_pairs_it_screened(tmp_path):
    rng = np.random.default_rng(16)
    truth = rng.normal(1400, 300, 300)
    people = pd.DataFrame(
        {
            "person": range(305),
            "chesscom_rapid": np.concatenate([truth + rng.normal(0, 50, 300), [900, 1000, 2100, 2200, 1200]]),
            "lichess_rapid": np.concatenate([650 + 0.75 * truth + rng.normal(0, 50, 300), [2300, 2350, 900, 950, 2400]]),
        }
    )
    fit = conversion.fit_pair(people, POOLS[0], POOLS[2], "linked accounts", replicates=50)
    assert fit.screened_out == 5 and fit.n == 300
    assert fit.line.slope == pytest.approx(0.75, abs=0.05)
    assert conversion.formulas_table(conversion.ConversionResults(fits={("a", "b"): fit}))["outliers_dropped"].iloc[0] == 5


def test_equipercentile_follows_a_bend_that_a_straight_line_misses():
    rng = np.random.default_rng(17)
    truth = rng.normal(1450, 330, 4000)

    def bent(t):
        return 1350 + 0.95 * (t - 1450) + 0.00045 * (t - 1450) ** 2

    x = truth + rng.normal(0, 30, truth.size)
    y = bent(truth) + rng.normal(0, 30 * 0.95, truth.size)
    link, line = conversion.equipercentile(x, y), conversion.linear_equating(x, y)
    grid = np.array([900.0, 1450.0, 2000.0])
    assert link(grid) == pytest.approx(bent(grid), abs=35)
    assert np.max(np.abs(line(grid) - bent(grid))) > 90
    assert not conversion.curvature(x, y).adequate


def test_the_equipercentile_link_inverts_exactly_even_in_its_tails():
    rng = np.random.default_rng(18)
    x = rng.normal(1500, 300, 800)
    y = 200 + 0.9 * x + rng.normal(0, 80, 800)
    link = conversion.equipercentile(x, y)
    ratings = np.array([400.0, 800.0, 1500.0, 2200.0, 2700.0])
    assert link.inverse()(link(ratings)) == pytest.approx(ratings)
    assert np.all(np.diff(link(np.linspace(300, 2800, 500))) > 0)


def test_formula_and_table_agree_where_the_relation_is_straight(conversion_results):
    formulas = conversion.formulas_table(conversion_results)
    assert (formulas["formula_vs_table_max_gap"] < 40).all()
    assert (formulas["valid_from"] < formulas["valid_to"]).all()


def test_the_rounded_formula_reads_like_the_line_it_rounds():
    line = conversion.Line(412.3, 0.9312)
    text = conversion.rounded_formula(line, POOLS[0], POOLS[3])
    assert text == "Lichess blitz ≈ 414 + 0.93 × chess.com rapid"


# -- conversion: the inputs -------------------------------------------------


def test_only_established_ratings_are_kept():
    fetched = pd.Timestamp("2026-09-30", tz="UTC")
    recent = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
    stale = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp())
    good = {"rating": 1500, "rd": 60, "games": 50, "last_date": recent}
    assert established_chesscom(good, fetched) == (1500, 60)
    assert established_chesscom({**good, "rd": 200}, fetched) is None
    assert established_chesscom({**good, "games": 5}, fetched) is None
    assert established_chesscom({**good, "last_date": stale}, fetched) is None
    assert established_lichess({"rating": 1800, "rd": 70, "games": 40, "provisional": True}) is None
    assert established_lichess({"rating": 1800, "rd": 70, "games": 40, "provisional": False}) == (1800, 70)


def test_a_half_written_survey_line_is_skipped(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text('{"username": "a"}\n{"username": "b", "blit', encoding="utf-8")
    assert read_jsonl(path) == [{"username": "a"}]
    assert read_jsonl(tmp_path / "absent.jsonl") == []


def test_missing_conversion_files_give_empty_sources_and_no_fits(tmp_path):
    inputs = load_conversion_inputs(tmp_path)
    assert all(not log.available for log in inputs.logs)
    results = conversion.conversion_study(inputs, replicates=10)
    assert results.fits == {}
    assert len(results.notes) == len(conversion.PAIRS)
    assert conversion.conversion_table(results).empty
    assert conversion.chain_table(results).empty


def test_links_declared_only_on_a_lichess_profile_are_used(tmp_path):
    write_conversion_inputs(tmp_path, people=200, seed=13)
    (tmp_path / "conversion" / "linked_accounts.jsonl").unlink()
    inputs = load_conversion_inputs(tmp_path)
    assert len(inputs.linked) > 50


# -- conversion: the whole study --------------------------------------------


@pytest.fixture(scope="module")
def conversion_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw")
    write_conversion_inputs(root, people=1500, seed=14)
    return conversion.conversion_study(load_conversion_inputs(root), replicates=200)


@pytest.mark.parametrize(("x_pool", "y_pool"), conversion.PAIRS, ids=lambda pool: pool.key)
def test_every_planted_conversion_is_recovered(conversion_results, x_pool, y_pool):
    fit = conversion_results.fits[(x_pool.key, y_pool.key)]
    intercept, slope = planted_line(x_pool.key, y_pool.key)
    assert fit.line.slope == pytest.approx(slope, abs=0.04)
    for rating in (1000, 1500, 2000):
        assert fit.line(rating) == pytest.approx(intercept + slope * rating, abs=30)
    assert fit.curvature.adequate


def test_within_site_pairs_prefer_the_survey_and_keep_the_rest_as_checks(conversion_results):
    within = conversion_results.fits[(POOLS[0].key, POOLS[1].key)]
    assert within.source == "current-rating survey"
    agreement = conversion.source_agreement(conversion_results)
    assert {"linked accounts"} <= set(agreement["check_source"])
    assert agreement["difference"].abs().max() < 60


def test_ols_is_pulled_toward_the_mean_and_the_formula_is_not(conversion_results):
    fit = conversion_results.fits[(POOLS[0].key, POOLS[3].key)]
    _, planted_slope = planted_line(POOLS[0].key, POOLS[3].key)
    assert fit.ols_y_on_x.slope < fit.line.slope
    assert abs(fit.line.slope - planted_slope) < abs(fit.ols_y_on_x.slope - planted_slope)


def test_the_conversion_table_runs_both_ways_and_round_trips(conversion_results):
    table = conversion.conversion_table(conversion_results)
    assert set(table["from"]) == {pool.label for pool in POOLS}
    assert len(table) == 12 * len(conversion.TABLE_RATINGS)
    forward = table[(table["from"] == "chess.com rapid") & (table["to"] == "Lichess blitz") & (table["rating"] == 1400)]
    value = float(forward["equivalent"].iloc[0])
    fit, is_forward = conversion_results.resolve(POOLS[3], POOLS[0])
    assert float(fit.published(is_forward)(value)) == pytest.approx(1400, abs=1e-6)
    assert (table["ci_low"] <= table["equivalent"]).all() and (table["equivalent"] <= table["ci_high"]).all()


def test_chained_conversions_agree_with_direct_ones_when_the_pools_are_consistent(conversion_results):
    chains = conversion.chain_table(conversion_results)
    assert not chains.empty
    assert chains["discrepancy"].abs().max() < 40


def test_the_benchmark_is_compared_where_this_study_has_the_pair(conversion_results):
    benchmark = conversion.benchmark_comparison(conversion_results)
    assert len(benchmark) == 9
    assert benchmark["this_study"].notna().all()


def test_the_accuracy_cross_check_lines_up_both_estimates(conversion_results):
    offsets = pd.DataFrame(
        {
            "year": [config.CONVERSION_YEAR], "time_class": ["blitz"], "source": ["chesscom"], "target": ["lichess"],
            "rating": [1500], "equivalent": [1800.0], "ci_low": [1700.0], "ci_high": [1900.0],
        }
    )
    check = conversion.accuracy_cross_check(conversion_results, offsets)
    blitz = check[(check["time_class"] == "blitz") & (check["chesscom_rating"] == 1500)].iloc[0]
    expected = planted_line("chesscom_blitz", "lichess_blitz")
    assert blitz.linked_accounts == pytest.approx(expected[0] + expected[1] * 1500, abs=30)
    assert blitz.difference == pytest.approx(1800.0 - blitz.linked_accounts)


# -- the instrument ---------------------------------------------------------


def test_validation_needs_enough_overlap_to_say_anything():
    rows = [{"game_id": f"g{i}", "username": f"p{i}", "reported_accuracy": 70.0} for i in range(5)]
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
                "reported_accuracy": round(accuracy * 0.8 + 10 + rng.normal(scale=0.5), 3),
            }
        )
    result = validate_against_reported(build_estimation_sample(frame(rows)))
    assert result["n"] == 60
    assert result["pearson_r"] > 0.95



def test_no_hub_contributes_more_than_its_cap():
    """Opponents of one rapid-heavy account are chosen by their rapid rating;
    letting a single hub flood the survey would bias the conversion."""
    from chess_elo_drift.analysis.surveys import cap_per_hub

    people = pd.DataFrame(
        {"person": [f"chesscom:p{i}" for i in range(30)] + ["chesscom:loner"], "chesscom_rapid": 1500.0}
    )
    hubs = {f"p{i}": "bighub" for i in range(30)}

    kept, dropped = cap_per_hub(people, hubs, max_per_hub=12)

    assert dropped == 18
    assert "chesscom:loner" in set(kept["person"])  # no hub: its own
    reversed_kept, _ = cap_per_hub(people.iloc[::-1].reset_index(drop=True), hubs, max_per_hub=12)
    assert set(kept["person"]) == set(reversed_kept["person"])  # file order does not matter



def test_the_survey_fit_weighs_both_discovery_cadences_equally():
    """People met in rapid games are rapid regulars; an unbalanced mix would pull
    the conversion toward whichever cadence supplied more of them."""
    from chess_elo_drift.analysis.surveys import balance_by_cadence

    people = pd.DataFrame({"person": [f"chesscom:p{i}" for i in range(10)] + ["chesscom:untagged"]})
    cadences = {f"p{i}": ("blitz" if i < 7 else "rapid") for i in range(10)}

    kept, note = balance_by_cadence(people, cadences)

    names = kept["person"].str.split(":").str[1]
    assert names.map(cadences).value_counts().to_dict() == {"blitz": 3, "rapid": 3}
    assert "chesscom:untagged" not in set(kept["person"])
    assert note is not None


def test_without_recorded_cadences_the_survey_is_left_alone():
    from chess_elo_drift.analysis.surveys import balance_by_cadence

    people = pd.DataFrame({"person": ["chesscom:a", "chesscom:b"]})
    kept, note = balance_by_cadence(people, {})
    assert kept.equals(people) and note is None
