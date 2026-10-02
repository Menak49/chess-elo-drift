"""The report stage must produce every artefact it promises, on thin or missing inputs too.

These tests run the real pipeline end to end, from synthetic engine output and
synthetic surveys written to disk, into a temporary directory -- so they cover
the stage that actually writes the study's conclusions.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chess_elo_drift import config
from chess_elo_drift.analysis import report
from chess_elo_drift.analysis.report import generate_report

from tests.test_analysis import frame, synthetic_evaluations, write_conversion_inputs

EXPECTED_TABLES = {
    "sample_attrition", "sample_sizes", "year_effects", "accuracy_at_ratings", "year_trends",
    "slope_drift", "band_by_year", "site_offsets", "conversion_formulas", "conversion_table",
    "conversion_chains", "conversion_source_agreement", "conversion_accuracy_check",
    "conversion_benchmark", "conversion_sources", "reported_accuracy_coverage",
}
EXPECTED_FIGURES = {"accuracy_by_year", "trend_robustness", "site_offsets", "conversion", "reported_accuracy_coverage"}

FAST = {"site_replicates": 20, "conversion_replicates": 40}


def run(tmp_path, evaluations: pd.DataFrame | None, *, surveys: bool):
    raw = tmp_path / "raw"
    raw.mkdir()
    evaluations_path = tmp_path / "evaluations.csv"
    if evaluations is not None:
        evaluations.to_csv(evaluations_path, index=False)
    if surveys:
        write_conversion_inputs(raw, people=400, seed=2)
    return generate_report(
        evaluations_path=evaluations_path, max_per_player=3, data_raw=raw, output_root=tmp_path / "out", **FAST
    )


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("full")
    evaluations = synthetic_evaluations(
        years=(2016, 2020, 2024, 2026), per_cell=90, trend_per_year=-0.125,
        site_maps={("lichess", "blitz"): (300.0, 1.0), ("lichess", "rapid"): (300.0, 1.0)}, seed=21,
    )
    return run(tmp_path, evaluations, surveys=True)


def test_a_full_run_writes_every_artefact(full_run):
    assert full_run.findings.exists() and full_run.findings.stat().st_size > 0
    assert set(full_run.tables) == EXPECTED_TABLES
    assert set(full_run.figures) == EXPECTED_FIGURES
    for path in [*full_run.tables.values(), *full_run.figures.values()]:
        assert path.exists() and path.stat().st_size > 0


def test_the_findings_answer_all_three_questions_and_name_their_method(full_run):
    note = full_run.findings.read_text(encoding="utf-8")
    assert "### 1. Year by year" in note and "### 2. chess.com against Lichess" in note
    assert f"### 3. {config.CONVERSION_YEAR} conversion formulas" in note
    assert "Lichess blitz ≈" in note and "× chess.com rapid" in note
    assert "Deming" in note and "clustered by player" in note
    assert f"depth {config.ENGINE_DEPTH}" in note
    assert "not from this study's data" in note  # the outside benchmark is labelled as such
    assert "_(no rows)_" not in note.split("## How it was measured")[0]


def test_the_conversion_table_is_written_in_every_direction(full_run):
    table = pd.read_csv(full_run.tables["conversion_table"])
    assert table.groupby(["from", "to"]).ngroups == 12
    assert set(table["rating"]) == set(range(800, 2201, 200))


def test_missing_surveys_are_reported_not_fatal(tmp_path):
    evaluations = synthetic_evaluations(years=(2025, 2026), per_cell=80, seed=22)
    artifacts = run(tmp_path, evaluations, surveys=False)
    note = artifacts.findings.read_text(encoding="utf-8")
    assert "No conversion could be fitted" in note
    assert "**missing**" in note
    for path in [*artifacts.tables.values(), *artifacts.figures.values()]:
        assert path.exists() and path.stat().st_size > 0


def test_a_sample_too_thin_to_analyse_still_produces_a_report(tmp_path):
    """The state this project is in before the engine has scored much."""
    artifacts = run(tmp_path, frame([{"game_id": "g1", "username": "solo"}]), surveys=False)
    assert artifacts.findings.exists()
    for path in [*artifacts.tables.values(), *artifacts.figures.values()]:
        assert path.exists() and path.stat().st_size > 0


def test_no_evaluations_file_at_all_still_produces_a_report(tmp_path):
    artifacts = run(tmp_path, None, surveys=True)
    note = artifacts.findings.read_text(encoding="utf-8")
    assert "No evaluated game has passed the inclusion rules yet" in note
    # The conversion does not need the engine and still runs.
    assert "Lichess blitz ≈" in note


def offsets_frame(rows):
    """Offsets at chess.com 1500 as (time_class, year, offset, has_interval)."""
    return pd.DataFrame(
        [
            {
                "time_class": time_class, "year": year, "rating": 1500, "offset": offset,
                "offset_ci_low": offset - 100 if interval else np.nan,
                "offset_ci_high": offset + 100 if interval else np.nan,
            }
            for time_class, year, offset, interval in rows
        ]
    )


def test_the_offset_range_ignores_offsets_with_no_interval_and_says_so():
    at = offsets_frame(
        [("blitz", 2016, 50, True), ("blitz", 2017, 300, True), ("blitz", 2018, 1235, False), ("blitz", 2019, -20, True)]
    )
    (line,) = report._offset_spread_lines(at)
    assert "across 3 years with an interval" in line
    assert "from -20 (2019) to +300 (2017)" in line
    assert "1235" not in line
    assert "1 more year has an offset with no interval" in line


def test_a_cadence_with_no_interval_anywhere_is_not_given_a_range():
    at = offsets_frame([("rapid", 2018, 770, False), ("rapid", 2019, 450, False)])
    (line,) = report._offset_spread_lines(at)
    assert "fewer than two years" in line and "770" not in line


def test_the_equating_caveat_says_the_sites_disagree_and_points_to_the_linked_conversion():
    at = offsets_frame([("blitz", 2016, 50, True), ("blitz", 2017, 450, True), ("blitz", 2021, 500, True)])
    results = SimpleNamespace(
        accuracy_check=pd.DataFrame({"difference": [-510.0, -365.0, -205.0]}),
        effects=pd.DataFrame(
            {"platform": ["chesscom", "lichess"], "time_class": ["blitz", "blitz"], "slope_per_100": [0.43, 0.575]}
        ),
    )
    text = report._equating_caveat(results, at)
    assert "do not agree with the conversion fitted on people who hold accounts on both sites" in text
    assert "median of -365 rating points" in text
    assert "median of 400 points between consecutive years" in text  # 2016 -> 2017 only; 2017 -> 2021 is not consecutive
    assert "about 175-235 rating points" in text  # 100 / 0.575 and 100 / 0.43
    assert "use the linked-account conversion" in text


def test_the_equating_caveat_does_not_claim_a_disagreement_that_is_absent():
    results = SimpleNamespace(accuracy_check=pd.DataFrame({"difference": [10.0, -20.0]}), effects=pd.DataFrame())
    text = report._equating_caveat(results, offsets_frame([("blitz", 2016, 50, True)]))
    assert "do not agree" not in text
    assert "agree with the conversion" in text


def test_the_equating_caveat_is_silent_without_anything_to_say():
    results = SimpleNamespace(accuracy_check=pd.DataFrame(), effects=pd.DataFrame())
    assert report._equating_caveat(results, offsets_frame([("blitz", 2016, 50, True)])) == ""


def trend_row(specification, first_year, last_year, trend, low, high, p_value, platform="chesscom", time_class="rapid"):
    return {
        "platform": platform, "time_class": time_class, "specification": specification,
        "first_year": first_year, "last_year": last_year, "trend_per_year": trend,
        "ci_low": low, "ci_high": high, "p_value": p_value,
    }


def effect_row(year, effect, std_error, points, platform="chesscom", time_class="rapid"):
    return {
        "platform": platform, "time_class": time_class, "year": year, "reference_year": 2026, "effect": effect,
        "std_error": std_error, "ci_low": effect - 2 * std_error, "ci_high": effect + 2 * std_error, "p_value": 0.01,
        "rating_points_equivalent": points,
    }


def test_a_pool_with_a_break_leads_with_both_sides_and_the_jump_then_the_single_line():
    results = SimpleNamespace(
        trends=pd.DataFrame(
            [
                trend_row("main", 2014, 2026, -0.058, -0.128, 0.011, 0.1),
                trend_row("before_break", 2014, 2019, 0.165, 0.003, 0.326, 0.04),
                trend_row("after_break", 2021, 2026, 0.096, -0.091, 0.282, 0.3),
            ]
        ),
        effects=pd.DataFrame([effect_row(2019, 0.8, 0.3, 120.0), effect_row(2021, -1.2, 0.3, -180.0), effect_row(2026, 0.0, 0.0, 0.0)]),
    )
    (line,) = [l for l in report._headline_years(results).splitlines() if l.startswith("- ")]
    text = line.split("): ", 1)[1]
    markers = ("reclassified 10|0 as rapid on 2020-09-10", "Before (2014-2019)", "After (2021-2026)", "Across the break", "One line across the whole span")
    order = [text.index(marker) for marker in markers]
    assert order == sorted(order)
    assert "a given rating buys more accuracy each year (+0.165" in text  # the before side has a trend the single line hides
    assert "2.00 points less accurately in 2021 than in 2019" in text
    assert "-300 rating points" in text.split("Across the break")[1].split("One line")[0]


def test_a_pool_without_a_break_keeps_its_single_trend_first():
    results = SimpleNamespace(
        trends=pd.DataFrame([trend_row("main", 2018, 2026, 0.069, -0.063, 0.202, 0.3, platform="lichess")]),
        effects=pd.DataFrame(columns=["platform", "time_class"]),
    )
    (line,) = [l for l in report._headline_years(results).splitlines() if l.startswith("- ")]
    assert line.startswith("- **Lichess rapid** (2018-2026): no detectable change (+0.069")
    assert "reclassified" not in line


def test_the_sites_section_names_the_years_skipped_for_being_thin():
    skipped = pd.DataFrame({"time_class": ["blitz", "rapid"], "year": [2022, 2015], "chesscom": [364, 411], "lichess": [0, 12]})
    text = report._skipped_years_text(skipped)
    assert "fewer than 60 usable rows" in text
    assert "blitz: 2022 (chess.com 364, Lichess 0)" in text and "rapid: 2015 (chess.com 411, Lichess 12)" in text
    assert report._skipped_years_text(skipped.iloc[0:0]) == ""


def test_the_chain_table_builds_its_mask_without_a_pandas_warning():
    from chess_elo_drift.analysis import conversion

    source, target = conversion.PAIRS[0]
    chains = pd.DataFrame(
        {
            "from": [source.label, "x"], "to": [target.label, "y"], "via": ["v", "v"],
            "rating": [conversion.CHECK_RATINGS[1]] * 2, "direct": [1500.0, 1500.0],
            "chained": [1510.0, 1490.0], "discrepancy": [10.0, -10.0],
        },
        index=[5, 9],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = report._chain_text(chains)
    assert "**Consistency.**" in text


def test_the_method_names_site_years_drawn_mostly_from_one_month():
    from chess_elo_drift.analysis.report import _season_balance_text

    sample = pd.DataFrame(
        {
            "platform": ["lichess"] * 10 + ["chesscom"] * 9,
            "year": [2017] * 10 + [2017] * 9,
            "month": ["2017-03"] * 9 + ["2017-06"] + ["2017-03", "2017-06", "2017-09"] * 3,
        }
    )
    text = _season_balance_text(sample)
    assert "Lichess 2017 (90%)" in text
    assert "chess.com" not in text
