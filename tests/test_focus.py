"""The single-clock corpus (part one) and the redone first study (part three)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.analysis import parts
from chess_elo_drift.analysis.dataset import apply_inclusion_rules
from chess_elo_drift.analysis.report import generate_report
from chess_elo_drift.collection.focus import FocusedSource, import_existing_games, is_focus_game
from chess_elo_drift.collection.sampler import CoverageTarget, band_floor, study_bands
from chess_elo_drift.collection.sources import MonthHarvest
from chess_elo_drift.collection.store import GameStore
from chess_elo_drift.config import YearMonth
from chess_elo_drift.records import GameRecord

from tests.test_analysis import CONTROLS, synthetic_evaluations, write_conversion_inputs

LONG_PGN = "1. e4 e5"


def record(game_id: str, time_class: str, time_control: str, platform: str = "chesscom") -> GameRecord:
    return GameRecord(
        game_id=game_id, url="https://example.invalid/" + game_id, platform=platform, year=2024,
        month="2024-03", time_class=time_class, time_control=time_control, ply_count=40,
        white_username="a", white_rating=1500, white_result="win",
        black_username="b", black_rating=1490, black_result="loss", pgn=LONG_PGN,
    )


class FakeSource:
    platform = "chesscom"
    index_is_free = True

    def __init__(self, games):
        self.games = games

    def active_months(self, username, era):
        return list(era.months)

    def month_games(self, username, month):
        return MonthHarvest(self.games, None)


def test_only_the_two_focus_clocks_are_kept_on_both_sites():
    keep = [
        record("1", "blitz", "180"),
        record("2", "blitz", "180+0", platform="lichess"),
        record("3", "rapid", "900+10"),
        record("9", "rapid", "600+0", platform="lichess"),
    ]
    drop = [
        record("4", "blitz", "300"),
        record("5", "blitz", "180+2"),
        record("6", "rapid", "600"),
        record("7", "rapid", "1800"),
        record("8", "blitz", "900+10"),
        record("10", "rapid", "900+10", platform="lichess"),
        record("11", "rapid", "600+0"),
    ]
    harvest = FocusedSource(FakeSource(keep + drop)).month_games("a", YearMonth(2024, 3))
    assert [game.game_id for game in harvest.games] == ["1", "2", "3", "9"]
    assert harvest.snapshot is None
    assert all(is_focus_game(game) for game in harvest.games)


def test_focus_rating_cap_limits_the_bands_and_ignores_stronger_players():
    bands = study_bands(rating_max=config.FOCUS_RATING_MAX)
    assert bands[0] == config.RATING_MIN
    assert bands[-1] + config.CRAWL_BAND_WIDTH == config.FOCUS_RATING_MAX
    assert band_floor(config.FOCUS_RATING_MAX, rating_max=config.FOCUS_RATING_MAX) == bands[-1]
    assert band_floor(config.FOCUS_RATING_MAX + 1, rating_max=config.FOCUS_RATING_MAX) is None
    target = CoverageTarget(10, rating_max=config.FOCUS_RATING_MAX)
    assert target.bands == bands


def test_existing_games_are_reused_without_duplicates(tmp_path, monkeypatch):
    main = tmp_path / "main"
    monkeypatch.setattr(config, "DATA_RAW", main)
    source = GameStore(config.games_path("chesscom", 2024))
    for game in (record("1", "blitz", "180"), record("2", "blitz", "300"), record("3", "rapid", "900+10")):
        source.add(game)
    focus = GameStore(tmp_path / "focus" / "games_2024.jsonl")
    assert import_existing_games("chesscom", 2024, focus) == 2
    assert import_existing_games("chesscom", 2024, focus) == 0
    assert len(focus) == 2


def evaluations_on_focus_clocks(years=(2018, 2022, 2026)) -> pd.DataFrame:
    controls = {
        key: (config.FOCUS_TIME_CONTROLS[key[0]][key[1]] and tuple(config.FOCUS_TIME_CONTROLS[key[0]][key[1]]))
        for key in CONTROLS
    }
    frame = synthetic_evaluations(years=years, per_cell=120, controls=controls, trend_per_year=-0.2, seed=5)
    return frame


def test_focus_sample_keeps_only_the_focus_clocks_and_ratings_up_to_the_cap():
    mixed = synthetic_evaluations(years=(2022, 2026), per_cell=80, seed=3)
    sample, attrition = parts.focus_sample(mixed, max_per_player=3)
    assert not sample.empty
    allowed = sample.apply(lambda r: r["time_control"] in config.FOCUS_TIME_CONTROLS[r["platform"]][r["time_class"]], axis=1)
    assert allowed.all()
    assert sample["rating"].max() <= config.FOCUS_RATING_MAX
    assert attrition["rows_left"].iloc[-1] == len(sample)


def test_focus_trend_recovers_a_planted_decline_without_a_clock_term():
    results = parts.compute_focus(evaluations_on_focus_clocks(), max_per_player=3)
    assert not results.trends.empty
    for row in results.trends.itertuples():
        assert -0.35 < row.trend_per_year < -0.05


def first_study_frame() -> pd.DataFrame:
    """chess.com rapid whose later period is mostly 10|0, on top of a true change of zero."""
    controls = dict(CONTROLS)
    controls[("chesscom", "rapid", "before")] = ("900+10", "1800", "1500+10", "900")
    controls[("chesscom", "rapid", "after")] = ("600", "600", "600", "900+10")
    controls[("chesscom", "blitz", "before")] = ("180", "300")
    controls[("chesscom", "blitz", "after")] = ("180", "300")
    return synthetic_evaluations(
        years=(2018, 2019, 2024, 2025), platforms=("chesscom",), per_cell=500, controls=controls,
        trend_per_year=0.0, clock_effect=2.5, seed=8,
    )


def test_the_first_study_gap_is_a_clock_artefact_that_the_clock_control_removes():
    sample, _ = apply_inclusion_rules(first_study_frame())
    study = parts.compute_first_study(sample)
    rapid = study.gaps[study.gaps["time_class"] == "rapid"].set_index("specification")
    assert rapid.loc["original", "gap"] < -0.8
    assert abs(rapid.loc["clock", "gap"]) < abs(rapid.loc["original", "gap"]) / 2
    composition = study.composition[study.composition["time_class"] == "rapid"].set_index("era")
    assert composition.loc["2024-2025", "share_10_0"] > composition.loc["2018-2019", "share_10_0"]
    assert np.isfinite(study.bands["mean_accuracy"]).all()


def test_the_findings_have_three_parts_and_flag_the_redrawn_figure(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_conversion_inputs(raw, people=300, seed=2)
    evaluations_path = tmp_path / "evaluations.csv"
    focus_path = tmp_path / "focus.csv"
    pd.concat([first_study_frame(), synthetic_evaluations(years=(2016, 2020, 2026), per_cell=60, seed=4)]).to_csv(
        evaluations_path, index=False
    )
    evaluations_on_focus_clocks().to_csv(focus_path, index=False)

    artefacts = generate_report(
        evaluations_path=evaluations_path, focus_evaluations_path=focus_path, data_raw=raw,
        output_root=tmp_path / "out", site_replicates=20, conversion_replicates=40,
    )
    text = artefacts.findings.read_text(encoding="utf-8")
    for heading in ("## Part 1.", "## Part 2.", "## Part 3."):
        assert heading in text
    assert text.index("## Part 1.") < text.index("## Part 2.") < text.index("## Part 3.")
    assert "3+0" in text and "15+10" in text
    assert "worthless" in text.lower() and "not the original" in text.lower()
    assert {"focus_by_year", "first_study"} <= set(artefacts.figures)
    assert "focus_year_trends" in artefacts.tables


def test_part_one_says_so_when_the_focus_corpus_is_missing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_conversion_inputs(raw, people=300, seed=2)
    evaluations_path = tmp_path / "evaluations.csv"
    first_study_frame().to_csv(evaluations_path, index=False)
    artefacts = generate_report(
        evaluations_path=evaluations_path, data_raw=raw, output_root=tmp_path / "out",
        site_replicates=20, conversion_replicates=40,
    )
    assert "has not been collected" in artefacts.findings.read_text(encoding="utf-8")
