"""The report stage must produce every artefact it promises, on thin samples too.

These tests run the real pipeline end to end from a synthetic engine output, so
they cover the stage that actually writes the study's conclusion.
"""

import numpy as np
import pandas as pd
import pytest

from chess_elo_drift import config
from chess_elo_drift.analysis.dataset import build_estimation_sample
from chess_elo_drift.analysis.report import generate_report

from tests.test_analysis import LEGACY, MODERN, frame


def evaluations(n: int = 240, era_effect: float = -2.0) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for index in range(n):
        era = LEGACY if index % 2 else MODERN
        rating = int(rng.integers(config.RATING_MIN, config.RATING_MAX + 1))
        accuracy = 75.0 + 1.5 * (rating - 1400) / 100 + (era_effect if era == MODERN else 0.0)
        rows.append(
            {
                "game_id": f"g{index}",
                "username": f"player{index}",
                "era": era,
                "time_class": "blitz" if index % 4 < 2 else "rapid",
                "rating": rating,
                "accuracy": round(accuracy + rng.normal(scale=0.3), 3),
                "acpl": round(60 - accuracy / 3 + rng.normal(scale=1.0), 2),
                "reported_accuracy": None if era == LEGACY else round(accuracy, 1),
            }
        )
    return frame(rows)


def test_a_full_run_writes_every_artefact(tmp_path):
    full = evaluations()
    artifacts = generate_report(build_estimation_sample(full), full, output_root=tmp_path)

    assert artifacts.findings.exists() and artifacts.findings.stat().st_size > 0
    assert set(artifacts.tables) == {
        "accuracy_by_band",
        "acpl_by_band",
        "band_comparison",
        "era_gaps",
        "reported_accuracy_coverage",
    }
    for path in [*artifacts.tables.values(), *artifacts.figures.values()]:
        assert path.exists() and path.stat().st_size > 0


def test_the_findings_note_states_the_answer_and_names_its_method(tmp_path):
    full = evaluations(era_effect=-2.0)
    artifacts = generate_report(build_estimation_sample(full), full, output_root=tmp_path)
    note = artifacts.findings.read_text(encoding="utf-8")

    assert "blitz" in note and "rapid" in note
    assert "clustered" in note.lower()
    assert str(config.ENGINE_DEPTH) in note


def test_a_sample_too_thin_to_analyse_still_produces_a_report(tmp_path):
    """The state this project was in before the engine had scored much.

    Every table is empty and no model can be fitted, but the run must finish and
    say so, rather than fail on the way to writing the note.
    """
    thin = frame([{"game_id": "g1", "username": "solo", "era": LEGACY}])
    artifacts = generate_report(build_estimation_sample(thin), thin, output_root=tmp_path)

    assert artifacts.findings.exists()
    for path in [*artifacts.tables.values(), *artifacts.figures.values()]:
        assert path.exists() and path.stat().st_size > 0


def test_a_report_on_no_data_at_all_does_not_crash(tmp_path):
    empty = frame([{"game_id": "x"}]).iloc[:0]  # the right columns, no rows
    sample = build_estimation_sample(empty)
    artifacts = generate_report(sample, empty, output_root=tmp_path)
    assert artifacts.findings.exists()
