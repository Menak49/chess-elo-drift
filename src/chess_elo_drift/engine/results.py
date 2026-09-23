"""The tidy dataset produced by the engine stage.

One row per *player-observation*: a single side of a single game, carrying the
rating that side held at the time and how well they played. That is the unit the
study compares, so it is the unit stored.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from chess_elo_drift.engine.evaluator import GameEvaluation
from chess_elo_drift.records import GameRecord

COLUMNS: Sequence[str] = (
    "game_id",
    "era",
    "month",
    "time_class",
    "time_control",
    "colour",
    "username",
    "rating",
    "opponent_rating",
    "result",
    "ply_count",
    "moves_scored",
    "accuracy",
    "acpl",
    "inaccuracies",
    "mistakes",
    "blunders",
    "reported_accuracy",
    "engine_depth",
)


def to_rows(record: GameRecord, evaluation: GameEvaluation) -> list[dict[str, object]]:
    """Flatten a game and its evaluation into one row per side."""
    rows: list[dict[str, object]] = []
    for side in record.sides():
        scored = evaluation.side(side.colour)
        rows.append(
            {
                "game_id": record.game_id,
                "era": record.era,
                "month": record.month,
                "time_class": record.time_class,
                "time_control": record.time_control,
                "colour": side.colour,
                "username": side.username,
                "rating": side.rating,
                "opponent_rating": side.opponent_rating,
                "result": side.result,
                "ply_count": record.ply_count,
                "moves_scored": scored.moves_scored,
                "accuracy": _round(scored.accuracy, 3),
                "acpl": _round(scored.acpl, 2),
                "inaccuracies": scored.inaccuracies,
                "mistakes": scored.mistakes,
                "blunders": scored.blunders,
                "reported_accuracy": side.reported_accuracy,
                "engine_depth": evaluation.engine_depth,
            }
        )
    return rows


class EvaluationWriter:
    """Streaming CSV writer that lets an interrupted run be resumed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._is_new = not self.path.exists() or self.path.stat().st_size == 0

    def completed_game_ids(self) -> set[str]:
        """Game ids already evaluated, so a resumed run skips them."""
        if self._is_new:
            return set()
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            return {row["game_id"] for row in csv.DictReader(handle) if row.get("game_id")}

    def append(self, rows: Iterable[dict[str, object]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
            if self._is_new:
                writer.writeheader()
                self._is_new = False
            writer.writerows(rows)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)
