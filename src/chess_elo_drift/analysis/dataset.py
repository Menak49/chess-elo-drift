"""Turning the engine output into the sample the study actually estimates on.

Collection deliberately over-gathers: the crawler stores a game whenever either
side helps fill a quota, which means a handful of very active accounts show up
far more often than everyone else. Deciding who counts is the analysis's job,
not the crawler's, and it happens here, once, in the open.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from chess_elo_drift import config

logger = logging.getLogger(__name__)

#: Rows per player, per era, per time class. A player who appears dozens of
#: times would otherwise contribute their personal strength to what is supposed
#: to be a population average.
MAX_OBSERVATIONS_PER_PLAYER = 3

METRICS: tuple[str, ...] = ("accuracy", "acpl")


def load_evaluations(path: Path | None = None) -> pd.DataFrame:
    """Read the engine output as typed columns."""
    path = path or config.DATA_PROCESSED / "evaluations.csv"
    frame = pd.read_csv(path)
    frame["era"] = frame["era"].astype("category")
    frame["time_class"] = frame["time_class"].astype("category")
    return frame


def build_estimation_sample(
    frame: pd.DataFrame,
    *,
    min_moves: int = config.MIN_MOVES_SCORED_PER_SIDE,
    max_per_player: int = MAX_OBSERVATIONS_PER_PLAYER,
) -> pd.DataFrame:
    """Apply the study's inclusion rules and add the columns the analysis uses.

    The rules, in order:

    1. the side must have been scored on enough moves to mean anything;
    2. its rating must sit inside the window the study is about;
    3. no player may contribute more than `max_per_player` rows to any one
       era-and-time-class, so the sample describes a population rather than its
       most prolific members.
    """
    before = len(frame)
    sample = frame.dropna(subset=["accuracy"])
    sample = sample[sample["moves_scored"] >= min_moves]
    sample = sample[sample["rating"].between(config.RATING_MIN, config.RATING_MAX)]
    sample = _cap_per_player(sample, max_per_player)

    sample = sample.assign(
        band=_report_band(sample["rating"]),
        rating_centred=sample["rating"] - _WINDOW_MIDPOINT,
        is_modern=(sample["era"] == config.MODERN_ERA.name).astype(int),
    )
    sample["band_label"] = sample["band"].map(lambda low: f"{low}-{low + config.REPORT_BAND_WIDTH - 1}")

    logger.info(
        "estimation sample: %d rows from %d (%d players, %d games)",
        len(sample), before, sample["username"].nunique(), sample["game_id"].nunique(),
    )
    return sample.reset_index(drop=True)


_WINDOW_MIDPOINT = (config.RATING_MIN + config.RATING_MAX) / 2


def _cap_per_player(frame: pd.DataFrame, max_per_player: int) -> pd.DataFrame:
    """Keep at most `max_per_player` rows per player, era and time class.

    Rows are ordered by game id first so the choice is reproducible rather than
    dependent on the order games happened to be collected in.
    """
    ordered = frame.sort_values(["username", "era", "time_class", "game_id"], kind="stable")
    ranked = ordered.groupby(["username", "era", "time_class"], observed=True).cumcount()
    return ordered[ranked < max_per_player]


def _report_band(ratings: pd.Series) -> pd.Series:
    width = config.REPORT_BAND_WIDTH
    offset = ((ratings - config.RATING_MIN) // width * width).clip(
        upper=config.RATING_MAX - config.RATING_MIN - width
    )
    return config.RATING_MIN + offset


def coverage_of_reported_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    """How often chess.com's own accuracy number is present, by era and cadence.

    This is the evidence for not using that field as the study's metric, so it
    is reported rather than merely asserted.
    """
    grouped = frame.groupby(["era", "time_class"], observed=True)["reported_accuracy"]
    return pd.DataFrame(
        {
            "observations": grouped.size(),
            "with_reported_accuracy": grouped.count(),
            "coverage": (grouped.count() / grouped.size()).round(4),
        }
    ).reset_index()
