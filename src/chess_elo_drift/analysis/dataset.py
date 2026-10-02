"""Turning the engine output into the sample the study actually estimates on.

Collection deliberately over-gathers: the crawler stores a game whenever either
side helps fill a quota, which means a handful of very active accounts show up
far more often than everyone else. Deciding who counts is the analysis's job,
not the crawler's, and it happens here, once, in the open -- every rule records
how many rows it removed, and that attrition is printed in the findings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.records import estimated_duration_seconds

logger = logging.getLogger(__name__)

#: Rows per player, per site, per year, per time class. A player who appears
#: dozens of times would otherwise contribute their personal strength to what is
#: supposed to be a population average.
MAX_OBSERVATIONS_PER_PLAYER = 3

#: Rating at which the models are centred, in the middle of the study window,
#: so that intercepts and year effects read as "at 1500".
RATING_CENTRE = 1500

#: The clock every prediction is made at: one common game length per cadence,
#: so a year (or a site) whose games happen to be longer is not credited with
#: the accuracy the extra time buys. 5+0 and 10+0 are the most played controls
#: of their cadence on both sites.
REFERENCE_CLOCK_SECONDS = {"blitz": 300, "rapid": 600}

ATTRITION_COLUMNS = ("rule", "rows_removed", "rows_left")


def load_evaluations(path: Path | None = None) -> pd.DataFrame:
    """Read the engine output, or an empty frame if the file does not exist yet."""
    path = path or config.EVALUATIONS_PATH
    if not Path(path).exists():
        logger.warning("no evaluations at %s", path)
        return pd.DataFrame(columns=_EVALUATION_COLUMNS)
    return pd.read_csv(path, dtype={"time_control": str, "username": str, "month": str})


def build_estimation_sample(
    frame: pd.DataFrame,
    *,
    min_moves: int = config.MIN_MOVES_SCORED_PER_SIDE,
    max_per_player: int = MAX_OBSERVATIONS_PER_PLAYER,
) -> pd.DataFrame:
    """Apply the inclusion rules and add the columns the models use."""
    sample, _ = apply_inclusion_rules(frame, min_moves=min_moves, max_per_player=max_per_player)
    return sample


def apply_inclusion_rules(
    frame: pd.DataFrame,
    *,
    min_moves: int = config.MIN_MOVES_SCORED_PER_SIDE,
    max_per_player: int = MAX_OBSERVATIONS_PER_PLAYER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The estimation sample and a table of what each rule removed.

    The rules, in order:

    1. the engine must have produced an accuracy, from enough scored moves;
    2. the cadence must be one of the two studied, with a known clock;
    3. the rating must sit inside the study window;
    4. Lichess rapid only counts from the month the pool existed -- before
       that the API labels Classical games "rapid" by today's speed rules, and
       the rating on them is a Classical rating;
    5. provisional Lichess ratings are dropped: they are the site's own
       statement that the number does not yet describe the player;
    6. no player contributes more than `max_per_player` rows to any one site,
       year and cadence, so a cell describes a population rather than its most
       prolific members.
    """
    sample = _normalise(frame)
    attrition: list[tuple[str, int, int]] = [("collected", 0, len(sample))]

    def keep(rule: str, mask: pd.Series) -> None:
        nonlocal sample
        removed = int((~mask).sum())
        sample = sample[mask]
        attrition.append((rule, removed, len(sample)))

    keep(
        f"accuracy scored on at least {min_moves} moves",
        sample["accuracy"].notna() & (sample["moves_scored"] >= min_moves),
    )
    keep(
        "blitz or rapid, with a known clock",
        sample["time_class"].isin(config.TIME_CLASSES) & sample["estimated_seconds"].gt(0),
    )
    keep(
        f"rating within {config.RATING_MIN}-{config.RATING_MAX}",
        sample["rating"].between(config.RATING_MIN, config.RATING_MAX),
    )
    keep("rating pool existed that month (Lichess rapid from 2018-01)", _pool_existed(sample))
    keep("Lichess rating not provisional", ~sample["provisional"])
    keep(f"at most {max_per_player} rows per player, site, year and cadence", _within_cap(sample, max_per_player))

    # A fixed row order makes every downstream result, bootstrap draws included,
    # independent of the order games happened to be collected in.
    sample = _add_model_columns(sample).sort_values(
        ["platform", "time_class", "year", "game_id", "colour"], kind="stable"
    ).reset_index(drop=True)
    logger.info(
        "estimation sample: %d rows from %d (%d players, %d games)",
        len(sample), len(frame), sample["player_id"].nunique(), sample["game_id"].nunique(),
    )
    return sample, pd.DataFrame(attrition, columns=list(ATTRITION_COLUMNS))


_EVALUATION_COLUMNS = (
    "game_id", "platform", "year", "month", "time_class", "time_control", "estimated_seconds",
    "colour", "username", "rating", "provisional", "opponent_rating", "result", "ply_count",
    "moves_scored", "accuracy", "acpl", "inaccuracies", "mistakes", "blunders",
    "reported_accuracy", "engine_depth",
)


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and fill columns an older or partial file may lack."""
    frame = frame.copy()
    for column in _EVALUATION_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    for column in ("rating", "moves_scored", "accuracy", "acpl", "reported_accuracy", "estimated_seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["platform"] = frame["platform"].fillna("chesscom").astype(str)
    frame["username"] = frame["username"].fillna("").astype(str)
    frame["time_class"] = frame["time_class"].astype(str)
    frame["time_control"] = frame["time_control"].fillna("").astype(str)
    frame["month"] = frame["month"].fillna("").astype(str)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    missing_year = frame["year"].isna()
    frame.loc[missing_year, "year"] = pd.to_numeric(frame.loc[missing_year, "month"].str[:4], errors="coerce")
    # Mapping the distinct controls and filling keeps the column's dtype; assigning
    # a list into the masked rows makes pandas warn when that mask is empty.
    clock_by_control = {
        control: estimated_duration_seconds(control) for control in frame["time_control"].unique()
    }
    derived_seconds = pd.to_numeric(frame["time_control"].map(clock_by_control), errors="coerce")
    frame["estimated_seconds"] = frame["estimated_seconds"].fillna(derived_seconds)
    frame["provisional"] = frame["provisional"].map(_as_flag).astype(bool)
    return frame


def _as_flag(value: object) -> bool:
    """chess.com leaves the provisional column empty; only an explicit yes counts."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _pool_existed(frame: pd.DataFrame) -> pd.Series:
    """`config.pool_exists`, applied to every row at once.

    "YYYY-MM" strings order like the months they name, so the comparison with a
    pool's first month can be done on text. A row without a month falls back to
    the end of its year.
    """
    months = frame["month"].where(
        frame["month"].str.match(r"^\d{4}-\d{2}$"), frame["year"].astype("Int64").astype(str) + "-12"
    )
    existed = pd.Series(True, index=frame.index)
    for (platform, time_class), start in config.POOL_START.items():
        same_pool = (frame["platform"] == platform) & (frame["time_class"] == time_class)
        existed &= ~(same_pool & (months < str(start)))
    return existed


def _within_cap(frame: pd.DataFrame, max_per_player: int) -> pd.Series:
    """Mark the first `max_per_player` rows of each player's site-year-cadence cell.

    Rows are ranked by game id so the choice is reproducible rather than
    dependent on the order games happened to be collected in.
    """
    ordered = frame.sort_values(["platform", "username", "year", "time_class", "game_id"], kind="stable")
    ranked = ordered.groupby(["platform", "username", "year", "time_class"], observed=True).cumcount()
    return (ranked < max_per_player).reindex(frame.index).astype(bool)


def _add_model_columns(sample: pd.DataFrame) -> pd.DataFrame:
    reference_clock = sample["time_class"].map(REFERENCE_CLOCK_SECONDS).astype(float)
    sample = sample.assign(
        year=sample["year"].astype(int),
        # A username is only unique within its site, so the cluster is the pair.
        player_id=sample["platform"] + ":" + sample["username"].str.lower(),
        rating_c=(sample["rating"] - RATING_CENTRE) / 100.0,
        log_clock=np.log(sample["estimated_seconds"].astype(float) / reference_clock),
        band=report_band(sample["rating"]),
    )
    sample["band_label"] = sample["band"].map(lambda low: f"{low}-{low + config.REPORT_BAND_WIDTH - 1}")
    return sample


def report_band(ratings: pd.Series) -> pd.Series:
    """The reporting band a rating falls in; the top of the window joins the last band."""
    width = config.REPORT_BAND_WIDTH
    offset = ((ratings - config.RATING_MIN) // width * width).clip(
        upper=config.RATING_MAX - config.RATING_MIN - width
    )
    return (config.RATING_MIN + offset).astype(int)


def sample_sizes(sample: pd.DataFrame) -> pd.DataFrame:
    """Observations and players per site, cadence and year."""
    columns = ["platform", "time_class", "year", "observations", "players", "median_rating"]
    if sample.empty:
        return pd.DataFrame(columns=columns)
    return (
        sample.groupby(["platform", "time_class", "year"], observed=True)
        .agg(
            observations=("accuracy", "size"),
            players=("player_id", "nunique"),
            median_rating=("rating", "median"),
        )
        .reset_index()[columns]
    )


def coverage_of_reported_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    """How often chess.com's own accuracy number is present, by year and cadence.

    This is the evidence for not using that field as the study's metric, so it
    is reported rather than merely asserted. Lichess publishes no comparable
    per-game figure through the archive used here, so only chess.com appears.
    """
    columns = ["time_class", "year", "observations", "with_reported_accuracy", "coverage"]
    frame = _normalise(frame)
    chesscom = frame[(frame["platform"] == "chesscom") & frame["time_class"].isin(config.TIME_CLASSES)]
    chesscom = chesscom.dropna(subset=["year"])
    if chesscom.empty:
        return pd.DataFrame(columns=columns)
    grouped = chesscom.groupby(["time_class", chesscom["year"].astype(int)], observed=True)["reported_accuracy"]
    table = pd.DataFrame(
        {
            "observations": grouped.size(),
            "with_reported_accuracy": grouped.count(),
            "coverage": (grouped.count() / grouped.size()).round(4),
        }
    ).reset_index()
    return table[columns]
