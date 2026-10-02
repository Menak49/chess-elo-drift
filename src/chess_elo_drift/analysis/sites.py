"""chess.com against Lichess: which rating on one site plays like a rating on the other?

The two sites' ratings are different scales, and no player-level link exists
in the engine sample. What both sites do share is the instrument: every game is
scored by the same engine at the same depth. So for one year and one cadence,
each site's accuracy-rating curve is fitted separately,

    accuracy ~ rating + rating² + log(clock)                (one fit per site)

and a chess.com rating R is matched to the Lichess rating whose predicted
accuracy is the same -- *accuracy equating*. The reverse direction is the same
curves read the other way. The offset reported is Lichess minus chess.com.

The clock term matters more here than anywhere: the sites draw their cadence
boundaries differently and their players favour different controls, so every
prediction is made at the same reference clock on both sides.

Uncertainty comes from a bootstrap that resamples *players* within each site
(rows of one player stay together), because the accuracy curve is shallow and
an equated rating is a ratio of noisy quantities whose error is far from
normal. Equal accuracy is not equal rating-pool membership: the claim is only
that the two groups play moves of the same engine-judged quality.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.analysis.dataset import RATING_CENTRE
from chess_elo_drift.analysis.statistics import weighted_least_squares
from chess_elo_drift.analysis.yearly import REFERENCE_RATINGS

#: Source ratings the equating is tabulated at.
EQUATING_GRID: tuple[int, ...] = tuple(range(800, 2201, 100))

#: Each site needs this many rows in a year and cadence before its curve is
#: trusted to equate anything.
MIN_SITE_OBSERVATIONS = 60

DEFAULT_BOOTSTRAP_REPLICATES = 300

#: Share of bootstrap replicates allowed to fail (equated rating outside the
#: sampled range) before the interval is declared unavailable rather than
#: silently computed from the survivors.
MAX_FAILED_SHARE = 0.2

DIRECTIONS = {"chesscom": "lichess", "lichess": "chesscom"}

OFFSET_COLUMNS = (
    "year", "time_class", "source", "target", "rating", "equivalent", "ci_low", "ci_high",
    "offset", "offset_ci_low", "offset_ci_high", "n_source", "n_target",
)


@dataclass(frozen=True)
class SiteCell:
    """One site's rows in one year and cadence, ready for repeated fitting."""

    design: np.ndarray
    outcome: np.ndarray
    player_codes: np.ndarray
    players: int
    rating_range: tuple[float, float]

    @classmethod
    def from_rows(cls, rows: pd.DataFrame) -> "SiteCell":
        rating = rows["rating_c"].to_numpy(dtype=float)
        columns = [np.ones(len(rows)), rating, rating**2]
        log_clock = rows["log_clock"].to_numpy(dtype=float)
        # Without spread in the clock there is nothing to control, and a
        # constant column would duplicate the intercept.
        columns.append(log_clock if np.std(log_clock) > 1e-6 else np.zeros(len(rows)))
        codes, unique = pd.factorize(rows["player_id"])
        return cls(
            design=np.column_stack(columns),
            outcome=rows["accuracy"].to_numpy(dtype=float),
            player_codes=codes,
            players=len(unique),
            rating_range=(float(rows["rating"].min()), float(rows["rating"].max())),
        )

    def coefficients(self, weights: np.ndarray | None = None) -> np.ndarray:
        if weights is None:
            weights = np.ones(len(self.outcome))
        return weighted_least_squares(self.design, self.outcome, weights)

    def resampled_weights(self, rng: np.random.Generator) -> np.ndarray:
        draws = np.bincount(rng.integers(0, self.players, self.players), minlength=self.players)
        return draws[self.player_codes].astype(float)


def predicted_accuracy(coefficients: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """The fitted curve at the reference clock (log clock = 0)."""
    r = (np.asarray(ratings, dtype=float) - RATING_CENTRE) / 100.0
    return coefficients[0] + coefficients[1] * r + coefficients[2] * r**2


def equate(
    source: np.ndarray,
    target: np.ndarray,
    ratings: np.ndarray,
    target_range: tuple[float, float],
) -> np.ndarray:
    """Target-site ratings whose predicted accuracy matches the source site's.

    The target curve is read only inside the ratings its site was sampled at,
    and only on its rising part: a match found by extrapolating the curve, or on
    a stretch where more rating predicts less accuracy, is no match at all and
    comes back as NaN.
    """
    wanted = predicted_accuracy(source, ratings)
    grid = np.arange(np.floor(target_range[0]), np.ceil(target_range[1]) + 1.0)
    curve = np.maximum.accumulate(predicted_accuracy(target, grid))
    if curve[-1] <= curve[0]:
        return np.full(len(wanted), np.nan)
    position = np.searchsorted(curve, wanted, side="left")
    inside = (wanted >= curve[0]) & (wanted <= curve[-1]) & (position < len(grid))
    matched = np.full(len(wanted), np.nan)
    matched[inside] = grid[np.clip(position[inside], 0, len(grid) - 1)]
    return matched


def site_offsets(
    sample: pd.DataFrame,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 20142026,
    grid: tuple[int, ...] = EQUATING_GRID,
) -> pd.DataFrame:
    """Equated ratings in both directions, for every year and cadence with both sites."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for time_class in config.TIME_CLASSES:
        for year in sorted(sample["year"].unique()) if not sample.empty else []:
            cells = _site_cells(sample, int(year), time_class)
            if cells is None:
                continue
            for source, target in DIRECTIONS.items():
                rows.extend(_equate_cell(cells[source], cells[target], int(year), time_class, source, target, grid, replicates, rng))
    return pd.DataFrame(rows, columns=list(OFFSET_COLUMNS))


def _usable_rows(sample: pd.DataFrame, platform: str, year: int, time_class: str) -> pd.DataFrame:
    rows = sample[(sample["platform"] == platform) & (sample["year"] == year) & (sample["time_class"] == time_class)]
    return rows.dropna(subset=["accuracy", "log_clock"])


def _site_cells(sample: pd.DataFrame, year: int, time_class: str) -> dict[str, SiteCell] | None:
    cells: dict[str, SiteCell] = {}
    for platform in DIRECTIONS:
        rows = _usable_rows(sample, platform, year, time_class)
        if len(rows) < MIN_SITE_OBSERVATIONS:
            return None
        cells[platform] = SiteCell.from_rows(rows)
    return cells


def skipped_cells(sample: pd.DataFrame) -> pd.DataFrame:
    """Years and cadences that have games but were too thin on a site to equate.

    A year missing from the offsets table is otherwise indistinguishable from a
    year nobody collected; this names the ones that were collected and lost to
    `MIN_SITE_OBSERVATIONS`, with the usable rows each site had. Years with no
    row on either site are not listed: nothing was thin there, it was absent.
    """
    columns = ["time_class", "year", *DIRECTIONS]
    rows = []
    for time_class in config.TIME_CLASSES:
        for year in sorted(sample["year"].unique()) if not sample.empty else []:
            counts = {platform: len(_usable_rows(sample, platform, int(year), time_class)) for platform in DIRECTIONS}
            if sum(counts.values()) and min(counts.values()) < MIN_SITE_OBSERVATIONS:
                rows.append({"time_class": time_class, "year": int(year), **counts})
    return pd.DataFrame(rows, columns=columns)


def _equate_cell(
    source: SiteCell,
    target: SiteCell,
    year: int,
    time_class: str,
    source_name: str,
    target_name: str,
    grid: tuple[int, ...],
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    ratings = np.asarray(grid, dtype=float)
    point = equate(source.coefficients(), target.coefficients(), ratings, target.rating_range)

    draws = np.full((replicates, len(ratings)), np.nan)
    for replicate in range(replicates):
        draws[replicate] = equate(
            source.coefficients(source.resampled_weights(rng)),
            target.coefficients(target.resampled_weights(rng)),
            ratings,
            target.rating_range,
        )
    low, high = _percentile_interval(draws)

    rows = []
    for index, rating in enumerate(grid):
        equivalent = point[index]
        valid = np.isfinite(equivalent) and np.isfinite(low[index])
        rows.append(
            {
                "year": year,
                "time_class": time_class,
                "source": source_name,
                "target": target_name,
                "rating": rating,
                "equivalent": equivalent,
                "ci_low": low[index] if valid else np.nan,
                "ci_high": high[index] if valid else np.nan,
                "offset": _lichess_minus_chesscom(source_name, rating, equivalent),
                "offset_ci_low": np.nan,
                "offset_ci_high": np.nan,
                "n_source": len(source.outcome),
                "n_target": len(target.outcome),
            }
        )
        if valid:
            offsets = sorted(
                (
                    _lichess_minus_chesscom(source_name, rating, low[index]),
                    _lichess_minus_chesscom(source_name, rating, high[index]),
                )
            )
            rows[-1]["offset_ci_low"], rows[-1]["offset_ci_high"] = offsets
    return rows


def _percentile_interval(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95% percentile interval per column, or NaN where too many replicates failed."""
    failed = np.mean(~np.isfinite(draws), axis=0)
    with np.errstate(all="ignore"):
        if np.all(np.isnan(draws)):
            nan = np.full(draws.shape[1], np.nan)
            return nan, nan.copy()
        # A column where every replicate failed is all-NaN: NaN is the right answer,
        # the warning is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.nanpercentile(draws, 2.5, axis=0)
            high = np.nanpercentile(draws, 97.5, axis=0)
    unusable = failed > MAX_FAILED_SHARE
    low[unusable] = np.nan
    high[unusable] = np.nan
    return low, high


def _lichess_minus_chesscom(source: str, rating: float, equivalent: float) -> float:
    if not np.isfinite(equivalent):
        return float("nan")
    return equivalent - rating if source == "chesscom" else rating - equivalent


def offsets_at(offsets: pd.DataFrame, ratings: tuple[int, ...] = REFERENCE_RATINGS) -> pd.DataFrame:
    """The chess.com-to-Lichess offsets at a few ratings: the figure's and headline's view."""
    selected = offsets[(offsets["source"] == "chesscom") & offsets["rating"].isin(ratings)]
    return selected.reset_index(drop=True)
