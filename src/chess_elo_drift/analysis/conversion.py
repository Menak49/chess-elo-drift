"""The 2026 conversions between the four rating pools.

For every pair of pools, a conversion fitted to people rated in both. These
are the numbers readers will quote, so the method matters.

**What "equivalent" means here.** One group of people holds both ratings. A
rating on pool A is equivalent to the rating on pool B that sits at the *same
percentile among those people* -- equipercentile linking, the standard
single-group design for putting two scales on one footing. It is symmetric (the
B-to-A table is exactly the inverse of the A-to-B one, so converting there and
back returns the starting rating), and it needs no error model and no assumed
shape, so it follows a relation that bends.

**Why not regression.** Regressing B on A answers a different question -- the
*average* B rating of people at a given A rating -- and is pulled toward the
mean by everything that makes a person's two ratings disagree; regressing A on
B is pulled the other way, and the two do not invert. That answer is still
reported, with its spread, as the typical-player prediction.

**Why not Deming as the headline.** A Deming fit with the error ratio taken from
the ratings' RDs assumes all disagreement between a person's two ratings is
rating noise. In the survey the residual spread is two to four times the RDs:
most of it is genuine preference for one cadence or site. The RD ratio then
misdescribes the errors, so Deming is kept only as a sensitivity column.

**The formula** is the straight-line member of the same family, *linear
equating*: the line that maps A's mean and standard deviation onto B's. Where
the relation is straight it coincides with the equipercentile table; where it
bends (checked with a squared term, `CURVATURE_TOLERANCE`), the text says so and
points to the table. Its valid range is the central 95% of the people it was
fitted on.

**The table** is piecewise linear through matched quantiles of the two ratings
at evenly spaced percentiles from 2.5% to 97.5%. The quantiles are presmoothed
(each is the average over a small window of percentiles) and their number
grows with the sample so each step rests on about `PEOPLE_PER_KNOT` people,
which keeps the curve from being jagged on a thin pair. Beyond the 2.5% and
97.5% points it continues along the linear-equating slope, and those rows are
marked as extrapolated. Intervals come from a bootstrap over people.

**Outliers.** A handful of pairs are plainly not one person's two current
ratings (a wrong or stale link, an abandoned pool): ratings 800 points apart
where everyone else is within 200. Before fitting, each pair's residual from a
median/MAD line (the robust version of linear equating) is compared with the
robust spread of all residuals, and pairs beyond `OUTLIER_ROBUST_SDS` robust
standard deviations are dropped and counted. Scaling vertical residuals by
their own spread is the same test as scaling perpendicular ones.

**Consistency** is checked by chaining: chess.com rapid -> Lichess rapid ->
Lichess blitz should land near the direct chess.com rapid -> Lichess blitz.

Sources: within a site, the current-rating survey (established ratings only)
is preferred to same-month snapshots, which carry no RD; the other serves as a
cross-check. Across sites, only self-declared linked accounts put one person on
both sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable

import numpy as np
import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.analysis.statistics import ols_clustered
from chess_elo_drift.analysis.surveys import POOLS, ConversionInputs, Pool, SourceLog

#: Ratings the conversion tables are printed at.
TABLE_RATINGS: tuple[int, ...] = tuple(range(800, 2201, 200))
#: Ratings the consistency checks are read at.
CHECK_RATINGS: tuple[int, ...] = (1000, 1500, 2000)

#: Pairs outside this window are left out of the fits. The rating floors
#: (100 on chess.com, 400 on Lichess) pile beginners up at the bottom and bend
#: the relation there, and a handful of titled players above 2800 would lever
#: the fit; the window still reaches 300 points beyond the published table.
FIT_WINDOW = (500, 2800)

MIN_PAIRS = 30
DEFAULT_BOOTSTRAP_REPLICATES = 1000

#: Pairs further than this many robust SDs from the robust line are dropped.
#: For well-behaved residuals four SDs flags about one pair in 15,000, so the
#: screen only catches pairs that cannot be one person's two current ratings.
OUTLIER_ROBUST_SDS = 4.0

#: The central band the equipercentile table is fitted on, and roughly how
#: many people each of its steps should rest on.
CENTRAL_BAND = (0.025, 0.975)
PEOPLE_PER_KNOT = 25
MIN_KNOTS, MAX_KNOTS = 5, 41

#: Largest gap, in rating points, between a straight and a curved fit over the
#: table's range that still counts as "a straight line is adequate": about the
#: RD of an established rating, i.e. within the rating's own uncertainty.
CURVATURE_TOLERANCE = 40.0

#: Two sources of the same conversion "agree" within this many points.
AGREEMENT_TOLERANCE = 50.0

CHESSCOM_RAPID, CHESSCOM_BLITZ, LICHESS_RAPID, LICHESS_BLITZ = POOLS

#: The six pairs, each fitted once; the reverse direction is the inverse.
PAIRS: tuple[tuple[Pool, Pool], ...] = (
    (CHESSCOM_RAPID, CHESSCOM_BLITZ),
    (LICHESS_RAPID, LICHESS_BLITZ),
    (CHESSCOM_RAPID, LICHESS_RAPID),
    (CHESSCOM_BLITZ, LICHESS_BLITZ),
    (CHESSCOM_RAPID, LICHESS_BLITZ),
    (CHESSCOM_BLITZ, LICHESS_RAPID),
)

Conversion = Callable[[np.ndarray], np.ndarray]


# -- straight lines ---------------------------------------------------------


@dataclass(frozen=True)
class Line:
    """y = intercept + slope * x."""

    intercept: float
    slope: float

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.intercept + self.slope * np.asarray(x, dtype=float)

    def inverse(self) -> "Line":
        return Line(-self.intercept / self.slope, 1.0 / self.slope)


def linear_equating(x: np.ndarray, y: np.ndarray) -> Line:
    """The line that maps x's mean and SD onto y's; its inverse does the reverse."""
    x_sd, y_sd = float(np.std(x)), float(np.std(y))
    if x_sd <= 0 or y_sd <= 0 or np.corrcoef(x, y)[0, 1] <= 0:
        return Line(float("nan"), float("nan"))
    slope = y_sd / x_sd
    return Line(float(np.mean(y)) - slope * float(np.mean(x)), slope)


def deming(x: np.ndarray, y: np.ndarray, delta: float = 1.0) -> Line:
    """Deming regression; `delta` is var(error in y) / var(error in x).

    delta -> infinity recovers OLS of y on x, delta -> 0 the inverse of OLS of
    x on y; fitting x on y with 1 / delta gives exactly the inverse line.
    """
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    s_xx = float(np.mean((x - x_mean) ** 2))
    s_yy = float(np.mean((y - y_mean) ** 2))
    s_xy = float(np.mean((x - x_mean) * (y - y_mean)))
    if abs(s_xy) < 1e-12:
        return Line(float("nan"), float("nan"))
    spread = s_yy - delta * s_xx
    slope = (spread + np.sqrt(spread**2 + 4 * delta * s_xy**2)) / (2 * s_xy)
    return Line(y_mean - slope * x_mean, float(slope))


def ols(x: np.ndarray, y: np.ndarray) -> Line:
    slope, intercept = np.polyfit(x, y, 1)
    return Line(float(intercept), float(slope))


# -- the equipercentile link ------------------------------------------------


@dataclass(frozen=True)
class Link:
    """Piecewise linear through matched quantiles, straight beyond them.

    Swapping the two knot sequences gives the exact inverse, which is what
    makes the table symmetric.
    """

    knots_x: np.ndarray
    knots_y: np.ndarray
    tail_slope: float

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = np.interp(x, self.knots_x, self.knots_y)
        below = self.knots_y[0] + self.tail_slope * (x - self.knots_x[0])
        above = self.knots_y[-1] + self.tail_slope * (x - self.knots_x[-1])
        return np.where(x < self.knots_x[0], below, np.where(x > self.knots_x[-1], above, inside))

    def inverse(self) -> "Link":
        return Link(self.knots_y, self.knots_x, 1.0 / self.tail_slope)


def knot_count(n: int) -> int:
    return int(np.clip(n // PEOPLE_PER_KNOT, MIN_KNOTS, MAX_KNOTS))


def equipercentile(x: np.ndarray, y: np.ndarray, knots: int | None = None, tail_slope: float | None = None) -> Link:
    """Match x and y at the same percentiles among the people who hold both."""
    knots = knots or knot_count(len(x))
    probabilities = np.linspace(*CENTRAL_BAND, knots)
    return Link(
        _strictly_increasing(_smoothed_quantiles(x, probabilities)),
        _strictly_increasing(_smoothed_quantiles(y, probabilities)),
        tail_slope if tail_slope is not None else linear_equating(x, y).slope,
    )


def _smoothed_quantiles(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Each quantile averaged over a window of half a knot spacing either side.

    A uniform-kernel smoothing of the quantile function: it removes the
    person-to-person jitter of raw order statistics without flattening a real bend.
    """
    spacing = (probabilities[-1] - probabilities[0]) / max(len(probabilities) - 1, 1)
    offsets = np.linspace(-spacing / 2, spacing / 2, 9)
    window = np.clip(probabilities[:, None] + offsets[None, :], 0.005, 0.995)
    return np.quantile(values, window.ravel()).reshape(window.shape).mean(axis=1)


def _strictly_increasing(values: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(values) + np.arange(len(values)) * 1e-6


# -- outliers ---------------------------------------------------------------


def _robust_sd(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def screen_outliers(x: np.ndarray, y: np.ndarray, threshold: float = OUTLIER_ROBUST_SDS) -> np.ndarray:
    """True for the pairs to keep: within `threshold` robust SDs of a median/MAD line."""
    x_spread, y_spread = _robust_sd(x), _robust_sd(y)
    if x_spread <= 0 or y_spread <= 0:
        return np.ones(len(x), dtype=bool)
    slope = y_spread / x_spread
    residuals = y - (np.median(y) - slope * np.median(x)) - slope * x
    scale = _robust_sd(residuals)
    if scale <= 0:
        return np.ones(len(x), dtype=bool)
    return np.abs(residuals - np.median(residuals)) <= threshold * scale


# -- one pair ---------------------------------------------------------------


@dataclass(frozen=True)
class Curvature:
    p_value: float
    max_deviation: float
    adequate: bool


@dataclass
class ConversionFit:
    """One pair's conversion, from one source."""

    x_pool: Pool
    y_pool: Pool
    source: str
    x: np.ndarray
    y: np.ndarray
    dropped_x: np.ndarray
    dropped_y: np.ndarray
    line: Line
    boot_lines: list[Line]
    link: Link
    boot_links: list[Link]
    deming: Line
    delta: float
    delta_from_rd: bool
    ols_y_on_x: Line
    ols_x_on_y: Line
    residual_sd_y: float
    residual_sd_x: float
    correlation: float
    curvature: Curvature
    x_range: tuple[float, float]
    y_range: tuple[float, float]

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def screened_out(self) -> int:
        return len(self.dropped_x)

    def published(self, forward: bool = True) -> Link:
        """The table's conversion in that direction."""
        return self.link if forward else self.link.inverse()

    def formula(self, forward: bool = True) -> Line:
        """The linear rule of thumb in that direction."""
        return self.line if forward else self.line.inverse()

    def bootstrap(self, forward: bool = True) -> list[Link]:
        return self.boot_links if forward else [link.inverse() for link in self.boot_links]

    def bootstrap_formulas(self, forward: bool = True) -> list[Line]:
        return self.boot_lines if forward else [line.inverse() for line in self.boot_lines]

    def typical(self, forward: bool = True) -> tuple[Line, float]:
        """The OLS prediction in that direction, and its residual SD."""
        return (self.ols_y_on_x, self.residual_sd_y) if forward else (self.ols_x_on_y, self.residual_sd_x)

    def source_range(self, forward: bool = True) -> tuple[float, float]:
        return self.x_range if forward else self.y_range

    def formula_gap(self, forward: bool = True) -> float:
        """Largest gap between formula and table over the table ratings the pair covers."""
        low, high = self.source_range(forward)
        grid = np.array([r for r in TABLE_RATINGS if low <= r <= high], dtype=float)
        if grid.size == 0:
            return float("nan")
        return float(np.max(np.abs(self.formula(forward)(grid) - self.published(forward)(grid))))


def fit_pair(
    people: pd.DataFrame,
    x_pool: Pool,
    y_pool: Pool,
    source: str,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    rng: np.random.Generator | None = None,
) -> ConversionFit | None:
    """Fit one pair from the people rated in both pools, or None if too few."""
    pairs = pair_frame(people, x_pool, y_pool)
    if len(pairs) < MIN_PAIRS:
        return None
    all_x, all_y = pairs["x"].to_numpy(dtype=float), pairs["y"].to_numpy(dtype=float)
    keep = screen_outliers(all_x, all_y)
    x, y = all_x[keep], all_y[keep]
    line = linear_equating(x, y)
    if len(x) < MIN_PAIRS or not (np.isfinite(line.slope) and line.slope > 0):
        # Ratings that do not rise together describe no conversion at all.
        return None

    rng = rng or np.random.default_rng(20142026)
    knots = knot_count(len(x))
    boot_lines, boot_links = [], []
    for _ in range(replicates):
        index = rng.integers(0, len(x), len(x))
        replicate = linear_equating(x[index], y[index])
        if np.isfinite(replicate.slope) and replicate.slope > 0:
            boot_lines.append(replicate)
            boot_links.append(equipercentile(x[index], y[index], knots, replicate.slope))

    rd = pairs.loc[keep, ["rd_x", "rd_y"]].to_numpy(dtype=float)
    delta, delta_from_rd = error_ratio(rd[:, 0], rd[:, 1])
    y_on_x, x_on_y = ols(x, y), ols(y, x)
    return ConversionFit(
        x_pool=x_pool,
        y_pool=y_pool,
        source=source,
        x=x,
        y=y,
        dropped_x=all_x[~keep],
        dropped_y=all_y[~keep],
        line=line,
        boot_lines=boot_lines,
        link=equipercentile(x, y, knots, line.slope),
        boot_links=boot_links,
        deming=deming(x, y, delta),
        delta=delta,
        delta_from_rd=delta_from_rd,
        ols_y_on_x=y_on_x,
        ols_x_on_y=x_on_y,
        residual_sd_y=float(np.std(y - y_on_x(x), ddof=2)),
        residual_sd_x=float(np.std(x - x_on_y(y), ddof=2)),
        correlation=float(np.corrcoef(x, y)[0, 1]),
        curvature=curvature(x, y),
        x_range=(float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))),
        y_range=(float(np.percentile(y, 2.5)), float(np.percentile(y, 97.5))),
    )


def pair_frame(people: pd.DataFrame, x_pool: Pool, y_pool: Pool) -> pd.DataFrame:
    """The people with an established rating in both pools, inside the fit window."""
    columns = {x_pool.key: "x", y_pool.key: "y", f"{x_pool.key}_rd": "rd_x", f"{y_pool.key}_rd": "rd_y"}
    if people.empty or not {x_pool.key, y_pool.key}.issubset(people.columns):
        return pd.DataFrame(columns=list(columns.values()))
    # A source without RDs (snapshots) still pairs ratings; its RDs are unknown.
    people = people.reindex(columns=list(columns))
    pairs = people.rename(columns=columns).apply(pd.to_numeric, errors="coerce")
    pairs = pairs.dropna(subset=["x", "y"])
    low, high = FIT_WINDOW
    return pairs[pairs["x"].between(low, high) & pairs["y"].between(low, high)].reset_index(drop=True)


def error_ratio(rd_x: np.ndarray, rd_y: np.ndarray) -> tuple[float, bool]:
    """var(error in y) / var(error in x) from the typical RDs, or 1 without them.

    The second value says whether RDs were available to estimate it.
    """
    known = np.isfinite(rd_x) & np.isfinite(rd_y) & (rd_x > 0) & (rd_y > 0)
    if known.sum() < 10:
        return 1.0, False
    return float(np.mean(rd_y[known] ** 2) / np.mean(rd_x[known] ** 2)), True


def curvature(x: np.ndarray, y: np.ndarray) -> Curvature:
    """Does a squared term change the fit enough to matter over the table's range?"""
    centred = (x - 1500.0) / 100.0
    design = np.column_stack([np.ones(len(x)), centred, centred**2])
    bent = ols_clustered(design, y, np.arange(len(x)))
    p_value = bent.contrast(np.array([0.0, 0.0, 1.0])).p_value

    straight = ols(x, y)
    low, high = np.percentile(x, [2.5, 97.5])
    grid = np.array([r for r in TABLE_RATINGS if low <= r <= high], dtype=float)
    if grid.size == 0:
        return Curvature(p_value, float("nan"), True)
    grid_c = (grid - 1500.0) / 100.0
    bent_values = bent.coefficients[0] + bent.coefficients[1] * grid_c + bent.coefficients[2] * grid_c**2
    max_deviation = float(np.max(np.abs(bent_values - straight(grid))))
    adequate = bool(max_deviation <= CURVATURE_TOLERANCE or not (p_value < 0.01))
    return Curvature(p_value, max_deviation, adequate)


# -- all pairs --------------------------------------------------------------


@dataclass
class ConversionResults:
    """The published fit per pair, plus the cross-check fits and the source logs."""

    fits: dict[tuple[str, str], ConversionFit] = field(default_factory=dict)
    cross_checks: list[ConversionFit] = field(default_factory=list)
    logs: list[SourceLog] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def resolve(self, source: Pool, target: Pool) -> tuple[ConversionFit, bool] | None:
        """The fit that converts `source` to `target`, and whether it runs forward."""
        if (source.key, target.key) in self.fits:
            return self.fits[(source.key, target.key)], True
        if (target.key, source.key) in self.fits:
            return self.fits[(target.key, source.key)], False
        return None


def conversion_study(
    inputs: ConversionInputs,
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = 20142026,
) -> ConversionResults:
    rng = np.random.default_rng(seed)
    results = ConversionResults(logs=inputs.logs)
    within = {
        "chesscom": (inputs.chesscom_survey, inputs.chesscom_snapshots),
        "lichess": (inputs.lichess_survey, inputs.lichess_snapshots),
    }
    for x_pool, y_pool in PAIRS:
        candidates: list[ConversionFit] = []
        if x_pool.platform == y_pool.platform:
            survey, snapshots = within[x_pool.platform]
            candidates = _ranked_within_site(
                fit_pair(survey, x_pool, y_pool, "current-rating survey", replicates=replicates, rng=rng),
                fit_pair(snapshots, x_pool, y_pool, f"{config.CONVERSION_YEAR} same-month snapshots", replicates=replicates, rng=rng),
            )
            linked = fit_pair(inputs.linked, x_pool, y_pool, "linked accounts", replicates=replicates, rng=rng)
            candidates += [linked] if linked is not None else []
        else:
            linked = fit_pair(inputs.linked, x_pool, y_pool, "linked accounts", replicates=replicates, rng=rng)
            candidates = [linked] if linked is not None else []

        if not candidates:
            results.notes.append(f"{x_pool.label} / {y_pool.label}: fewer than {MIN_PAIRS} people rated in both.")
            continue
        results.fits[(x_pool.key, y_pool.key)] = candidates[0]
        results.cross_checks.extend(candidates[1:])
    return results


def _ranked_within_site(survey: ConversionFit | None, snapshots: ConversionFit | None) -> list[ConversionFit]:
    """Survey first unless it is much smaller: its ratings were checked for RD and recency."""
    if survey is not None and snapshots is not None and survey.n * 2 < snapshots.n:
        return [snapshots, survey]
    return [fit for fit in (survey, snapshots) if fit is not None]


# -- tables -----------------------------------------------------------------


def convert(fit: ConversionFit, forward: bool, ratings: np.ndarray) -> dict[str, np.ndarray]:
    """Table conversion with its bootstrap interval, the formula, and the typical-player prediction."""
    ratings = np.asarray(ratings, dtype=float)
    boot = np.array([link(ratings) for link in fit.bootstrap(forward)]) if fit.boot_links else None
    nan = np.full(len(ratings), np.nan)
    typical, residual_sd = fit.typical(forward)
    low, high = fit.source_range(forward)
    typical_values = np.asarray(typical(ratings), dtype=float)
    return {
        "equivalent": np.asarray(fit.published(forward)(ratings), dtype=float),
        "ci_low": np.percentile(boot, 2.5, axis=0) if boot is not None else nan,
        "ci_high": np.percentile(boot, 97.5, axis=0) if boot is not None else nan,
        "formula": np.asarray(fit.formula(forward)(ratings), dtype=float),
        "typical": typical_values,
        "typical_low": typical_values - 1.96 * residual_sd,
        "typical_high": typical_values + 1.96 * residual_sd,
        "extrapolated": (ratings < low) | (ratings > high),
    }


def rounded_formula(line: Line, source: Pool, target: Pool) -> str:
    """`target ≈ a + b × source`, rounded so no table value moves by more than ~4 points.

    The slope is rounded to two decimals and the intercept re-anchored so the
    rounded line still passes through the exact one at 1500.
    """
    slope = round(line.slope, 2)
    intercept = round(float(line(1500.0)) - slope * 1500.0)
    return f"{target.label} ≈ {intercept:.0f} + {slope:.2f} × {source.label}"


def formulas_table(results: ConversionResults) -> pd.DataFrame:
    """Both directions of every published pair: the formula and how far to trust it."""
    rows = []
    for fit in results.fits.values():
        for forward in (True, False):
            source, target = (fit.x_pool, fit.y_pool) if forward else (fit.y_pool, fit.x_pool)
            line = fit.formula(forward)
            slopes = np.array([b.slope for b in fit.bootstrap_formulas(forward)])
            typical, residual_sd = fit.typical(forward)
            deming_line = fit.deming if forward else fit.deming.inverse()
            low, high = fit.source_range(forward)
            rows.append(
                {
                    "from": source.label,
                    "to": target.label,
                    "formula": rounded_formula(line, source, target),
                    "intercept": line.intercept,
                    "slope": line.slope,
                    "slope_ci_low": np.percentile(slopes, 2.5) if slopes.size else np.nan,
                    "slope_ci_high": np.percentile(slopes, 97.5) if slopes.size else np.nan,
                    "valid_from": low,
                    "valid_to": high,
                    "formula_vs_table_max_gap": fit.formula_gap(forward),
                    "straight_line_adequate": fit.curvature.adequate,
                    "curvature_max_deviation": fit.curvature.max_deviation,
                    "curvature_p_value": fit.curvature.p_value,
                    "source": fit.source,
                    "people": fit.n,
                    "outliers_dropped": fit.screened_out,
                    "correlation": fit.correlation,
                    "deming_slope": deming_line.slope,
                    "deming_error_ratio": fit.delta if forward else 1.0 / fit.delta,
                    "deming_ratio_from_rd": fit.delta_from_rd,
                    "ols_intercept": typical.intercept,
                    "ols_slope": typical.slope,
                    "ols_residual_sd": residual_sd,
                }
            )
    return pd.DataFrame(rows)


CONVERSION_TABLE_COLUMNS = (
    "from", "rating", "to", "equivalent", "ci_low", "ci_high", "formula", "typical", "typical_low",
    "typical_high", "extrapolated", "people", "source",
)


def conversion_table(results: ConversionResults, ratings: tuple[int, ...] = TABLE_RATINGS) -> pd.DataFrame:
    """Every pool's ratings expressed in every other pool."""
    rows = []
    for source, target in permutations(POOLS, 2):
        resolved = results.resolve(source, target)
        if resolved is None:
            continue
        fit, forward = resolved
        converted = convert(fit, forward, np.asarray(ratings))
        for index, rating in enumerate(ratings):
            rows.append(
                {
                    "from": source.label,
                    "rating": rating,
                    "to": target.label,
                    **{key: values[index] for key, values in converted.items()},
                    "people": fit.n,
                    "source": fit.source,
                }
            )
    return pd.DataFrame(rows, columns=list(CONVERSION_TABLE_COLUMNS))


def chain_table(results: ConversionResults, ratings: tuple[int, ...] = CHECK_RATINGS) -> pd.DataFrame:
    """Direct conversion against every two-step route through a third pool."""
    columns = ["from", "to", "via", "rating", "direct", "chained", "discrepancy"]
    rows = []
    for source, target in permutations(POOLS, 2):
        direct = _published(results, source, target)
        if direct is None:
            continue
        for via in POOLS:
            if via in (source, target):
                continue
            first, second = _published(results, source, via), _published(results, via, target)
            if first is None or second is None:
                continue
            for rating in ratings:
                direct_value = float(direct(np.array([rating]))[0])
                chained_value = float(second(first(np.array([rating])))[0])
                rows.append(
                    {
                        "from": source.label,
                        "to": target.label,
                        "via": via.label,
                        "rating": rating,
                        "direct": direct_value,
                        "chained": chained_value,
                        "discrepancy": chained_value - direct_value,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _published(results: ConversionResults, source: Pool, target: Pool) -> Conversion | None:
    resolved = results.resolve(source, target)
    if resolved is None:
        return None
    fit, forward = resolved
    return fit.published(forward)


def source_agreement(results: ConversionResults, ratings: tuple[int, ...] = CHECK_RATINGS) -> pd.DataFrame:
    """The published fit next to each cross-check fit of the same pair."""
    columns = ["from", "to", "rating", "published_source", "published", "check_source", "check", "check_people", "difference"]
    rows = []
    for check in results.cross_checks:
        published = results.fits.get((check.x_pool.key, check.y_pool.key))
        if published is None:
            continue
        for rating in ratings:
            main_value = float(published.link(np.array([rating]))[0])
            check_value = float(check.link(np.array([rating]))[0])
            rows.append(
                {
                    "from": check.x_pool.label,
                    "to": check.y_pool.label,
                    "rating": rating,
                    "published_source": published.source,
                    "published": main_value,
                    "check_source": check.source,
                    "check": check_value,
                    "check_people": check.n,
                    "difference": check_value - main_value,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def accuracy_cross_check(results: ConversionResults, offsets: pd.DataFrame, ratings: tuple[int, ...] = CHECK_RATINGS) -> pd.DataFrame:
    """Linked-account conversion against accuracy equating, same cadence, conversion year."""
    columns = ["time_class", "chesscom_rating", "linked_accounts", "accuracy_equating", "accuracy_ci_low", "accuracy_ci_high", "difference"]
    rows = []
    year = offsets[(offsets["year"] == config.CONVERSION_YEAR) & (offsets["source"] == "chesscom")] if not offsets.empty else offsets
    for time_class in config.TIME_CLASSES:
        linked = _published(results, Pool("chesscom", time_class), Pool("lichess", time_class))
        for rating in ratings:
            match = year[(year["time_class"] == time_class) & (year["rating"] == rating)] if not year.empty else year
            if linked is None and match.empty:
                continue
            linked_value = float(linked(np.array([rating]))[0]) if linked is not None else np.nan
            equated = float(match["equivalent"].iloc[0]) if not match.empty else np.nan
            rows.append(
                {
                    "time_class": time_class,
                    "chesscom_rating": rating,
                    "linked_accounts": linked_value,
                    "accuracy_equating": equated,
                    "accuracy_ci_low": float(match["ci_low"].iloc[0]) if not match.empty else np.nan,
                    "accuracy_ci_high": float(match["ci_high"].iloc[0]) if not match.empty else np.nan,
                    "difference": equated - linked_value,
                }
            )
    return pd.DataFrame(rows, columns=columns)


# -- an outside benchmark ---------------------------------------------------

#: A published comparison, typed in from its source -- *not* computed from this
#: study's data, and labelled as such wherever it appears. ChessGoals' rating
#: comparison ("Updated July 2026"): a voluntary survey whose respondents'
#: usernames were matched across sites, Lichess RD below 150. Respondents: 10,156
#: with a chess.com rapid rating, 1,935 with Lichess blitz, 1,066 with Lichess
#: rapid. Keys are (from pool, to pool); values map a rating to its equivalent.
BENCHMARK_SOURCE = "ChessGoals rating comparison, updated July 2026 (https://chessgoals.com/rating-comparison/)"
BENCHMARK: dict[tuple[str, str], dict[int, int]] = {
    (CHESSCOM_BLITZ.key, LICHESS_BLITZ.key): {1000: 1425, 1500: 1755, 2000: 2080},
    (CHESSCOM_BLITZ.key, LICHESS_RAPID.key): {1000: 1615, 1500: 1905, 2000: 2165},
    (CHESSCOM_BLITZ.key, CHESSCOM_RAPID.key): {1000: 1255, 1500: 1655, 2000: 1995},
}


def benchmark_comparison(results: ConversionResults) -> pd.DataFrame:
    """This study's published conversion and typical-player prediction beside the benchmark's."""
    columns = ["from", "to", "rating", "benchmark", "this_study", "this_study_ci_low", "this_study_ci_high", "this_study_formula", "this_study_typical", "difference"]
    by_key = {pool.key: pool for pool in POOLS}
    rows = []
    for (source_key, target_key), values in BENCHMARK.items():
        source, target = by_key[source_key], by_key[target_key]
        resolved = results.resolve(source, target)
        ratings = np.array(sorted(values), dtype=float)
        converted = convert(*resolved, ratings) if resolved is not None else None
        for index, rating in enumerate(sorted(values)):
            ours = converted["equivalent"][index] if converted is not None else np.nan
            rows.append(
                {
                    "from": source.label,
                    "to": target.label,
                    "rating": rating,
                    "benchmark": values[rating],
                    "this_study": ours,
                    "this_study_ci_low": converted["ci_low"][index] if converted is not None else np.nan,
                    "this_study_ci_high": converted["ci_high"][index] if converted is not None else np.nan,
                    "this_study_formula": converted["formula"][index] if converted is not None else np.nan,
                    "this_study_typical": converted["typical"][index] if converted is not None else np.nan,
                    "difference": ours - values[rating],
                }
            )
    return pd.DataFrame(rows, columns=columns)
