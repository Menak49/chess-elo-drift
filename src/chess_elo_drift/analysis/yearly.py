"""Year by year: how much playing strength does a given rating buy?

One model family, fitted separately for every site and cadence -- a blitz and a
rapid rating are different scales, and so are the two sites', so none of the
four pools ever shares a coefficient with another:

    accuracy ~ C(year) + rating + rating² + log(clock) + C(month)   (clustered by player)

* **Rating enters continuously** (centred at 1500, per 100 points) so a year is
  never credited with skill just because its sample sits higher inside a band.
  The square lets the curve flatten at the top without committing to a shape.
* **The clock is a control, not a nuisance.** chess.com moved 10|0 from blitz to
  rapid on 2020-09-10, and the mix of time controls inside each cadence keeps
  shifting on both sites. More time buys accuracy, so without the control a
  year whose "blitz" was on average longer would look stronger. The control is
  `log(estimated_seconds / reference clock)`, which makes every prediction one
  for a 5+0 blitz or a 10+0 rapid game. The result without the control, and on
  chess.com's never-reclassified time controls alone, is reported beside it.
* **The calendar month is a control too.** Every year is sampled in March,
  June and September, but the crawl does not fill the three months equally
  (Lichess years collected before the per-month quotas came from a single
  month), so the month is held constant rather than trusted to cancel. It is
  deviation-coded: a prediction with every month term at zero is the average
  over the sampled months, not one of them.
* **The year effects are measured against the latest year** (the conversion
  year when present), so each reads "a rating in year Y bought this much more
  accuracy than the same rating does now".
* **A linear trend** (points per year) summarises the thirteen effects in one
  number with one interval; the per-year effects show whether a line is fair.
* **The rating slope may itself drift.** If it does, the year gap depends on
  the rating it is read at. A second model gives every year its own slope
  (sharing the curvature and the clock term); it produces the accuracy-at-1000/
  1500/2000 table and a joint test of "one slope for all years". The drift is
  also summarised as one rating × year coefficient, which is easier to read.
* **chess.com has a structural break.** The 2020 reclassification moved a whole
  population of 10|0 players into rapid (rapid went from under a tenth of the
  site's activity to about a third) and seeded their rapid rating from blitz. A
  single straight line across that break can be driven by the jump alone, so for
  chess.com the trend is also fitted separately before (to 2019) and after (from
  2021) it, leaving out 2020, whose September sample straddles the change.

Turning an accuracy gap into rating points divides it by the accuracy-per-rating
slope, which is small (a few tenths of a point per 100 rating). The ratio is
therefore reported as an order of magnitude and never as a figure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.analysis.dataset import RATING_CENTRE
from chess_elo_drift.analysis.statistics import Estimate, OlsFit, ols_clustered

logger = logging.getLogger(__name__)

#: Ratings the fitted curves are read at, spanning the study window.
REFERENCE_RATINGS: tuple[int, ...] = (1000, 1500, 2000)

#: A year with fewer rows than this is left out of its cell's model: its
#: dummy would be estimated from a handful of players and add noise, not signal.
MIN_YEAR_OBSERVATIONS = 30

#: The calendar-month control needs this many years sampled in two months or
#: more, each month with at least this many rows, to be estimated at all.
MIN_SEASON_YEARS = 2
MIN_SEASON_ROWS = 30

#: Below this, a change is too small to matter whatever its p-value: it is a
#: fraction of the accuracy gap between neighbouring 300-point bands.
NEGLIGIBLE_ACCURACY_POINTS = 0.5


#: chess.com reclassified 10|0 from blitz to rapid on 2020-09-10
#: (https://www.chess.com/news/view/10-minute-chess-now-rapid-rated-bullet-ratings-increased).
#: 2020 itself straddles the change, so the segments stop before it and resume after.
CHESSCOM_RECLASSIFICATION = "2020-09-10"
LAST_YEAR_BEFORE_BREAK = 2019
FIRST_YEAR_AFTER_BREAK = 2021

#: Lichess's rating floor was 800 until mid-2019, then 600, then 400; a floor
#: holds weak players' ratings up, so the bottom of the window is not
#: comparable across those years. From 1000 up, no floor was ever close.
ABOVE_FLOORS_RATING = 1000


@dataclass(frozen=True)
class Specification:
    """One way of fitting the trend, for the robustness table."""

    name: str
    label: str
    metric: str = "accuracy"
    clock: bool = True
    seasons: bool = True
    stable_controls_only: bool = False
    one_row_per_player: bool = False
    min_rating: int | None = None
    years: tuple[int, int] | None = None
    platforms: tuple[str, ...] = config.PLATFORMS


MAIN = Specification("main", "main model (clock-controlled)")
BEFORE_BREAK = Specification(
    "before_break", f"chess.com before the 10|0 reclassification (to {LAST_YEAR_BEFORE_BREAK})",
    years=(config.FIRST_YEAR, LAST_YEAR_BEFORE_BREAK), platforms=("chesscom",),
)
AFTER_BREAK = Specification(
    "after_break", f"chess.com after the 10|0 reclassification (from {FIRST_YEAR_AFTER_BREAK})",
    years=(FIRST_YEAR_AFTER_BREAK, config.LAST_YEAR), platforms=("chesscom",),
)
SPECIFICATIONS: tuple[Specification, ...] = (
    MAIN,
    Specification("no_clock", "without the clock control", clock=False),
    Specification("no_seasons", "without the calendar-month control", seasons=False),
    Specification(
        "stable_controls", "chess.com time controls never reclassified",
        stable_controls_only=True, platforms=("chesscom",),
    ),
    BEFORE_BREAK,
    AFTER_BREAK,
    Specification("one_per_player", "one game per player and year", one_row_per_player=True),
    Specification(
        "above_floors", f"ratings {ABOVE_FLOORS_RATING}+ only (clear of every rating floor)",
        min_rating=ABOVE_FLOORS_RATING,
    ),
    Specification("acpl", "average centipawn loss instead of accuracy", metric="acpl"),
)

EFFECT_COLUMNS = (
    "platform", "time_class", "year", "reference_year", "effect", "std_error", "ci_low",
    "ci_high", "p_value", "slope_per_100", "rating_points_equivalent", "observations", "players",
)
LEVEL_COLUMNS = ("platform", "time_class", "year", "rating", "accuracy", "ci_low", "ci_high", "observations")
TREND_COLUMNS = (
    "platform", "time_class", "specification", "label", "metric", "trend_per_year", "std_error",
    "ci_low", "ci_high", "p_value", "first_year", "last_year", "years", "observations", "players",
    "slope_per_100", "rating_points_per_year",
)
SLOPE_COLUMNS = (
    "platform", "time_class", "slope_change_per_year", "ci_low", "ci_high", "p_value",
    "equal_slopes_p_value", "first_year", "slope_first_year", "last_year", "slope_last_year",
)


# -- cells ------------------------------------------------------------------


def cells(sample: pd.DataFrame) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Every (site, cadence) with at least two usable years, in report order."""
    for platform in config.PLATFORMS:
        for time_class in config.TIME_CLASSES:
            cell = sample[(sample["platform"] == platform) & (sample["time_class"] == time_class)]
            cell = usable_years(cell)
            if cell["year"].nunique() >= 2:
                yield platform, time_class, cell


def usable_years(cell: pd.DataFrame) -> pd.DataFrame:
    counts = cell["year"].value_counts()
    return cell[cell["year"].isin(counts[counts >= MIN_YEAR_OBSERVATIONS].index)]


def reference_year(years: list[int]) -> int:
    return config.CONVERSION_YEAR if config.CONVERSION_YEAR in years else max(years)


def restrict(cell: pd.DataFrame, spec: Specification) -> pd.DataFrame:
    """The rows a specification is fitted on."""
    if spec.years is not None:
        cell = cell[cell["year"].between(*spec.years)]
    if spec.min_rating is not None:
        cell = cell[cell["rating"] >= spec.min_rating]
    if spec.stable_controls_only and not cell.empty:
        stable = cell["time_control"].isin(config.STABLE_TIME_CONTROLS.get(str(cell["time_class"].iloc[0]), ()))
        cell = cell[stable & (cell["platform"] == "chesscom")]
    if spec.one_row_per_player:
        ordered = cell.sort_values(["player_id", "year", "game_id"], kind="stable")
        cell = ordered[ordered.groupby(["player_id", "year"]).cumcount() == 0]
    return usable_years(cell.dropna(subset=[spec.metric]))


# -- design -----------------------------------------------------------------


@dataclass(frozen=True)
class Fitted:
    """A fit together with the names of its columns."""

    fit: OlsFit
    names: tuple[str, ...]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def weights(self, **terms: float) -> np.ndarray:
        vector = np.zeros(len(self.names))
        for name, value in terms.items():
            if name in self.names:
                vector[self.index(name)] = value
        return vector

    def coefficient(self, name: str) -> Estimate:
        return self.fit.contrast(self.weights(**{name: 1.0}))


def _controls(cell: pd.DataFrame, clock: bool, seasons: bool = True) -> dict[str, np.ndarray]:
    rating = cell["rating_c"].to_numpy(dtype=float)
    columns = {"rating": rating, "rating_sq": rating**2}
    log_clock = cell["log_clock"].to_numpy(dtype=float)
    # A cell played at a single clock has nothing to control for, and the
    # column would only duplicate the intercept.
    if clock and np.nanstd(log_clock) > 1e-6:
        columns["log_clock"] = log_clock
    if seasons:
        columns.update(_season_columns(cell))
    return columns


def _season_columns(cell: pd.DataFrame) -> dict[str, np.ndarray]:
    """Deviation-coded calendar months: the first month sampled is -1 on every term.

    A month's effect is learnt from the years that sampled it beside another
    month, so it is only controlled for when at least `MIN_SEASON_YEARS` years
    hold `MIN_SEASON_ROWS` rows in two months or more. Below that the estimate
    rests on a handful of games (seven March games in a September year were
    enough to move every September year by several points) and the control adds more
    noise than it removes. A cell drawn from a single month gets no column,
    like a single clock.
    """
    calendar_month = pd.to_numeric(cell["month"].str[5:7], errors="coerce")
    rows = pd.DataFrame({"year": cell["year"].to_numpy(), "month": calendar_month.to_numpy()}).dropna()
    counts = rows.value_counts()
    covered = counts[counts >= MIN_SEASON_ROWS].reset_index()
    mixed_years = (covered.groupby("year")["month"].nunique() >= 2).sum()
    if mixed_years < MIN_SEASON_YEARS:
        return {}
    present = sorted(int(month) for month in rows["month"].unique())
    baseline = present[0]
    values = calendar_month.to_numpy()
    columns = {}
    for month in present[1:]:
        column = (values == month).astype(float) - (values == baseline).astype(float)
        columns[f"month_{month:02d}"] = column
    return columns


def _fit(cell: pd.DataFrame, columns: dict[str, np.ndarray], metric: str) -> Fitted:
    """Least squares on `columns`, without month terms the design cannot identify.

    A month's effect is only learnt from years that sampled it beside another
    month. When a pool's years each come from one month, the month terms are a
    combination of the year terms, and the minimum-norm solution then shares
    the year levels out between them at random. They are dropped instead: the
    year effects are then confounded with the season, which is logged.
    """
    design = np.column_stack(list(columns.values()))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        seasons = [name for name in columns if name.startswith("month_")]
        if seasons:
            logger.warning(
                "%s %s: calendar months cannot be separated from years (%s), fitted without them",
                cell["platform"].iloc[0], cell["time_class"].iloc[0], ", ".join(seasons),
            )
            columns = {name: column for name, column in columns.items() if name not in seasons}
            design = np.column_stack(list(columns.values()))
    fit = ols_clustered(design, cell[metric].to_numpy(dtype=float), cell["player_id"].to_numpy())
    return Fitted(fit, tuple(columns))


def fit_year_effects(
    cell: pd.DataFrame, *, clock: bool = True, metric: str = "accuracy", seasons: bool = True
) -> tuple[Fitted, int]:
    """The main model: one intercept shift per year, common rating curve and clock term."""
    years = sorted(cell["year"].unique())
    reference = reference_year(years)
    columns = {"const": np.ones(len(cell))}
    for year in years:
        if year != reference:
            columns[f"year_{year}"] = (cell["year"] == year).to_numpy(dtype=float)
    columns.update(_controls(cell, clock, seasons))
    return _fit(cell, columns, metric), reference


def fit_trend(
    cell: pd.DataFrame,
    *,
    clock: bool = True,
    metric: str = "accuracy",
    slope_drift: bool = False,
    seasons: bool = True,
) -> Fitted:
    """Year as a straight line; optionally let the rating slope drift with it."""
    year_c = (cell["year"] - reference_year(sorted(cell["year"].unique()))).to_numpy(dtype=float)
    columns = {"const": np.ones(len(cell)), "year": year_c}
    columns.update(_controls(cell, clock, seasons))
    if slope_drift:
        columns["rating_x_year"] = cell["rating_c"].to_numpy(dtype=float) * year_c
    return _fit(cell, columns, metric)


def fit_year_slopes(cell: pd.DataFrame, *, clock: bool = True) -> Fitted:
    """Every year its own level and its own rating slope; shared curvature and clock."""
    columns: dict[str, np.ndarray] = {}
    rating = cell["rating_c"].to_numpy(dtype=float)
    for year in sorted(cell["year"].unique()):
        in_year = (cell["year"] == year).to_numpy(dtype=float)
        columns[f"level_{year}"] = in_year
        columns[f"slope_{year}"] = in_year * rating
    controls = _controls(cell, clock)
    controls.pop("rating")
    columns.update(controls)
    return _fit(cell, columns, "accuracy")


# -- tables -----------------------------------------------------------------


def year_effects(sample: pd.DataFrame) -> pd.DataFrame:
    """Each year's accuracy at a fixed rating, minus the reference year's."""
    rows: list[dict[str, object]] = []
    for platform, time_class, cell in cells(sample):
        fitted, reference = fit_year_effects(cell)
        slope = fitted.coefficient("rating").value
        for year in sorted(cell["year"].unique()):
            in_year = cell[cell["year"] == year]
            if year == reference:
                effect = Estimate(0.0, 0.0, float("nan"), float("nan"), float("nan"))
            else:
                effect = fitted.coefficient(f"year_{year}")
            rows.append(
                {
                    "platform": platform,
                    "time_class": time_class,
                    "year": int(year),
                    "reference_year": int(reference),
                    "effect": effect.value,
                    "std_error": effect.std_error,
                    "ci_low": effect.ci_low,
                    "ci_high": effect.ci_high,
                    "p_value": effect.p_value,
                    "slope_per_100": slope,
                    "rating_points_equivalent": _rating_points(effect.value, slope),
                    "observations": len(in_year),
                    "players": in_year["player_id"].nunique(),
                }
            )
    return pd.DataFrame(rows, columns=list(EFFECT_COLUMNS))


def accuracy_at_ratings(sample: pd.DataFrame, ratings: tuple[int, ...] = REFERENCE_RATINGS) -> pd.DataFrame:
    """Predicted accuracy at fixed ratings, per year, from the per-year-slope model."""
    rows: list[dict[str, object]] = []
    for platform, time_class, cell in cells(sample):
        fitted = fit_year_slopes(cell)
        for year in sorted(cell["year"].unique()):
            for rating in ratings:
                r = (rating - RATING_CENTRE) / 100.0
                estimate = fitted.fit.contrast(
                    fitted.weights(**{f"level_{year}": 1.0, f"slope_{year}": r, "rating_sq": r * r})
                )
                rows.append(
                    {
                        "platform": platform,
                        "time_class": time_class,
                        "year": int(year),
                        "rating": rating,
                        "accuracy": estimate.value,
                        "ci_low": estimate.ci_low,
                        "ci_high": estimate.ci_high,
                        "observations": int((cell["year"] == year).sum()),
                    }
                )
    return pd.DataFrame(rows, columns=list(LEVEL_COLUMNS))


def trends(sample: pd.DataFrame, specifications: tuple[Specification, ...] = SPECIFICATIONS) -> pd.DataFrame:
    """The linear trend per site and cadence, under every specification."""
    rows: list[dict[str, object]] = []
    for platform in config.PLATFORMS:
        for time_class in config.TIME_CLASSES:
            base = sample[(sample["platform"] == platform) & (sample["time_class"] == time_class)]
            for spec in specifications:
                if platform not in spec.platforms:
                    continue
                cell = restrict(base, spec) if not base.empty else base
                if cell.empty or cell["year"].nunique() < 2:
                    continue
                rows.append(_trend_row(platform, time_class, spec, cell))
    return pd.DataFrame(rows, columns=list(TREND_COLUMNS))


def _trend_row(platform: str, time_class: str, spec: Specification, cell: pd.DataFrame) -> dict[str, object]:
    fitted = fit_trend(cell, clock=spec.clock, metric=spec.metric, seasons=spec.seasons)
    trend = fitted.coefficient("year")
    slope = fitted.coefficient("rating").value
    years = sorted(cell["year"].unique())
    return {
        "platform": platform,
        "time_class": time_class,
        "specification": spec.name,
        "label": spec.label,
        "metric": spec.metric,
        "trend_per_year": trend.value,
        "std_error": trend.std_error,
        "ci_low": trend.ci_low,
        "ci_high": trend.ci_high,
        "p_value": trend.p_value,
        "first_year": int(years[0]),
        "last_year": int(years[-1]),
        "years": len(years),
        "observations": len(cell),
        "players": cell["player_id"].nunique(),
        "slope_per_100": slope,
        "rating_points_per_year": _rating_points(trend.value, slope),
    }


def slope_drift(sample: pd.DataFrame) -> pd.DataFrame:
    """Whether the accuracy-per-rating slope changes over the years."""
    rows: list[dict[str, object]] = []
    for platform, time_class, cell in cells(sample):
        drift = fit_trend(cell, slope_drift=True).coefficient("rating_x_year")
        per_year = fit_year_slopes(cell)
        years = sorted(cell["year"].unique())
        restrictions = [
            per_year.weights(**{f"slope_{year}": 1.0, f"slope_{years[-1]}": -1.0}) for year in years[:-1]
        ]
        rows.append(
            {
                "platform": platform,
                "time_class": time_class,
                "slope_change_per_year": drift.value,
                "ci_low": drift.ci_low,
                "ci_high": drift.ci_high,
                "p_value": drift.p_value,
                "equal_slopes_p_value": per_year.fit.wald_p_value(np.array(restrictions)),
                "first_year": int(years[0]),
                "slope_first_year": per_year.coefficient(f"slope_{years[0]}").value,
                "last_year": int(years[-1]),
                "slope_last_year": per_year.coefficient(f"slope_{years[-1]}").value,
            }
        )
    return pd.DataFrame(rows, columns=list(SLOPE_COLUMNS))


def _rating_points(accuracy_points: float, slope_per_100: float) -> float:
    """An accuracy difference re-expressed in rating points, via the fitted slope.

    Only an order of magnitude: the slope is a few tenths of a point per 100
    rating and sits in the denominator, so a modest error in it moves the ratio
    a lot. A slope that is not clearly positive gives no conversion at all.
    """
    if not np.isfinite(slope_per_100) or slope_per_100 <= 0.05:
        return float("nan")
    return accuracy_points / slope_per_100 * 100.0


def verdict(trend_row: pd.Series) -> str:
    """One phrase for the headline, from a row of the trend table."""
    span = trend_row["last_year"] - trend_row["first_year"]
    total = trend_row["trend_per_year"] * span
    if trend_row["p_value"] >= 0.05 or abs(total) < NEGLIGIBLE_ACCURACY_POINTS:
        return "no detectable change"
    return "a given rating buys less accuracy each year" if total < 0 else "a given rating buys more accuracy each year"
