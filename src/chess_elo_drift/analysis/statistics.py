"""The comparisons the study rests on.

Two views of the same question, deliberately kept side by side:

* band by band, which is easy to read and makes no assumption about shape;
* one regression per cadence, which uses rating as a continuous control and so
  is not fooled by players sitting at different points *within* a band -- the
  exact artefact that a study of rating drift has to rule out.

Standard errors are clustered by player throughout. Sampling follows the
opponent graph, so the same account can appear more than once, and treating
those rows as independent would understate the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from chess_elo_drift import config


#: The columns each table carries. Naming them means a comparison that selects
#: nothing still returns an empty table of the right shape, rather than a
#: column-less frame that blows up on the next `sort_values`. A thin sample is
#: a normal early state of this study, not an error.
_BAND_SUMMARY_COLUMNS = (
    "time_class", "band", "band_label", "era", "observations", "players", "mean", "std", "sem",
)
_BAND_COMPARISON_COLUMNS = (
    "time_class", "band", "band_label", "n_legacy", "n_modern",
    "mean_legacy", "mean_modern", "difference", "p_value",
)
#: The keys `GapEstimate.as_row` emits, which are report labels rather than the
#: dataclass field names.
_GAP_COLUMNS = (
    "time_class", "metric", "modern_minus_legacy", "std_error", "t", "p_value",
    "ci_95_low", "ci_95_high", "observations", "players",
)


@dataclass(frozen=True)
class GapEstimate:
    """The modern-minus-legacy difference in one metric, with its uncertainty."""

    time_class: str
    metric: str
    estimate: float
    standard_error: float
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    observations: int
    clusters: int

    def as_row(self) -> dict[str, object]:
        return {
            "time_class": self.time_class,
            "metric": self.metric,
            "modern_minus_legacy": round(self.estimate, 3),
            "std_error": round(self.standard_error, 3),
            "t": round(self.t_statistic, 2),
            "p_value": float(f"{self.p_value:.2}"),
            "ci_95_low": round(self.ci_low, 3),
            "ci_95_high": round(self.ci_high, 3),
            "observations": self.observations,
            "players": self.clusters,
        }


def describe_by_band(sample: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Mean metric per era, cadence and rating band, with a clustered error bar."""
    rows: list[dict[str, object]] = []
    grouping = ["time_class", "band", "band_label", "era"]
    for (time_class, band, label, era), cell in sample.groupby(grouping, observed=True):
        values = cell[metric].to_numpy(dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "time_class": time_class,
                "band": band,
                "band_label": label,
                "era": era,
                "observations": int(values.size),
                "players": int(cell["username"].nunique()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                "sem": _clustered_sem(cell, metric),
            }
        )
    return (
        pd.DataFrame(rows, columns=list(_BAND_SUMMARY_COLUMNS))
        .sort_values(["time_class", "band", "era"])
        .reset_index(drop=True)
    )


def compare_bands(sample: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Band-by-band era difference, with a Welch test on each band.

    These are many small tests; they are meant as a shape check on the headline
    regression, not as sixteen independent findings.
    """
    rows: list[dict[str, object]] = []
    for (time_class, band, label), cell in sample.groupby(
        ["time_class", "band", "band_label"], observed=True
    ):
        legacy = cell.loc[cell["era"] == config.LEGACY_ERA.name, metric].to_numpy(dtype=float)
        modern = cell.loc[cell["era"] == config.MODERN_ERA.name, metric].to_numpy(dtype=float)
        if legacy.size < 2 or modern.size < 2:
            continue
        test = stats.ttest_ind(modern, legacy, equal_var=False)
        rows.append(
            {
                "time_class": time_class,
                "band": band,
                "band_label": label,
                "n_legacy": int(legacy.size),
                "n_modern": int(modern.size),
                "mean_legacy": round(float(legacy.mean()), 2),
                "mean_modern": round(float(modern.mean()), 2),
                "difference": round(float(modern.mean() - legacy.mean()), 2),
                "p_value": float(f"{test.pvalue:.2}"),
            }
        )
    return (
        pd.DataFrame(rows, columns=list(_BAND_COMPARISON_COLUMNS))
        .sort_values(["time_class", "band"])
        .reset_index(drop=True)
    )


def estimate_gap(
    sample: pd.DataFrame, time_class: str, metric: str = "accuracy", *, quadratic: bool = True
) -> GapEstimate | None:
    """Fit `metric ~ era + rating (+ rating squared)` for one cadence.

    The era coefficient is the study's answer: how much better or worse a player
    of the *same rating* plays now than then. Rating enters continuously so the
    estimate cannot be driven by the two eras sitting at different places inside
    the same band.
    """
    cell = sample[sample["time_class"] == time_class]
    if cell["era"].nunique() < 2 or len(cell) < 20:
        return None

    rating = cell["rating_centred"].to_numpy(dtype=float) / 100.0
    columns = [np.ones(len(cell)), cell["is_modern"].to_numpy(dtype=float), rating]
    if quadratic:
        columns.append(rating**2)

    design = np.column_stack(columns)
    outcome = cell[metric].to_numpy(dtype=float)
    fit = _ols_with_clustered_errors(design, outcome, cell["username"].to_numpy())

    index = 1  # the era dummy
    estimate = float(fit.coefficients[index])
    std_error = float(fit.standard_errors[index])
    t_statistic = estimate / std_error if std_error else float("nan")
    p_value = float(2 * stats.t.sf(abs(t_statistic), df=max(fit.clusters - 1, 1)))
    critical = float(stats.t.ppf(0.975, df=max(fit.clusters - 1, 1)))

    return GapEstimate(
        time_class=time_class,
        metric=metric,
        estimate=estimate,
        standard_error=std_error,
        t_statistic=t_statistic,
        p_value=p_value,
        ci_low=estimate - critical * std_error,
        ci_high=estimate + critical * std_error,
        observations=len(cell),
        clusters=fit.clusters,
    )


def estimate_all_gaps(sample: pd.DataFrame, metrics: tuple[str, ...] = ("accuracy", "acpl")) -> pd.DataFrame:
    """Run the headline model for every cadence and metric."""
    estimates = [
        estimate_gap(sample, time_class, metric)
        for time_class in config.TIME_CLASSES
        for metric in metrics
    ]
    rows = [e.as_row() for e in estimates if e is not None]
    return pd.DataFrame(rows, columns=list(_GAP_COLUMNS))


def accuracy_slope_per_100(sample: pd.DataFrame, time_class: str) -> tuple[float, float] | None:
    """How much accuracy rises per 100 rating points, and that slope's error.

    Returned alongside any conversion into rating points, because it is the
    denominator of that conversion and the reader cannot judge the result
    without it.
    """
    cell = sample[sample["time_class"] == time_class]
    if len(cell) < 20 or cell["era"].nunique() < 2:
        return None

    rating = cell["rating_centred"].to_numpy(dtype=float) / 100.0
    design = np.column_stack([np.ones(len(cell)), cell["is_modern"].to_numpy(dtype=float), rating])
    fit = _ols_with_clustered_errors(
        design, cell["accuracy"].to_numpy(dtype=float), cell["username"].to_numpy()
    )
    return float(fit.coefficients[2]), float(fit.standard_errors[2])


def rating_equivalent_of_gap(sample: pd.DataFrame, time_class: str) -> float | None:
    """Express the era gap in rating points.

    Accuracy rises with rating at some slope; dividing the era gap by that slope
    says how many rating points the difference is worth, which is the form the
    original question was asked in.

    Read it as an order of magnitude, never as a figure. The slope is shallow --
    a few tenths of an accuracy point per 100 rating -- so it sits in the
    denominator of a ratio whose own uncertainty is large, and two cadences with
    near-identical accuracy gaps can convert to rating gaps that differ by a
    factor of two. `accuracy_slope_per_100` returns the denominator so the
    conversion can be reported with the slope that produced it.
    """
    gap = estimate_gap(sample, time_class, "accuracy")
    slope = accuracy_slope_per_100(sample, time_class)
    if gap is None or slope is None:
        return None

    slope_per_100, _ = slope
    if abs(slope_per_100) < 1e-9:
        return None
    return gap.estimate / slope_per_100 * 100.0


def validate_against_reported(sample: pd.DataFrame) -> dict[str, float]:
    """Check the recomputed metric against chess.com's own, where both exist.

    The two scales differ by construction, so agreement is judged by how
    strongly they move together, not by how close the numbers are.
    """
    both = sample.dropna(subset=["reported_accuracy", "accuracy"])
    if len(both) < 30:
        return {"n": float(len(both))}
    pearson = stats.pearsonr(both["accuracy"], both["reported_accuracy"])
    spearman = stats.spearmanr(both["accuracy"], both["reported_accuracy"])
    return {
        "n": float(len(both)),
        "pearson_r": float(pearson.statistic),
        "spearman_rho": float(spearman.statistic),
        "mean_recomputed": float(both["accuracy"].mean()),
        "mean_reported": float(both["reported_accuracy"].mean()),
    }


# -- least squares with clustered standard errors -----------------------------


@dataclass(frozen=True)
class _OlsFit:
    coefficients: np.ndarray
    standard_errors: np.ndarray
    clusters: int


def _ols_with_clustered_errors(design: np.ndarray, outcome: np.ndarray, groups: np.ndarray) -> _OlsFit:
    """Ordinary least squares with cluster-robust (CR1) standard errors."""
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residuals = outcome - design @ coefficients

    gram_inverse = np.linalg.pinv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]))
    unique_groups = np.unique(groups)
    for group in unique_groups:
        mask = groups == group
        contribution = design[mask].T @ residuals[mask]
        meat += np.outer(contribution, contribution)

    n_obs, n_params = design.shape
    n_clusters = len(unique_groups)
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))
    covariance = gram_inverse @ meat @ gram_inverse * correction

    return _OlsFit(
        coefficients=coefficients,
        standard_errors=np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        clusters=n_clusters,
    )


def _clustered_sem(cell: pd.DataFrame, metric: str) -> float:
    """Standard error of a cell mean, treating each player as one observation."""
    per_player = cell.groupby("username", observed=True)[metric].mean().to_numpy(dtype=float)
    if per_player.size < 2:
        return float("nan")
    return float(per_player.std(ddof=1) / np.sqrt(per_player.size))
