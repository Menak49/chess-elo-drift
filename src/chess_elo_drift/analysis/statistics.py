"""The statistical machinery every comparison of the study shares.

Three things live here, because the year-by-year models, the site comparison
and the instrument check all need them and must not each grow their own copy:

* least squares with standard errors clustered by player. Sampling follows the
  opponent graph, so the same account appears more than once -- within a year
  and across years -- and treating those rows as independent would understate
  every uncertainty in the report;
* linear contrasts and Wald tests on such a fit, which is how a prediction at a
  fixed rating, or "did the rating slope change", gets an honest error bar;
* the descriptive band table and the cross-check of the recomputed accuracy
  against chess.com's own number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

#: Rows of a band-by-year table. Named so that an empty selection still yields
#: a table of the right shape: a thin sample is a normal early state here.
BAND_YEAR_COLUMNS = (
    "platform", "time_class", "band", "band_label", "year",
    "observations", "players", "mean", "sem",
)


@dataclass(frozen=True)
class OlsFit:
    """A least-squares fit with a cluster-robust covariance matrix."""

    coefficients: np.ndarray
    covariance: np.ndarray
    clusters: int
    observations: int

    @property
    def standard_errors(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def degrees_of_freedom(self) -> int:
        # With clustered errors the effective sample is the number of players.
        return max(self.clusters - 1, 1)

    def critical_value(self, level: float = 0.95) -> float:
        return float(stats.t.ppf(0.5 + level / 2, df=self.degrees_of_freedom))

    def contrast(self, weights: np.ndarray) -> "Estimate":
        """The linear combination `weights @ coefficients`, with its uncertainty."""
        weights = np.asarray(weights, dtype=float)
        value = float(weights @ self.coefficients)
        std_error = float(np.sqrt(max(weights @ self.covariance @ weights, 0.0)))
        return Estimate.from_value(value, std_error, self)

    def wald_p_value(self, restrictions: np.ndarray) -> float:
        """p-value of the joint hypothesis `restrictions @ coefficients == 0`.

        Uses the F form with the cluster count as denominator degrees of freedom,
        the usual small-sample choice for clustered errors.
        """
        restrictions = np.atleast_2d(np.asarray(restrictions, dtype=float))
        if restrictions.size == 0:
            return float("nan")
        difference = restrictions @ self.coefficients
        middle = restrictions @ self.covariance @ restrictions.T
        statistic = float(difference @ np.linalg.pinv(middle) @ difference)
        q = restrictions.shape[0]
        if self.degrees_of_freedom <= q:
            return float(stats.chi2.sf(statistic, df=q))
        return float(stats.f.sf(statistic / q, q, self.degrees_of_freedom))


@dataclass(frozen=True)
class Estimate:
    """One number with a 95% interval and a two-sided p-value against zero."""

    value: float
    std_error: float
    ci_low: float
    ci_high: float
    p_value: float

    @classmethod
    def from_value(cls, value: float, std_error: float, fit: OlsFit) -> "Estimate":
        critical = fit.critical_value()
        if std_error > 0:
            t_statistic = value / std_error
            p_value = float(2 * stats.t.sf(abs(t_statistic), df=fit.degrees_of_freedom))
        else:
            p_value = float("nan")
        return cls(value, std_error, value - critical * std_error, value + critical * std_error, p_value)

    def scaled(self, factor: float) -> "Estimate":
        """The same estimate in other units; the p-value does not change."""
        low, high = sorted((self.ci_low * factor, self.ci_high * factor))
        return Estimate(self.value * factor, abs(self.std_error * factor), low, high, self.p_value)


def ols_clustered(design: np.ndarray, outcome: np.ndarray, groups: np.ndarray) -> OlsFit:
    """Ordinary least squares with cluster-robust (CR1) standard errors."""
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residuals = outcome - design @ coefficients

    gram_inverse = np.linalg.pinv(design.T @ design)
    codes, unique_groups = pd.factorize(pd.Series(groups), sort=False)
    scores = design * residuals[:, None]
    # Summing each cluster's score rows at once is the whole "meat" of the
    # sandwich; a Python loop over players is two orders of magnitude slower.
    per_cluster = np.zeros((len(unique_groups), design.shape[1]))
    np.add.at(per_cluster, codes, scores)
    meat = per_cluster.T @ per_cluster

    n_obs, n_params = design.shape
    n_clusters = len(unique_groups)
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))
    covariance = gram_inverse @ meat @ gram_inverse * correction

    return OlsFit(
        coefficients=coefficients,
        covariance=covariance,
        clusters=n_clusters,
        observations=n_obs,
    )


def weighted_least_squares(design: np.ndarray, outcome: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Coefficients only, for bootstrap replicates expressed as row weights.

    Resampling players with replacement is the same as weighting each row by
    how many times its player was drawn, which avoids copying the data on every
    replicate.
    """
    root = np.sqrt(weights)[:, None]
    coefficients, *_ = np.linalg.lstsq(design * root, outcome * root[:, 0], rcond=None)
    return coefficients


def clustered_sem(cell: pd.DataFrame, metric: str) -> float:
    """Standard error of a cell mean, treating each player as one observation."""
    per_player = cell.groupby("player_id", observed=True)[metric].mean().to_numpy(dtype=float)
    if per_player.size < 2:
        return float("nan")
    return float(per_player.std(ddof=1) / np.sqrt(per_player.size))


def describe_band_by_year(sample: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Mean metric per site, cadence, rating band and year.

    The shape check beside the regressions: it assumes nothing about how
    accuracy depends on rating, at the price of comparing players who sit at
    different points inside a band.
    """
    rows: list[dict[str, object]] = []
    grouping = ["platform", "time_class", "band", "band_label", "year"]
    for (platform, time_class, band, label, year), cell in sample.groupby(grouping, observed=True):
        values = cell[metric].to_numpy(dtype=float)
        rows.append(
            {
                "platform": platform,
                "time_class": time_class,
                "band": int(band),
                "band_label": label,
                "year": int(year),
                "observations": int(values.size),
                "players": int(cell["player_id"].nunique()),
                "mean": round(float(values.mean()), 2),
                "sem": round(clustered_sem(cell, metric), 2),
            }
        )
    return pd.DataFrame(rows, columns=list(BAND_YEAR_COLUMNS))


def validate_against_reported(sample: pd.DataFrame) -> dict[str, float]:
    """Check the recomputed metric against chess.com's own, where both exist.

    The two scales differ by construction, so agreement is judged by how
    strongly they move together, not by how close the numbers are.
    """
    if "reported_accuracy" not in sample:
        return {"n": 0.0}
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
