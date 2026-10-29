"""The two extra parts of the findings: the single-clock comparison and the first study.

Part one compares years on 3+0 blitz and 15+10 rapid only, so the clock never
changes and nothing has to be corrected for it. Part three redoes the first
version of the study (chess.com, 2018-2019 against 2024-2025, no clock control)
on today's corpus, to show what chess.com's 2020 reclassification did to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.analysis import yearly
from chess_elo_drift.analysis.dataset import apply_inclusion_rules, sample_sizes
from chess_elo_drift.analysis.statistics import ols_clustered

logger = logging.getLogger(__name__)

#: The clock each cadence is reduced to in part one.
FOCUS_LABELS = {"blitz": "3+0", "rapid": "15+10"}

#: Ratings the single-clock curves are read at, inside the range both clocks cover.
FOCUS_RATINGS = (1000, 1400, 1800)


# -- part one: one clock per cadence -----------------------------------------


@dataclass
class FocusResults:
    sample: pd.DataFrame
    attrition: pd.DataFrame
    sizes: pd.DataFrame
    effects: pd.DataFrame
    levels: pd.DataFrame
    trends: pd.DataFrame


def focus_sample(evaluations: pd.DataFrame, max_per_player: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The estimation sample restricted to 3+0 and 15+10, and what each restriction removed."""
    if evaluations.empty:
        return apply_inclusion_rules(evaluations, max_per_player=max_per_player)
    controls = evaluations["time_control"].astype(str)
    on_clock = pd.Series(
        [
            control in config.FOCUS_TIME_CONTROLS.get(time_class, frozenset())
            for control, time_class in zip(controls, evaluations["time_class"])
        ],
        index=evaluations.index,
    )
    sample, attrition = apply_inclusion_rules(evaluations[on_clock], max_per_player=max_per_player)
    capped = sample[sample["rating"] <= config.FOCUS_RATING_MAX]
    attrition.loc[len(attrition)] = (
        f"rating up to {config.FOCUS_RATING_MAX} (15+10 is rare above)", len(sample) - len(capped), len(capped),
    )
    head = pd.DataFrame(
        [
            ("evaluated player-sides", 0, len(evaluations)),
            ("on the 3+0 or 15+10 clock", int((~on_clock).sum()), int(on_clock.sum())),
        ],
        columns=attrition.columns,
    )
    return capped.reset_index(drop=True), pd.concat([head, attrition.iloc[1:]], ignore_index=True)


def compute_focus(evaluations: pd.DataFrame, max_per_player: int) -> FocusResults:
    sample, attrition = focus_sample(evaluations, max_per_player)
    return FocusResults(
        sample=sample,
        attrition=attrition,
        sizes=sample_sizes(sample),
        effects=yearly.year_effects(sample),
        levels=yearly.accuracy_at_ratings(sample, FOCUS_RATINGS),
        trends=yearly.trends(sample, (yearly.MAIN,)),
    )


# -- part three: the first study, redone --------------------------------------

#: The first version of the study: chess.com only, ratings 600-2200, two eras.
FIRST_STUDY_PLATFORM = "chesscom"
FIRST_STUDY_RATING_MAX = 2200
FIRST_STUDY_ERAS = {"2018-2019": (2018, 2019), "2024-2025": (2024, 2025)}
FIRST_STUDY_BAND_WIDTH = 400

#: What the first report printed (commit fd0c436, 1,507 observations from 1,200 games).
FIRST_REPORT_GAPS = {"blitz": -1.338, "rapid": -1.393}

SPECS = {
    "original": "first study's method (no clock control)",
    "clock": "holding the clock fixed",
    "stable": "only time controls whose class never changed",
}

GAP_COLUMNS = ("time_class", "specification", "label", "gap", "ci_low", "ci_high", "p_value", "observations", "players")
COMPOSITION_COLUMNS = ("time_class", "era", "observations", "share_10_0", "mean_clock_minutes")
BAND_COLUMNS = ("time_class", "era", "band_label", "observations", "mean_accuracy", "sem")


@dataclass
class FirstStudyResults:
    gaps: pd.DataFrame
    composition: pd.DataFrame
    bands: pd.DataFrame
    sample: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


def first_study_sample(sample: pd.DataFrame) -> pd.DataFrame:
    """chess.com games of the two eras, tagged with their era."""
    if sample.empty:
        return sample.assign(era=pd.Series(dtype=object))
    rows = sample[(sample["platform"] == FIRST_STUDY_PLATFORM) & (sample["rating"] <= FIRST_STUDY_RATING_MAX)]
    era_of = {year: name for name, years in FIRST_STUDY_ERAS.items() for year in years}
    rows = rows[rows["year"].isin(era_of)]
    return rows.assign(era=rows["year"].map(era_of))


def compute_first_study(sample: pd.DataFrame) -> FirstStudyResults:
    rows = first_study_sample(sample)
    gaps, composition, bands = [], [], []
    modern = list(FIRST_STUDY_ERAS)[-1]
    for time_class in config.TIME_CLASSES:
        cell = rows[rows["time_class"] == time_class]
        if cell.empty or cell["era"].nunique() < 2:
            continue
        stable = cell[cell["time_control"].isin(config.STABLE_TIME_CONTROLS[time_class])]
        for name, data, clock in (("original", cell, False), ("clock", cell, True), ("stable", stable, False)):
            gap = _era_gap(data, modern, clock)
            if gap is not None:
                gaps.append({"time_class": time_class, "specification": name, "label": SPECS[name], **gap})
        for era, part in cell.groupby("era"):
            composition.append(
                {
                    "time_class": time_class,
                    "era": era,
                    "observations": len(part),
                    "share_10_0": float((part["time_control"] == "600").mean()),
                    "mean_clock_minutes": float(part["estimated_seconds"].mean() / 60.0),
                }
            )
        bands.extend(_era_bands(cell, time_class))
    return FirstStudyResults(
        gaps=pd.DataFrame(gaps, columns=list(GAP_COLUMNS)),
        composition=pd.DataFrame(composition, columns=list(COMPOSITION_COLUMNS)),
        bands=pd.DataFrame(bands, columns=list(BAND_COLUMNS)),
        sample=rows,
    )


def _era_gap(cell: pd.DataFrame, modern: str, clock: bool) -> dict[str, object] | None:
    """Later era minus earlier era in accuracy at the same rating (the first study's regression)."""
    if cell.empty or cell["era"].nunique() < 2:
        return None
    rating = cell["rating_c"].to_numpy(dtype=float)
    columns = [np.ones(len(cell)), (cell["era"] == modern).to_numpy(dtype=float), rating, rating**2]
    if clock and np.nanstd(cell["log_clock"]) > 1e-6:
        columns.append(cell["log_clock"].to_numpy(dtype=float))
    design = np.column_stack(columns)
    fit = ols_clustered(design, cell["accuracy"].to_numpy(dtype=float), cell["player_id"].to_numpy())
    estimate = fit.contrast(np.eye(design.shape[1])[1])
    return {
        "gap": estimate.value,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "p_value": estimate.p_value,
        "observations": len(cell),
        "players": cell["player_id"].nunique(),
    }


def _era_bands(cell: pd.DataFrame, time_class: str) -> list[dict[str, object]]:
    width = FIRST_STUDY_BAND_WIDTH
    top = FIRST_STUDY_RATING_MAX - width
    floor = (config.RATING_MIN + (cell["rating"] - config.RATING_MIN) // width * width).clip(upper=top)
    rows = []
    for (era, low), part in cell.assign(band=floor).groupby(["era", "band"]):
        high = FIRST_STUDY_RATING_MAX if low == top else low + width - 1
        rows.append(
            {
                "time_class": time_class,
                "era": era,
                "band_label": f"{low}-{high}",
                "observations": len(part),
                "mean_accuracy": float(part["accuracy"].mean()),
                "sem": float(part["accuracy"].std(ddof=1) / np.sqrt(len(part))) if len(part) > 1 else float("nan"),
            }
        )
    return rows
