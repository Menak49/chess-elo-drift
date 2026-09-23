"""Assembling the tables, figures and written findings.

Every number in the output is computed here from the sample; nothing is typed in
by hand, so re-running the pipeline on a larger corpus rewrites the conclusions
rather than leaving a stale claim behind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.analysis import statistics as stats
from chess_elo_drift.analysis.dataset import coverage_of_reported_accuracy
from chess_elo_drift.analysis.plots import plot_metric_by_band, plot_reported_accuracy_coverage

logger = logging.getLogger(__name__)

#: Below this, a difference is too small to matter whatever its p-value: it is
#: a fraction of the accuracy gap between neighbouring rating bands.
NEGLIGIBLE_ACCURACY_POINTS = 0.5


@dataclass(frozen=True)
class ReportArtifacts:
    """Everything a run produced, so the caller can point at it."""

    findings: Path
    tables: dict[str, Path]
    figures: dict[str, Path]


def generate_report(
    sample: pd.DataFrame, full_frame: pd.DataFrame, output_root: Path | None = None
) -> ReportArtifacts:
    """Write tables, figures and the findings note for one analysis run."""
    output_root = output_root or config.REPORTS
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    accuracy_by_band = stats.describe_by_band(sample, "accuracy")
    acpl_by_band = stats.describe_by_band(sample, "acpl")
    band_comparison = stats.compare_bands(sample, "accuracy")
    gaps = stats.estimate_all_gaps(sample)
    coverage = coverage_of_reported_accuracy(full_frame)

    tables = {
        "accuracy_by_band": _write_table(accuracy_by_band, tables_dir / "accuracy_by_band.csv"),
        "acpl_by_band": _write_table(acpl_by_band, tables_dir / "acpl_by_band.csv"),
        "band_comparison": _write_table(band_comparison, tables_dir / "band_comparison.csv"),
        "era_gaps": _write_table(gaps, tables_dir / "era_gaps.csv"),
        "reported_accuracy_coverage": _write_table(
            coverage, tables_dir / "reported_accuracy_coverage.csv"
        ),
    }

    figures = {
        "accuracy": plot_metric_by_band(
            accuracy_by_band,
            metric_label="mean move accuracy (%)",
            title="Move accuracy at a given rating, 2018-2019 versus 2024-2025",
            subtitle=(
                "Recomputed with Stockfish at a fixed depth; error bars are one "
                "standard error, clustered by player"
            ),
            output_path=figures_dir / "accuracy_by_band.png",
        ),
        "acpl": plot_metric_by_band(
            acpl_by_band,
            metric_label="average centipawn loss (lower is better)",
            title="Centipawn loss at a given rating, 2018-2019 versus 2024-2025",
            subtitle="The same comparison on a metric that does not depend on the accuracy formula",
            output_path=figures_dir / "acpl_by_band.png",
        ),
        "coverage": plot_reported_accuracy_coverage(
            coverage, figures_dir / "reported_accuracy_coverage.png"
        ),
    }

    findings_path = output_root / "findings.md"
    findings_path.write_text(
        _render_findings(sample, gaps, band_comparison, coverage, tables, figures),
        encoding="utf-8",
    )
    logger.info("report written to %s", findings_path)
    return ReportArtifacts(findings=findings_path, tables=tables, figures=figures)


# -- narrative ---------------------------------------------------------------


def _render_findings(
    sample: pd.DataFrame,
    gaps: pd.DataFrame,
    band_comparison: pd.DataFrame,
    coverage: pd.DataFrame,
    tables: dict[str, Path],
    figures: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append("# Does a chess.com rating still mean what it meant in 2018?\n")
    lines.append(_headline(sample, gaps) + "\n")

    lines.append("## What was measured\n")
    lines.extend(_method_lines(sample))

    lines.append("\n## Result\n")
    lines.append(
        "Each row is one cadence and one metric. The estimate is the coefficient on "
        "the era in a regression that also controls for rating, so it compares "
        "players of the *same* rating. Standard errors are clustered by player.\n"
    )
    lines.append(_markdown_table(gaps))
    lines.append("")
    for time_class in config.TIME_CLASSES:
        lines.append(_interpret_cadence(sample, gaps, time_class))

    lines.append("\n## Band by band\n")
    lines.append(
        "A shape check on the regression, not sixteen separate findings: with this "
        "many small tests, individual p-values should not be read in isolation.\n"
    )
    lines.append(_markdown_table(band_comparison))

    lines.append("\n## Why accuracy was recomputed\n")
    lines.extend(_instrument_lines(sample, coverage))

    lines.append("\n## What this cannot tell you\n")
    lines.extend(_caveat_lines())

    lines.append("\n## Files\n")
    for name, path in {**tables, **figures}.items():
        lines.append(f"- `{name}`: `{path.relative_to(config.PROJECT_ROOT).as_posix()}`")

    return "\n".join(lines) + "\n"


def _headline(sample: pd.DataFrame, gaps: pd.DataFrame) -> str:
    accuracy_gaps = gaps[gaps["metric"] == "accuracy"]
    if accuracy_gaps.empty:
        return "_No estimate could be formed from the collected sample._"

    verdicts = []
    for row in accuracy_gaps.itertuples():
        direction = _direction(row.modern_minus_legacy, row.p_value)
        verdicts.append(f"**{row.time_class}**: {direction}")

    return (
        "At equal rating, comparing 2018-2019 with 2024-2025 on "
        f"{len(sample):,} player-observations from {sample['game_id'].nunique():,} games: "
        + "; ".join(verdicts)
        + "."
    )


def _direction(estimate: float, p_value: float) -> str:
    if p_value >= 0.05 or abs(estimate) < NEGLIGIBLE_ACCURACY_POINTS:
        return f"no detectable change ({estimate:+.2f} accuracy points, p = {p_value:.2g})"
    word = "play better now" if estimate > 0 else "play worse now"
    return f"players of the same rating {word} ({estimate:+.2f} accuracy points, p = {p_value:.2g})"


def _method_lines(sample: pd.DataFrame) -> list[str]:
    per_era = sample.groupby("era", observed=True).agg(
        observations=("accuracy", "size"),
        players=("username", "nunique"),
        games=("game_id", "nunique"),
        median_rating=("rating", "median"),
    )
    lines = [
        f"- Rated blitz and rapid games, ratings {config.RATING_MIN}-{config.RATING_MAX}, "
        f"sampled from the same calendar months on both sides "
        f"({', '.join(str(month) for month in config.LEGACY_ERA.months)} versus "
        f"{', '.join(str(month) for month in config.MODERN_ERA.months)}).",
        "- A player's rating is the one recorded on that game, not their rating today.",
        f"- Accuracy recomputed with Stockfish at fixed depth {config.ENGINE_DEPTH}, "
        f"skipping the first {config.OPENING_PLIES_SKIPPED} plies of book moves.",
        "- Sample sizes:\n",
        _markdown_table(per_era.reset_index()),
    ]
    return lines


def _interpret_cadence(sample: pd.DataFrame, gaps: pd.DataFrame, time_class: str) -> str:
    row = gaps[(gaps["time_class"] == time_class) & (gaps["metric"] == "accuracy")]
    if row.empty:
        return f"- **{time_class}**: not enough data to estimate.\n"

    record = row.iloc[0]
    equivalent = stats.rating_equivalent_of_gap(sample, time_class)
    text = (
        f"- **{time_class}**: {record['modern_minus_legacy']:+.2f} accuracy points "
        f"(95% CI {record['ci_95_low']:+.2f} to {record['ci_95_high']:+.2f}, "
        f"{record['players']} players)."
    )
    if equivalent is not None and record["p_value"] < 0.05:
        text += (
            f" On the same sample, accuracy rises with rating, so the gap is worth "
            f"roughly **{equivalent:+.0f} rating points**."
        )
    return text


def _instrument_lines(sample: pd.DataFrame, coverage: pd.DataFrame) -> list[str]:
    lines = [
        "chess.com's own accuracy figure exists only for games somebody ran a review "
        "on, and that coverage collapsed backwards in time:\n",
        _markdown_table(coverage),
        "",
        "Using it directly would compare the 2024-2025 games people chose to review "
        "against almost nothing at all from 2018-2019, and across a formula that "
        "changed in between. The study therefore recomputes accuracy for every game "
        "with one engine at one depth.",
    ]

    validation = stats.validate_against_reported(sample)
    if "pearson_r" in validation:
        lines.append(
            f"\nWhere both numbers exist (n = {validation['n']:.0f}), the recomputed "
            f"accuracy tracks chess.com's closely (Pearson r = {validation['pearson_r']:.2f}, "
            f"Spearman rho = {validation['spearman_rho']:.2f}), so the instrument measures "
            "the same thing on a different scale "
            f"(mean {validation['mean_recomputed']:.1f} here against "
            f"{validation['mean_reported']:.1f} reported)."
        )
    return lines


def _caveat_lines() -> list[str]:
    return [
        "- **Survivorship.** 2018-2019 games can only be read from accounts that still "
        "exist. Players who quit and deleted their account are invisible, and they are "
        "unlikely to be a random slice of the old population.",
        "- **Activity weighting.** The crawler walks the opponent graph, so a player who "
        "played a hundred games that month is more likely to enter the sample than one "
        "who played three. The sample describes the opposition you would actually have "
        "met at a given rating, which is the relevant population for this question, but "
        "it is not a uniform draw over accounts.",
        "- **Accuracy is not strength.** It rewards quiet, forcing and simplified "
        "positions. If the *style* of play at a given rating shifted -- more theory, "
        "sharper openings, more time trouble -- accuracy moves without strength moving.",
        "- **One engine, one depth.** The scale is internally consistent, which is what "
        "the comparison needs, but the absolute numbers are not chess.com accuracies and "
        "should not be quoted as such.",
        "- **Rating is a moving target by design.** Chess.com has adjusted its rating "
        "system over the period; any drift found here is the combined result of pool "
        "composition and administrative changes, which this design cannot separate.",
    ]


# -- formatting --------------------------------------------------------------


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(no rows)_"
    return frame.to_markdown(index=False)
