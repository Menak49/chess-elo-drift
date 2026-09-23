"""Figures for the report.

One chart per question, each one a small multiple split by cadence -- blitz and
rapid are different rating scales and are never pooled, so they never share an
axis either. Colour carries the era and nothing else, and every series is also
named in the legend, so the figures survive being printed in grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from chess_elo_drift import config

#: Era colours, slots 1 and 2 of a palette validated for colour-vision
#: deficiency against a light surface.
ERA_COLOURS = {
    config.LEGACY_ERA.name: "#2a78d6",
    config.MODERN_ERA.name: "#eb6834",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

_MARKERS = {config.LEGACY_ERA.name: "o", config.MODERN_ERA.name: "s"}


def plot_metric_by_band(
    summary: pd.DataFrame,
    *,
    metric_label: str,
    title: str,
    subtitle: str,
    output_path: Path,
) -> Path:
    """Draw the metric against rating band, one panel per cadence.

    Error bars are one clustered standard error either side of the mean: each
    player counts once inside a cell, so a prolific account cannot shrink the
    interval it sits in.
    """
    cadences = [tc for tc in config.TIME_CLASSES if tc in set(summary["time_class"])]
    if not cadences:
        return _placeholder(
            output_path, title, "No observations survived the inclusion rules yet."
        )

    figure, axes = plt.subplots(
        1, len(cadences), figsize=(5.6 * len(cadences), 5.0), sharey=True, facecolor=SURFACE
    )
    axes = axes if len(cadences) > 1 else [axes]

    for axis, time_class in zip(axes, cadences):
        panel = summary[summary["time_class"] == time_class]
        for era, series in panel.groupby("era", observed=True):
            series = series.sort_values("band")
            axis.errorbar(
                series["band"],
                series["mean"],
                yerr=series["sem"],
                label=era,
                color=ERA_COLOURS.get(str(era), INK),
                marker=_MARKERS.get(str(era), "o"),
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                linewidth=2,
                capsize=4,
                elinewidth=1.2,
            )
        _style_axis(axis, time_class, panel)

    axes[0].set_ylabel(metric_label, color=INK_MUTED, fontsize=11)
    axes[0].legend(frameon=False, loc="best", fontsize=10, labelcolor=INK)

    # Explicit y for both: the default suptitle position sits close enough to the
    # subtitle that the two descenders touch at this figure height.
    figure.suptitle(title, fontsize=15, color=INK, x=0.02, y=0.975, ha="left", weight="medium")
    figure.text(0.02, 0.9, subtitle, fontsize=10.5, color=INK_MUTED, ha="left", va="top")
    figure.tight_layout(rect=(0, 0, 1, 0.87))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return output_path


def plot_reported_accuracy_coverage(coverage: pd.DataFrame, output_path: Path) -> Path:
    """Show how unusable the API's own accuracy field is for a time comparison."""
    if coverage.empty:
        return _placeholder(
            output_path,
            "chess.com accuracy coverage",
            "No evaluated games yet.",
        )

    figure, axis = plt.subplots(figsize=(7.4, 4.4), facecolor=SURFACE)

    labels = [f"{row.time_class}\n{row.era}" for row in coverage.itertuples()]
    positions = range(len(labels))
    colours = [ERA_COLOURS.get(str(row.era), INK) for row in coverage.itertuples()]

    bars = axis.bar(
        positions, coverage["coverage"] * 100, color=colours, width=0.62, zorder=3
    )
    for bar, row in zip(bars, coverage.itertuples()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{row.coverage:.0%}\nn={row.observations}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK_MUTED,
        )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=10, color=INK_MUTED)
    axis.set_ylabel("share of games carrying an accuracy score", color=INK_MUTED, fontsize=11)
    axis.set_ylim(0, max(60, float(coverage["coverage"].max() * 100) + 18))
    _clean_spines(axis)
    axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)

    figure.suptitle(
        "chess.com only publishes an accuracy score for games somebody reviewed",
        fontsize=13.5, color=INK, x=0.02, y=0.975, ha="left", weight="medium",
    )
    figure.text(
        0.02, 0.885,
        "Why the study recomputes accuracy instead of reading it from the API",
        fontsize=10.5, color=INK_MUTED, ha="left", va="top",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return output_path


def _placeholder(output_path: Path, title: str, message: str) -> Path:
    """A figure that says there is nothing to draw.

    A run on a sample too thin to plot is a normal early state of the study, and
    the report should say so on the page rather than fail on the way to writing
    it.
    """
    figure, axis = plt.subplots(figsize=(7.4, 4.0), facecolor=SURFACE)
    axis.axis("off")
    axis.text(0.5, 0.55, message, ha="center", va="center", fontsize=12, color=INK_MUTED)
    figure.suptitle(title, fontsize=13.5, color=INK, x=0.02, ha="left", weight="medium")
    figure.tight_layout(rect=(0, 0, 1, 0.9))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return output_path


def _style_axis(axis, time_class: str, panel: pd.DataFrame) -> None:
    axis.set_title(time_class, fontsize=12, color=INK, pad=10)
    axis.set_xlabel("rating at the time of the game", color=INK_MUTED, fontsize=11)
    ticks = sorted(panel["band"].unique())
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [f"{band}-{band + config.REPORT_BAND_WIDTH - 1}" for band in ticks],
        fontsize=9.5, color=INK_MUTED, rotation=20,
    )
    axis.tick_params(axis="y", labelsize=9.5, colors=INK_MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_facecolor(SURFACE)
    _clean_spines(axis)


def _clean_spines(axis) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
