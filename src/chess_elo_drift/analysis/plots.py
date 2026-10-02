"""Figures for the report.

Every chart is a small multiple split by site and cadence -- the four pools are
different rating scales, so they never share a panel -- and every panel of one
figure shares its y axis, so heights compare across panels.

Colour does one job per figure. Where the series are reference ratings (1000,
1500, 2000) they are *ordered*, so they take three steps of one blue ramp, light
to dark, rather than unrelated hues. Where series are categories (the fitted
lines of the conversion, the two cadences) they take the first slots of a
categorical palette validated for colour-vision deficiency on all pairs. Every
series is also named in a legend and labelled at its end, so no chart depends
on colour alone, and every chart has a CSV twin under `tables/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chess_elo_drift import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Categorical slots 1-3; this trio passes the all-pairs CVD check.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

#: Light-to-dark steps of the blue ramp for ordered series. The lightest step is
#: the lightest that still clears 2:1 against the surface.
RATING_RAMP = ("#86b6ef", "#2a78d6", "#104281")

CI_ALPHA = 0.12

#: Height reserved above the panels for title, subtitle and legend, in inches.
HEADER_INCHES = 1.25
NO_LEGEND_HEADER_INCHES = 0.8
DPI = 160


def rating_colours(ratings: Iterable[int]) -> dict[int, str]:
    ordered = sorted(ratings)
    steps = RATING_RAMP if len(ordered) == 3 else tuple(
        matplotlib.colors.to_hex(c)
        for c in matplotlib.colors.LinearSegmentedColormap.from_list("ramp", RATING_RAMP)(np.linspace(0, 1, max(len(ordered), 1)))
    )
    return dict(zip(ordered, steps))


def _panel_title(platform: str, time_class: str) -> str:
    return f"{config.PLATFORM_LABELS.get(platform, platform)} {time_class}"


# -- year by year -----------------------------------------------------------


def plot_accuracy_by_year(levels: pd.DataFrame, output_path: Path) -> Path:
    """Accuracy at fixed ratings, per year, one panel per site and cadence."""
    title = "What a rating bought, year by year"
    if levels.empty:
        return _placeholder(output_path, title, "Not enough evaluated games to fit any year yet.")

    figure, axes = _grid(len(config.PLATFORMS), len(config.TIME_CLASSES), (5.4, 3.6))
    colours = rating_colours(levels["rating"].unique())
    ends: dict = {}
    for row, platform in enumerate(config.PLATFORMS):
        for column, time_class in enumerate(config.TIME_CLASSES):
            axis = axes[row][column]
            panel = levels[(levels["platform"] == platform) & (levels["time_class"] == time_class)]
            _style_axis(axis, _panel_title(platform, time_class))
            if panel.empty:
                _empty_panel(axis)
                continue
            for rating, series in panel.groupby("rating"):
                ends.setdefault(axis, []).append(
                    _line_with_band(axis, series["year"], series["accuracy"], series["ci_low"], series["ci_high"], colours[rating], f"{rating}")
                )
            _year_ticks(axis, panel["year"])
            if platform == "chesscom":
                _mark_break(axis, panel["year"])
    for axis in axes[-1]:
        axis.set_xlabel("year", color=INK_SECONDARY, fontsize=10)
    for row in axes:
        row[0].set_ylabel("predicted accuracy (%)", color=INK_SECONDARY, fontsize=10)
    _legend(figure, [(colours[r], f"rated {r}") for r in sorted(colours)])
    _titles(
        figure,
        title,
        "Engine-recomputed move accuracy of a player at a fixed rating, at a 5+0 blitz or 10+0 rapid clock. "
        "Bands: 95% CI, clustered by player. A gap: no fit that year.",
    )
    return _save(figure, output_path, end_labels=ends)


def plot_trend_robustness(trends: pd.DataFrame, output_path: Path) -> Path:
    """The accuracy trend per year under each specification, per site and cadence."""
    title = "The trend under different specifications"
    accuracy = trends[trends["metric"] == "accuracy"] if not trends.empty else trends
    if accuracy.empty:
        return _placeholder(output_path, title, "No trend could be estimated yet.")

    labels = list(dict.fromkeys(accuracy["label"]))
    figure, axes = _grid(1, 4, (3.4, 0.5 * len(labels) + 1.2), sharey=True, sharex=True, header=NO_LEGEND_HEADER_INCHES)
    cells = [(p, t) for p in config.PLATFORMS for t in config.TIME_CLASSES]
    positions = {label: len(labels) - 1 - i for i, label in enumerate(labels)}
    for axis, (platform, time_class) in zip(axes[0], cells):
        panel = accuracy[(accuracy["platform"] == platform) & (accuracy["time_class"] == time_class)]
        _style_axis(axis, _panel_title(platform, time_class), grid_axis="x")
        axis.axvline(0, color=BASELINE, linewidth=1, zorder=1)
        for row in panel.itertuples():
            y = positions[row.label]
            main = row.specification == "main"
            axis.plot([row.ci_low, row.ci_high], [y, y], color=BLUE if main else INK_MUTED, linewidth=2, solid_capstyle="round", zorder=2)
            axis.plot(row.trend_per_year, y, "o", color=BLUE if main else INK_MUTED, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        axis.set_xlabel("accuracy points per year", color=INK_SECONDARY, fontsize=9.5)
    axes[0][0].set_yticks(list(positions.values()))
    axes[0][0].set_yticklabels(list(positions.keys()), fontsize=9.5, color=INK_SECONDARY)
    axes[0][0].set_ylim(-0.6, len(labels) - 0.4)
    _titles(
        figure,
        title,
        "Linear trend in accuracy at a fixed rating, with 95% CI. Blue: the main model. "
        "Stable time controls exist only for chess.com.",
    )
    return _save(figure, output_path, header=NO_LEGEND_HEADER_INCHES)


# -- sites ------------------------------------------------------------------


def plot_site_offsets(offsets: pd.DataFrame, output_path: Path) -> Path:
    """Lichess minus chess.com rating at equal accuracy, per year, per cadence."""
    title = "How far apart the two sites' ratings sit"
    if offsets.empty:
        return _placeholder(output_path, title, "No year has enough games on both sites yet.")

    figure, axes = _grid(1, len(config.TIME_CLASSES), (5.6, 4.0))
    colours = rating_colours(offsets["rating"].unique())
    ends: dict = {}
    for axis, time_class in zip(axes[0], config.TIME_CLASSES):
        panel = offsets[offsets["time_class"] == time_class]
        _style_axis(axis, time_class)
        axis.axhline(0, color=BASELINE, linewidth=1, zorder=1)
        if panel.empty:
            _empty_panel(axis)
            continue
        banded = sorted(colours)[len(colours) // 2]
        for rating, series in panel.groupby("rating"):
            # Three overlapping intervals turn into one grey mass; the middle
            # rating's band shows the typical uncertainty, the table has the rest.
            low, high = (series["offset_ci_low"], series["offset_ci_high"]) if rating == banded else (np.nan, np.nan)
            ends.setdefault(axis, []).append(
                _line_with_band(axis, series["year"], series["offset"], low, high, colours[rating], f"{rating}")
            )
        _year_ticks(axis, panel["year"])
        axis.set_xlabel("year", color=INK_SECONDARY, fontsize=10)
    axes[0][0].set_ylabel("Lichess rating minus chess.com rating", color=INK_SECONDARY, fontsize=10)
    _legend(figure, [(colours[r], f"chess.com {r}") for r in sorted(colours)])
    _titles(
        figure,
        title,
        "The Lichess rating that plays as accurately as a chess.com rating, minus that rating. "
        f"Band: 95% bootstrap CI for chess.com {sorted(colours)[len(colours) // 2]}. "
        "A gap: no estimate that year.",
    )
    return _save(figure, output_path, end_labels=ends)


# -- conversion -------------------------------------------------------------


def plot_conversions(results, output_path: Path) -> Path:
    """The six pairs of the conversion study: people, fitted lines, identity."""
    from chess_elo_drift.analysis.conversion import FIT_WINDOW, PAIRS

    title = f"{config.CONVERSION_YEAR} rating conversions between the four pools"
    if not results.fits:
        return _placeholder(output_path, title, "No conversion source has enough people rated in two pools yet.")

    figure, axes = _grid(2, 3, (4.2, 4.5), sharey=False)
    for axis, (x_pool, y_pool) in zip([a for row in axes for a in row], PAIRS):
        _style_axis(axis, f"{y_pool.label} vs {x_pool.label}", grid_axis="both", title_size=10.5)
        axis.set_xlabel(x_pool.label, color=INK_SECONDARY, fontsize=9.5)
        axis.set_ylabel(y_pool.label, color=INK_SECONDARY, fontsize=9.5)
        fit = results.fits.get((x_pool.key, y_pool.key))
        if fit is None:
            _empty_panel(axis, "too few people rated in both")
            continue
        low = max(FIT_WINDOW[0], min(fit.x.min(), fit.y.min()) - 50)
        high = min(FIT_WINDOW[1], max(fit.x.max(), fit.y.max()) + 50)
        span = np.array([low, high])
        curve = np.linspace(max(low, fit.x.min()), min(high, fit.x.max()), 200)
        axis.scatter(fit.x, fit.y, s=9, color=INK_MUTED, alpha=min(0.5, 60 / np.sqrt(fit.n) / 4 + 0.1), linewidths=0, zorder=2, rasterized=True)
        if fit.screened_out:
            axis.scatter(
                np.clip(fit.dropped_x, low, high), np.clip(fit.dropped_y, low, high),
                s=22, facecolors="none", edgecolors=INK_SECONDARY, linewidths=1, zorder=2,
            )
        axis.plot(span, span, color=BASELINE, linewidth=1, zorder=1)
        axis.plot(span, fit.ols_y_on_x(span), color=ORANGE, linewidth=2, zorder=3)
        axis.plot(span, fit.line(span), color=AQUA, linewidth=2, zorder=4)
        axis.plot(curve, fit.link(curve), color=BLUE, linewidth=2, zorder=5)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        dropped = f", {fit.screened_out} dropped" if fit.screened_out else ""
        axis.text(
            0.03, 0.97, f"n = {fit.n:,}{dropped}\n{fit.source}",
            transform=axis.transAxes, ha="left", va="top", fontsize=8.5, color=INK_SECONDARY,
        )
    _legend(
        figure,
        [
            (BLUE, "conversion table (equipercentile)"),
            (AQUA, "rule-of-thumb formula"),
            (ORANGE, "average at that rating (OLS)"),
            (BASELINE, "same number on both"),
        ],
    )
    _titles(
        figure,
        title,
        "Each dot is one person with established ratings in both pools; open circles were dropped as outliers. "
        "The OLS line is pulled toward the average.",
    )
    return _save(figure, output_path)


# -- instrument -------------------------------------------------------------


def plot_reported_accuracy_coverage(coverage: pd.DataFrame, output_path: Path) -> Path:
    """How rarely chess.com's own accuracy field exists, year by year."""
    title = "chess.com only publishes an accuracy score for games somebody reviewed"
    if coverage.empty:
        return _placeholder(output_path, title, "No evaluated chess.com games yet.")

    figure, axes = _grid(1, 1, (8.0, 4.2))
    axis = axes[0][0]
    _style_axis(axis, "")
    styles = {"blitz": (BLUE, "o"), "rapid": (AQUA, "s")}
    for time_class, series in coverage.groupby("time_class"):
        colour, marker = styles.get(time_class, (INK_MUTED, "o"))
        series = series.sort_values("year")
        values = series["coverage"].to_numpy(dtype=float) * 100
        axis.plot(series["year"], values, color=colour, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=time_class, zorder=3)
        axis.annotate(
            time_class, (series["year"].iloc[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=10, color=INK_SECONDARY,
        )
    _year_ticks(axis, coverage["year"])
    axis.set_ylim(0, max(10.0, float(coverage["coverage"].max() * 100) * 1.25))
    axis.set_xlabel("year", color=INK_SECONDARY, fontsize=10)
    axis.set_ylabel("share of sides carrying an accuracy (%)", color=INK_SECONDARY, fontsize=10)
    present = [t for t in styles if t in set(coverage["time_class"])]
    _legend(figure, [(styles[t][0], t) for t in present], [styles[t][1] for t in present])
    _titles(figure, title, "Why the study recomputes accuracy with one engine instead of reading it from the API.")
    return _save(figure, output_path)


# -- shared chrome ----------------------------------------------------------


def _grid(
    rows: int,
    columns: int,
    panel_size: tuple[float, float],
    sharey: bool = True,
    sharex: bool = False,
    header: float = HEADER_INCHES,
):
    figure, axes = plt.subplots(
        rows, columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows + header),
        sharey=sharey, sharex=sharex, squeeze=False, facecolor=SURFACE,
    )
    return figure, axes


def _style_axis(axis, title: str, grid_axis: str = "y", title_size: float = 11.5) -> None:
    if title:
        axis.set_title(title, fontsize=title_size, color=INK, pad=8, loc="left")
    axis.set_facecolor(SURFACE)
    axis.tick_params(labelsize=9, colors=INK_SECONDARY, length=0)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BASELINE)


def _line_with_band(axis, x, y, low, high, colour: str, label: str):
    """One series with its band, broken wherever a year has no value.

    The series is laid on every year between its first and last, with NaN for the
    ones it lacks: matplotlib leaves a gap at a NaN instead of joining the
    neighbours, so a run of missing years is not drawn as a trend. A year with a
    value but no neighbours cannot carry a band (it has no width), so its
    interval is drawn as a short vertical bar. Returns where the series ends, for
    `_place_end_labels`.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    low = np.broadcast_to(np.asarray(low, dtype=float), x.shape)
    high = np.broadcast_to(np.asarray(high, dtype=float), x.shape)
    if len(x) == 0:
        return None
    order = np.argsort(x)
    years = np.arange(np.min(x), np.max(x) + 1)
    position = np.searchsorted(years, x[order])
    filled = {}
    for name, values in (("y", y), ("low", low), ("high", high)):
        column = np.full(len(years), np.nan)
        column[position] = values[order]
        filled[name] = column
    y, low, high = filled["y"], filled["low"], filled["high"]
    axis.fill_between(years, low, high, color=colour, alpha=CI_ALPHA, linewidth=0, zorder=1)
    has_value = np.isfinite(y)
    neighbours = np.zeros(len(years), dtype=bool)
    neighbours[1:] |= has_value[:-1]
    neighbours[:-1] |= has_value[1:]
    alone = has_value & ~neighbours & np.isfinite(low) & np.isfinite(high)
    if alone.any():
        axis.vlines(years[alone], low[alone], high[alone], color=colour, alpha=0.35, linewidth=3, zorder=1)
    axis.plot(years, y, color=colour, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    axis.plot(years, y, "o", color=colour, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    if not has_value.any():
        return None
    last = np.flatnonzero(has_value)[-1]
    return years[last], y[last], label


#: Closest two end labels may sit vertically, in points, before the lower one is
#: pushed down and the upper one up; a little over the 8.5 pt type's height.
MIN_LABEL_GAP_POINTS = 11.0


def _spread_labels(positions: list[float], gap: float) -> list[float]:
    """Move positions apart until neighbours are at least `gap` from each other.

    Labels that already clear each other stay put. Labels that collide form a
    block, spaced `gap` apart and centred on where they started, so that each
    stays close to its own line rather than the whole run drifting one way.
    Blocks that end up touching are merged and respaced.
    """
    if not positions:
        return []
    order = np.argsort(positions)
    wanted = np.asarray(positions, dtype=float)[order]
    blocks = [[i] for i in range(len(wanted))]

    def centre(block):
        return float(np.mean(wanted[block]))

    def lowest(block):
        return centre(block) - (len(block) - 1) / 2 * gap

    def highest(block):
        return centre(block) + (len(block) - 1) / 2 * gap

    merged = True
    while merged:
        merged = False
        for index in range(len(blocks) - 1):
            if lowest(blocks[index + 1]) - highest(blocks[index]) < gap - 1e-9:
                blocks[index : index + 2] = [blocks[index] + blocks[index + 1]]
                merged = True
                break
    placed = np.empty(len(wanted))
    for block in blocks:
        for slot, member in enumerate(block):
            placed[member] = lowest(block) + slot * gap
    result = np.empty(len(positions))
    result[order] = placed
    return list(result)


def _place_end_labels(ends: dict) -> None:
    """Label each series at its last point, nudged vertically so labels never overlap.

    Runs after layout, when a panel's height in points is known, because the
    separation that matters is on the page and not in data units.
    """
    for axis, series in ends.items():
        series = [entry for entry in series if entry is not None]
        if not series:
            continue
        low, high = axis.get_ylim()
        height = axis.get_window_extent().height * 72.0 / axis.figure.dpi
        per_unit = height / (high - low) if high > low else 1.0
        natural = [(y - low) * per_unit for _, y, _ in series]
        placed = _spread_labels(natural, MIN_LABEL_GAP_POINTS)
        for (x, y, label), before, after in zip(series, natural, placed):
            axis.annotate(
                label, (x, y), xytext=(7, after - before), textcoords="offset points",
                va="center", fontsize=8.5, color=INK_SECONDARY, annotation_clip=False,
            )


def _year_ticks(axis, years: pd.Series) -> None:
    years = sorted(int(y) for y in pd.unique(years))
    step = 2 if len(years) > 7 else 1
    axis.set_xticks(years[::step])
    axis.set_xlim(years[0] - 0.5, years[-1] + 1.3)


#: chess.com's 10|0 reclassification (2020-09-10), as a position on a year axis
#: where each year's point sits at its integer: past the June sample, inside September's.
CHESSCOM_BREAK_POSITION = 2020.7


def _mark_break(axis, years: pd.Series) -> None:
    """A hairline at chess.com's 10|0 reclassification, if the panel spans it."""
    if not (years.min() < CHESSCOM_BREAK_POSITION < years.max()):
        return
    axis.axvline(CHESSCOM_BREAK_POSITION, color=BASELINE, linewidth=1, zorder=0)
    axis.text(
        CHESSCOM_BREAK_POSITION + 0.12, 0.03, "10|0 becomes rapid",
        transform=axis.get_xaxis_transform(), fontsize=8, color=INK_MUTED, ha="left", va="bottom",
    )


def _empty_panel(axis, message: str = "not enough data") -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", fontsize=10, color=INK_MUTED)


def _legend(figure, entries: list[tuple[str, str]], markers: list[str] | None = None) -> None:
    markers = markers or ["o"] * len(entries)
    handles = [
        plt.Line2D([], [], color=colour, linewidth=2, marker=marker, markersize=6, markeredgecolor=SURFACE)
        for (colour, _), marker in zip(entries, markers)
    ]
    figure.legend(
        handles, [label for _, label in entries], loc="upper left", bbox_to_anchor=(0.006, 1 - 0.86 / figure.get_figheight()),
        ncol=len(entries), frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=1.8,
    )


def _titles(figure, title: str, subtitle: str) -> None:
    height = figure.get_figheight()
    figure.suptitle(title, fontsize=14, color=INK, x=0.012, y=1 - 0.18 / height, ha="left", va="top", weight="medium")
    figure.text(0.012, 1 - 0.52 / height, subtitle, fontsize=9.5, color=INK_SECONDARY, ha="left", va="top", wrap=True)


def _save(figure, output_path: Path, header: float = HEADER_INCHES, end_labels: dict | None = None) -> Path:
    height = figure.get_figheight()
    figure.tight_layout(rect=(0, 0, 1, 1 - header / height), h_pad=1.6)
    if end_labels:
        _place_end_labels(end_labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    return output_path


def _placeholder(output_path: Path, title: str, message: str) -> Path:
    """A figure that says there is nothing to draw.

    A run on a sample too thin to plot is a normal early state of the study, and
    the report should say so on the page rather than fail on the way to writing
    it.
    """
    figure, axis = plt.subplots(figsize=(7.4, 3.2), facecolor=SURFACE)
    axis.axis("off")
    axis.text(0.5, 0.45, message, ha="center", va="center", fontsize=11.5, color=INK_MUTED)
    figure.suptitle(title, fontsize=13.5, color=INK, x=0.02, y=0.94, ha="left", weight="medium")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    return output_path
