"""Assembling the tables, figures and written findings.

Every number in the output is computed here from the data; nothing is typed in
by hand, so re-running the pipeline on a larger corpus rewrites the conclusions
rather than leaving a stale claim behind. Every input may be missing or thin --
the survey might still be running, a year might have no games yet -- and each
section then says so in place instead of the run failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chess_elo_drift import config
from chess_elo_drift.analysis import conversion, plots, sites, yearly
from chess_elo_drift.analysis.dataset import (
    MAX_OBSERVATIONS_PER_PLAYER,
    REFERENCE_CLOCK_SECONDS,
    apply_inclusion_rules,
    coverage_of_reported_accuracy,
    load_evaluations,
    sample_sizes,
)
from chess_elo_drift.analysis.statistics import describe_band_by_year, validate_against_reported
from chess_elo_drift.analysis.surveys import (
    MAX_ESTABLISHED_RD,
    MIN_POOL_GAMES,
    POOLS,
    RECENT_DAYS,
    SNAPSHOT_MIN_GAMES,
    load_conversion_inputs,
)

logger = logging.getLogger(__name__)

SEED = 20142026


@dataclass(frozen=True)
class ReportArtifacts:
    """Everything a run produced, so the caller can point at it."""

    findings: Path
    tables: dict[str, Path]
    figures: dict[str, Path]


@dataclass
class Results:
    """Every table the narrative draws on, computed once."""

    sample: pd.DataFrame
    attrition: pd.DataFrame
    sizes: pd.DataFrame
    effects: pd.DataFrame
    levels: pd.DataFrame
    trends: pd.DataFrame
    slopes: pd.DataFrame
    bands: pd.DataFrame
    offsets: pd.DataFrame
    conversions: conversion.ConversionResults
    formulas: pd.DataFrame
    conversion_table: pd.DataFrame
    chains: pd.DataFrame
    agreement: pd.DataFrame
    accuracy_check: pd.DataFrame
    benchmark: pd.DataFrame
    sources: pd.DataFrame
    coverage: pd.DataFrame
    validation: dict[str, float]
    max_per_player: int = MAX_OBSERVATIONS_PER_PLAYER
    site_replicates: int = sites.DEFAULT_BOOTSTRAP_REPLICATES


def generate_report(
    evaluations_path: Path | None = None,
    max_per_player: int = MAX_OBSERVATIONS_PER_PLAYER,
    *,
    data_raw: Path | None = None,
    output_root: Path | None = None,
    site_replicates: int = sites.DEFAULT_BOOTSTRAP_REPLICATES,
    conversion_replicates: int = conversion.DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> ReportArtifacts:
    """Run every comparison and write tables, figures and the findings note.

    `data_raw` is where the rating snapshots (`<platform>/rating_snapshots.jsonl`)
    and the conversion survey (`conversion/*.jsonl`) are read from; it and
    `output_root` default to the project's directories and exist so a run can
    be pointed elsewhere.
    """
    output_root = output_root or config.REPORTS
    tables_dir, figures_dir = output_root / "tables", output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    evaluations = load_evaluations(evaluations_path or config.EVALUATIONS_PATH)
    results = compute(
        evaluations,
        load_conversion_inputs(data_raw),
        max_per_player=max_per_player,
        site_replicates=site_replicates,
        conversion_replicates=conversion_replicates,
        seed=seed,
    )

    tables = {name: _write_table(frame, tables_dir / f"{name}.csv") for name, frame in _table_frames(results).items()}
    figures = {
        "accuracy_by_year": plots.plot_accuracy_by_year(results.levels, figures_dir / "accuracy_by_year.png"),
        "trend_robustness": plots.plot_trend_robustness(results.trends, figures_dir / "trend_robustness.png"),
        "site_offsets": plots.plot_site_offsets(sites.offsets_at(results.offsets), figures_dir / "site_offsets.png"),
        "conversion": plots.plot_conversions(results.conversions, figures_dir / f"conversion_{config.CONVERSION_YEAR}.png"),
        "reported_accuracy_coverage": plots.plot_reported_accuracy_coverage(
            results.coverage, figures_dir / "reported_accuracy_coverage.png"
        ),
    }

    findings_path = output_root / "findings.md"
    findings_path.write_text(render_findings(results, tables, figures), encoding="utf-8")
    logger.info("report written to %s", findings_path)
    return ReportArtifacts(findings=findings_path, tables=tables, figures=figures)


def compute(
    evaluations: pd.DataFrame,
    inputs,
    *,
    max_per_player: int = MAX_OBSERVATIONS_PER_PLAYER,
    site_replicates: int = sites.DEFAULT_BOOTSTRAP_REPLICATES,
    conversion_replicates: int = conversion.DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> Results:
    sample, attrition = apply_inclusion_rules(evaluations, max_per_player=max_per_player)
    offsets = sites.site_offsets(sample, replicates=site_replicates, seed=seed)
    conversions = conversion.conversion_study(inputs, replicates=conversion_replicates, seed=seed)
    return Results(
        sample=sample,
        attrition=attrition,
        sizes=sample_sizes(sample),
        effects=yearly.year_effects(sample),
        levels=yearly.accuracy_at_ratings(sample),
        trends=yearly.trends(sample),
        slopes=yearly.slope_drift(sample),
        bands=describe_band_by_year(sample),
        offsets=offsets,
        conversions=conversions,
        formulas=conversion.formulas_table(conversions),
        conversion_table=conversion.conversion_table(conversions),
        chains=conversion.chain_table(conversions),
        agreement=conversion.source_agreement(conversions),
        accuracy_check=conversion.accuracy_cross_check(conversions, offsets),
        benchmark=conversion.benchmark_comparison(conversions),
        sources=_sources_frame(conversions),
        coverage=coverage_of_reported_accuracy(evaluations),
        validation=validate_against_reported(sample[sample["platform"] == "chesscom"] if not sample.empty else sample),
        max_per_player=max_per_player,
        site_replicates=site_replicates,
    )


def _table_frames(results: Results) -> dict[str, pd.DataFrame]:
    return {
        "sample_attrition": results.attrition,
        "sample_sizes": results.sizes,
        "year_effects": results.effects,
        "accuracy_at_ratings": results.levels,
        "year_trends": results.trends,
        "slope_drift": results.slopes,
        "band_by_year": results.bands,
        "site_offsets": results.offsets,
        "conversion_formulas": results.formulas,
        "conversion_table": results.conversion_table,
        "conversion_chains": results.chains,
        "conversion_source_agreement": results.agreement,
        "conversion_accuracy_check": results.accuracy_check,
        "conversion_benchmark": results.benchmark,
        "conversion_sources": results.sources,
        "reported_accuracy_coverage": results.coverage,
    }


def _established_text(kept: dict[str, int]) -> str:
    labels = {pool.key: pool.label for pool in POOLS}
    parts = [f"{labels[key]} {count:,}" for key, count in kept.items() if key in labels]
    both = kept.get("all of these")
    if both is not None and len(parts) > 1:
        parts.append(f"{'both' if len(parts) == 2 else 'all four'} {both:,}")
    return ", ".join(parts)


def _sources_frame(conversions: conversion.ConversionResults) -> pd.DataFrame:
    rows = []
    for log in conversions.logs:
        row: dict[str, object] = {"source": log.name, "file_present": log.available, "records": log.records}
        row.update({f"established_{key}".replace(" ", "_"): value for key, value in log.kept.items()})
        row["established"] = _established_text(log.kept)
        row["notes"] = "; ".join(log.notes)
        rows.append(row)
    return pd.DataFrame(rows)


# -- narrative --------------------------------------------------------------


def render_findings(results: Results, tables: dict[str, Path], figures: dict[str, Path]) -> str:
    sections = [
        _title_block(results),
        "## The answers\n",
        _headline_years(results),
        _headline_sites(results),
        _headline_conversion(results),
        _method_section(results),
        _years_section(results, figures),
        _sites_section(results, figures),
        _conversion_section(results, figures),
        _robustness_section(results, figures),
        _bands_section(results),
        _instrument_section(results, figures),
        _caveats_section(),
        _files_section(tables, figures),
    ]
    return "\n".join(section.rstrip() + "\n" for section in sections)


def _title_block(results: Results) -> str:
    sample = results.sample
    first, last = config.YEARS[0], config.YEARS[-1]
    lines = [f"# What does a chess rating buy? chess.com and Lichess, {first}-{last}\n"]
    if sample.empty:
        lines.append("_No evaluated game has passed the inclusion rules yet, so the year-by-year and site comparisons below are empty._\n")
    else:
        lines.append(
            f"Built from {len(sample):,} player-observations ({sample['game_id'].nunique():,} games, "
            f"{sample['player_id'].nunique():,} players) re-analysed with one engine, plus the "
            f"{config.CONVERSION_YEAR} rating surveys. Every number here is computed by "
            "`analysis/report.py`; re-running the report rewrites them.\n"
        )
    return "\n".join(lines)


def _headline_years(results: Results) -> str:
    lines = ["### 1. Year by year: does a given rating buy the same play as it used to?\n"]
    main = results.trends[results.trends["specification"] == "main"] if not results.trends.empty else results.trends
    if main.empty:
        lines.append("_Not enough years with evaluated games to estimate any trend yet._")
        return "\n".join(lines) + "\n"
    for row in main.itertuples():
        label = _pool_label(row.platform, row.time_class)
        single = (
            f"{yearly.verdict(pd.Series(row._asdict()))} ({row.trend_per_year:+.3f} accuracy points per year, "
            f"95% CI {row.ci_low:+.3f} to {row.ci_high:+.3f})"
        )
        segments = _break_segments(results.trends, row.platform, row.time_class)
        if segments is None:
            text = f"- **{label}** ({row.first_year}-{row.last_year}): {single}."
        else:
            text = f"- **{label}** ({row.first_year}-{row.last_year}): {_break_text(segments, results.effects, row, single)}"
        first = _first_year_effect(results.effects, row.platform, row.time_class)
        if first is not None:
            text += (
                f" A {yearly.REFERENCE_RATINGS[1]} in {first.year} played {abs(first.effect):.2f} points "
                f"{'more' if first.effect > 0 else 'less'} accurately than a {yearly.REFERENCE_RATINGS[1]} in "
                f"{first.reference_year} (95% CI {first.ci_low:+.2f} to {first.ci_high:+.2f})"
            )
            if np.isfinite(first.rating_points_equivalent) and first.p_value < 0.05:
                text += (
                    f", worth very roughly {_rating(first.rating_points_equivalent, signed=True)} rating points "
                    "(an order of magnitude only)"
                )
            text += "."
        if segments is None:
            text += _segment_text(results.trends, row.platform, row.time_class)
        lines.append(text)
    lines.append(
        "\nNegative means ratings have become easier to reach: the same number buys less accurate play now. "
        "Every comparison holds the rating and the clock fixed; see *Year by year* for each year."
    )
    return "\n".join(lines) + "\n"


def _segment_rows(trends: pd.DataFrame, platform: str, time_class: str) -> dict[str, pd.Series]:
    """The before and after trend rows of a pool, for whichever of them exist."""
    found = {}
    for spec in (yearly.BEFORE_BREAK, yearly.AFTER_BREAK):
        row = trends[
            (trends["platform"] == platform)
            & (trends["time_class"] == time_class)
            & (trends["specification"] == spec.name)
        ]
        if not row.empty:
            found[spec.name] = row.iloc[0]
    return found


def _break_segments(trends: pd.DataFrame, platform: str, time_class: str) -> dict[str, pd.Series] | None:
    """Both sides of the 2020 break, or None when the pool has no break or one side is missing."""
    found = _segment_rows(trends, platform, time_class)
    return found if len(found) == 2 else None


def _break_text(segments: dict[str, pd.Series], effects: pd.DataFrame, main, single_trend: str) -> str:
    """A verdict for a pool that crosses a documented break: each side, the step, then the single line.

    One straight line across the reclassification mixes the drift with the step
    that the move itself caused, so it can hide both. The two sides and the step
    come first and the line across everything is given for comparison.
    """
    before, after = segments[yearly.BEFORE_BREAK.name], segments[yearly.AFTER_BREAK.name]
    parts = [f"chess.com reclassified 10|0 as rapid on {yearly.CHESSCOM_RECLASSIFICATION}, so each side is read on its own."]
    for when, row in (("Before", before), ("After", after)):
        parts.append(
            f"{when} ({int(row['first_year'])}-{int(row['last_year'])}): {yearly.verdict(row)} "
            f"({row['trend_per_year']:+.3f} accuracy points per year, 95% CI {row['ci_low']:+.3f} to {row['ci_high']:+.3f})."
        )
    step = _break_step(effects, main.platform, main.time_class)
    if step is not None:
        parts.append(step)
    parts.append(f"One line across the whole span, for comparison: {single_trend}.")
    return " ".join(parts)


def _break_step(effects: pd.DataFrame, platform: str, time_class: str) -> str | None:
    """How far the level moved between the last year before the break and the first after it.

    Both year effects are measured against the same reference year, so their
    difference is the change from one to the other. Their errors are combined as
    if independent; they share the reference year and are in fact positively
    correlated, so the interval is a little wider than it needs to be.
    """
    cell = effects[(effects["platform"] == platform) & (effects["time_class"] == time_class)].set_index("year")
    last_before, first_after = yearly.LAST_YEAR_BEFORE_BREAK, yearly.FIRST_YEAR_AFTER_BREAK
    if last_before not in cell.index or first_after not in cell.index:
        return None
    before, after = cell.loc[last_before], cell.loc[first_after]
    step = after["effect"] - before["effect"]
    error = float(np.hypot(before["std_error"], after["std_error"]))
    if not (np.isfinite(step) and np.isfinite(error)):
        return None
    text = (
        f"Across the break, a given rating played {abs(step):.2f} points {'more' if step > 0 else 'less'} accurately "
        f"in {first_after} than in {last_before} (about 95% CI {step - 1.96 * error:+.2f} to {step + 1.96 * error:+.2f})"
    )
    points = after["rating_points_equivalent"] - before["rating_points_equivalent"]
    if np.isfinite(points) and abs(step) > 1.96 * error:
        text += f", worth very roughly {_rating(points, signed=True)} rating points"
    return text + "."


def _segment_text(trends: pd.DataFrame, platform: str, time_class: str) -> str:
    """When only one side of the 2020 break could be fitted: what there is, and why it is partial."""
    parts = []
    for spec, when in ((yearly.BEFORE_BREAK, "before"), (yearly.AFTER_BREAK, "after")):
        r = _segment_rows(trends, platform, time_class).get(spec.name)
        if r is not None:
            interval = _signed_ci(r["trend_per_year"], r["ci_low"], r["ci_high"], 3)
            parts.append(f"{when}, {interval} per year ({r['first_year']}-{r['last_year']})")
    if not parts:
        return ""
    return (
        " The line crosses chess.com's September 2020 reclassification of 10|0 as rapid; fitted on each side "
        f"of it (2020 left out) the trend is {'; '.join(parts)}."
    )


def _first_year_effect(effects: pd.DataFrame, platform: str, time_class: str):
    cell = effects[(effects["platform"] == platform) & (effects["time_class"] == time_class)]
    if cell.empty:
        return None
    row = cell.sort_values("year").iloc[0]
    return None if row["year"] == row["reference_year"] else row


def _headline_sites(results: Results) -> str:
    lines = ["### 2. chess.com against Lichess\n"]
    at = sites.offsets_at(results.offsets)
    if at.empty:
        lines.append("_No year has enough evaluated games on both sites in the same cadence yet._")
        return "\n".join(lines) + "\n"
    year = config.CONVERSION_YEAR if config.CONVERSION_YEAR in set(at["year"]) else int(at["year"].max())
    latest = at[at["year"] == year]
    lines.append(
        f"At equal move accuracy in {year}, the Lichess rating sits this far from the chess.com one "
        "(Lichess minus chess.com, 95% bootstrap CI):\n"
    )
    table = latest.assign(
        cadence=latest["time_class"],
        chesscom=latest["rating"],
        offset_text=[_signed_ci(r.offset, r.offset_ci_low, r.offset_ci_high, 0) for r in latest.itertuples()],
    ).pivot(index="cadence", columns="chesscom", values="offset_text")
    table.columns = [f"chess.com {c}" for c in table.columns]
    lines.append(_markdown(table.reset_index()))
    lines.append("")
    lines += _offset_spread_lines(at)
    lines.append(_equating_caveat(results, at))
    lines.append(
        "\nThis is *accuracy equating*: the engine judges both sites' moves the same way, so the rating on each "
        "site that produces the same accuracy is matched. It is an independent estimate, not a head count of "
        "players who play on both sites -- that is part 3."
    )
    return "\n".join(lines) + "\n"


def _offset_spread_lines(at: pd.DataFrame) -> list[str]:
    """How far the headline rating's offset ranged over the years, counting only offsets with an interval.

    An offset without an interval is one the bootstrap could not reproduce, in
    practice an extrapolation past the ratings sampled on the other site, and its
    point value can be absurd (+1000). Letting it set the range would report
    the most unreliable number as the extreme.
    """
    rating = yearly.REFERENCE_RATINGS[1]
    lines = []
    for time_class, cell in at[at["rating"] == rating].groupby("time_class"):
        cell = cell.dropna(subset=["offset"])
        has_interval = cell["offset_ci_low"].notna() & cell["offset_ci_high"].notna()
        rated, left_out = cell[has_interval], int((~has_interval).sum())
        note = ""
        if left_out:
            note = (
                f" {left_out} more {'year has' if left_out == 1 else 'years have'} an offset with no interval "
                "(an extrapolation, or unstable across resamples) and "
                f"{'is' if left_out == 1 else 'are'} left out."
            )
        if len(rated) >= 2:
            low, high = rated.loc[rated["offset"].idxmin()], rated.loc[rated["offset"].idxmax()]
            lines.append(
                f"- {time_class}: across {len(rated)} years with an interval the offset at {rating} ranged from "
                f"{_rating(low['offset'], signed=True)} ({int(low['year'])}) to {_rating(high['offset'], signed=True)} ({int(high['year'])})."
                + note
            )
        elif left_out:
            lines.append(f"- {time_class}: fewer than two years have an offset with an interval at {rating}.{note}")
    return lines


def _year_to_year_swing(at: pd.DataFrame) -> tuple[float, float] | None:
    """Median and largest change in the headline rating's offset between consecutive years."""
    changes = []
    for _, cell in at[at["rating"] == yearly.REFERENCE_RATINGS[1]].groupby("time_class"):
        cell = cell.dropna(subset=["offset"]).sort_values("year")
        years, offsets = cell["year"].to_numpy(), cell["offset"].to_numpy()
        changes += [abs(b - a) for (y0, a), (y1, b) in zip(zip(years, offsets), zip(years[1:], offsets[1:])) if y1 - y0 == 1]
    return (float(np.median(changes)), float(np.max(changes))) if changes else None


def _rating_points_per_accuracy_point(effects: pd.DataFrame) -> tuple[float, float] | None:
    """The fewest and most rating points one accuracy point is worth, over the fitted pools."""
    if effects.empty:
        return None
    slopes = effects.groupby(["platform", "time_class"])["slope_per_100"].first()
    slopes = slopes[np.isfinite(slopes) & (slopes > 0.05)]
    if slopes.empty:
        return None
    points = 100.0 / slopes
    return float(points.min()), float(points.max())


def _equating_caveat(results: Results, at: pd.DataFrame) -> str:
    """Say next to the headline figures how far accuracy equating can be trusted, from the data.

    Equating reads a rating off a shallow curve (a point of accuracy spans
    hundreds of rating points), so sampling noise in accuracy is magnified into
    the offset. The linked-account conversion does not go through accuracy at
    all, which is why it is the one to convert with.
    """
    gaps = results.accuracy_check["difference"].dropna() if not results.accuracy_check.empty else pd.Series(dtype=float)
    sentences = []
    disagrees = len(gaps) > 0 and float(gaps.abs().max()) > conversion.AGREEMENT_TOLERANCE
    if len(gaps) and disagrees:
        sentences.append(
            "These offsets do not agree with the conversion fitted on people who hold accounts on both sites (part 3): in the "
            f"accuracy cross-check, equating minus linked accounts has a median of {_rating(float(gaps.median()), signed=True)} "
            f"rating points (from {_rating(float(gaps.min()), signed=True)} to {_rating(float(gaps.max()), signed=True)} "
            f"over {len(gaps)} comparisons)."
        )
    elif len(gaps):
        sentences.append(
            "These offsets agree with the conversion fitted on people who hold accounts on both sites (part 3) to within "
            f"{_rating(float(gaps.abs().max()))} rating points in the accuracy cross-check."
        )
    swing = _year_to_year_swing(at)
    if swing is not None:
        sentences.append(
            f"At chess.com {yearly.REFERENCE_RATINGS[1]} the offset moves by a median of {_rating(swing[0])} points "
            f"between consecutive years (largest {_rating(swing[1])})."
        )
    worth = _rating_points_per_accuracy_point(results.effects)
    if worth is not None:
        span = _rating(worth[0]) if _rating(worth[0]) == _rating(worth[1]) else f"{_rating(worth[0])}-{_rating(worth[1])}"
        sentences.append(
            f"The cause is a shallow accuracy-against-rating curve: one accuracy point is worth only about {span} "
            "rating points, so ordinary sampling noise in accuracy becomes a large rating error."
        )
    if not sentences:
        return ""
    advice = (
        "To convert a rating between the sites, use the linked-account conversion in part 3; read the offsets above "
        "as rough evidence of which way and how far the sites differ, not as a conversion."
        if len(gaps)
        else "Read the offsets above as rough evidence of which way and how far the sites differ, not as a conversion."
    )
    return "\n**Caution.** " + " ".join(sentences + [advice])


def _headline_conversion(results: Results) -> str:
    year = config.CONVERSION_YEAR
    lines = [f"### 3. {year} conversion formulas between the four pools\n"]
    formulas = results.formulas
    if formulas.empty:
        lines.append(
            "_No conversion could be fitted: no source has enough people with established ratings in two pools. "
            "See *Conversion details* for what each source contained._"
        )
        return "\n".join(lines) + "\n"
    lines.append(
        "A rating in one pool is matched to the rating in another that sits at the *same percentile* among the "
        "people with established ratings in both (equipercentile linking). The match is symmetric, so converting "
        "there and back returns the starting rating, and it follows the relation where it bends.\n"
    )
    lines.append(
        "**Rules of thumb.** Each formula is the straight line that matches the two pools' means and spreads (linear "
        "equating, the straight-line version of the same idea). Use it only inside its valid range; the last column "
        "says how far it strays from the table there.\n"
    )
    shown = pd.DataFrame(
        {
            "formula (rounded)": formulas["formula"],
            "valid for": [f"{_rating(lo)}–{_rating(hi)}" for lo, hi in zip(formulas["valid_from"], formulas["valid_to"])],
            "people": [f"{n:,}" for n in formulas["people"]],
            "source": formulas["source"],
            "off the table by up to": [f"{_rating(gap)}" for gap in formulas["formula_vs_table_max_gap"]],
        }
    )
    lines.append(_markdown(shown))
    curved = formulas[~formulas["straight_line_adequate"].astype(bool)].iloc[::2]
    for _, row in curved.iterrows():
        lines.append(
            f"\n_{row['from']} and {row['to']}: the relation bends, so the straight formula misses by up to "
            f"{_rating(row['formula_vs_table_max_gap'])} points inside its range; the table below is the better guide._"
        )
    for note in results.conversions.notes:
        lines.append(f"\n_{note}_")

    lines.append(
        "\n**Conversion table** (equipercentile). Each row is one rating in the first column's pool; 95% bootstrap "
        "interval in brackets."
    )
    lines.append(
        "`*` marks a rating outside the range where 95% of that pair's people sit (an extrapolation); `–` means the "
        f"converted value would fall outside {conversion.FIT_WINDOW[0]}-{conversion.FIT_WINDOW[1]}, where nothing was fitted.\n"
    )
    for pool in POOLS:
        block = _conversion_block(results.conversion_table, pool.label)
        if block is not None:
            lines.append(f"From **{pool.label}**:\n")
            lines.append(block)
            lines.append("")
    spreads = formulas.groupby("to")["ols_residual_sd"].median() if "ols_residual_sd" in formulas else pd.Series(dtype=float)
    if not spreads.empty:
        lines.append(
            f"These convert a *level*. An individual's actual rating in the other pool scatters around it: "
            f"the typical 95% range is ±{_rating(1.96 * spreads.median())} points, because people are genuinely "
            "better at one cadence or site than another. The per-pair spread is in `conversion_table.csv` "
            "(`typical_low`/`typical_high`)."
        )
    return "\n".join(lines) + "\n"


def _conversion_block(table: pd.DataFrame, source_label: str) -> str | None:
    rows = table[table["from"] == source_label]
    if rows.empty:
        return None
    rows = rows.assign(text=[_conversion_cell(r) for r in rows.itertuples()])
    wide = rows.pivot(index="rating", columns="to", values="text")
    order = [pool.label for pool in POOLS if pool.label in wide.columns]
    wide = wide[order].reset_index().rename(columns={"rating": source_label})
    return _markdown(wide)


def _conversion_cell(row) -> str:
    """One converted rating with its interval; nothing where the line has left the data entirely."""
    low, high = conversion.FIT_WINDOW
    if not low <= row.equivalent <= high:
        return "–"
    return f"{_rating(row.equivalent)} ({_range(row.ci_low, row.ci_high)}){'*' if row.extrapolated else ''}"


# -- sections ---------------------------------------------------------------


#: A site-year whose largest month holds more than this share of its rows is
#: named in the method section: an even split would be a third.
SEASON_IMBALANCE_SHARE = 0.6


def _season_balance_text(sample: pd.DataFrame) -> str:
    """Name the site-years the crawl still drew mostly from one month."""
    if sample.empty or "month" not in sample:
        return ""
    counts = sample.groupby(["platform", "year", "month"]).size()
    share = (counts.groupby(["platform", "year"]).max() / counts.groupby(["platform", "year"]).sum()).round(2)
    lopsided = share[share > SEASON_IMBALANCE_SHARE]
    if lopsided.empty:
        return f"No site-year draws more than {SEASON_IMBALANCE_SHARE:.0%} of its rows from one month."
    named = "; ".join(
        f"{config.PLATFORM_LABELS[platform]} {year} ({value:.0%})" for (platform, year), value in lopsided.items()
    )
    return (
        f"The quotas were not always met: these site-years still draw more than {SEASON_IMBALANCE_SHARE:.0%} of "
        f"their rows from one month, so their year effects lean on the month control, or are confounded with the "
        f"season where a pool has too few mixed years to estimate it: {named}."
    )


def _method_section(results: Results) -> str:
    months = ", ".join(pd.Timestamp(2000, m, 1).strftime("%B") for m in config.SAMPLED_MONTHS)
    lines = [
        "## How it was measured\n",
        f"- Rated blitz and rapid games from chess.com and Lichess, {config.YEARS[0]}-{config.YEARS[-1]}, "
        f"crawled from the same calendar months every year ({months}) with an equal quota per month, and every "
        f"model controls for the calendar month as well. {_season_balance_text(results.sample)}",
        "- A player's rating is the one printed on that game. Each site's blitz and rapid are separate pools and are "
        "never pooled; nor are the two sites.",
        f"- Every game is re-scored with Stockfish at fixed depth {config.ENGINE_DEPTH}, skipping the first "
        f"{config.OPENING_PLIES_SKIPPED} plies of book moves; accuracy is derived from that one pass.",
        f"- Ratings {config.RATING_MIN}-{config.RATING_MAX}; each side scored on at least "
        f"{config.MIN_MOVES_SCORED_PER_SIDE} moves; at most {results.max_per_player} rows per player, site, year and "
        "cadence. Provisional Lichess ratings are dropped. "
        "Standard errors are clustered by player (site + username) throughout.",
        f"- Clock: every model controls for log(base + 40 × increment), and predictions are made at "
        f"{_clock_text()}. chess.com moved 10|0 from blitz to rapid on {yearly.CHESSCOM_RECLASSIFICATION} "
        "([announcement](https://www.chess.com/news/view/10-minute-chess-now-rapid-rated-bullet-ratings-increased)) "
        "and each cadence's mix of controls keeps changing, so without this a year with longer games would look "
        "stronger. Before that date chess.com rapid was essentially 15-minute-plus games, so its 10+0 prediction "
        "for early years leans on the clock term.",
        "\nWhat each inclusion rule removed:\n",
        _markdown(results.attrition),
    ]
    if not results.sizes.empty:
        sizes = results.sizes.assign(cell=[_pool_label(p, t) for p, t in zip(results.sizes["platform"], results.sizes["time_class"])])
        wide = sizes.pivot(index="year", columns="cell", values="observations").fillna(0).astype(int)
        lines += ["\nObservations per year after the rules (Lichess rapid starts in 2018, when the pool was created):\n", _markdown(wide.reset_index())]
    return "\n".join(lines) + "\n"


def _clock_text() -> str:
    return " and ".join(f"{seconds // 60}+0 {tc}" for tc, seconds in REFERENCE_CLOCK_SECONDS.items())


def _years_section(results: Results, figures: dict[str, Path]) -> str:
    lines = [
        "## Year by year\n",
        "One model per site and cadence: `accuracy ~ C(year) + rating + rating² + log(clock)`, clustered by player. "
        "The year coefficients compare each year with the reference year at the *same* rating and clock; the trend "
        "replaces them with a straight line in the year.\n",
        f"![accuracy by year]({_relative(figures['accuracy_by_year'])})\n",
        "Two breaks sit inside these series. On chess.com, 10|0 became rapid on "
        f"{yearly.CHESSCOM_RECLASSIFICATION} (the vertical line): rapid's share of play went from under a tenth to "
        "about a third, and players who moved over had their rapid rating set from their blitz rating. The "
        "chess.com trend is therefore also fitted on each side of it (*Robustness*). On Lichess, rapid began on "
        "2017-12-01 with every player's rating copied from their Classical one, so 2018 rapid ratings still carry "
        "Classical's calibration.\n",
    ]
    main = results.trends[results.trends["specification"] == "main"] if not results.trends.empty else results.trends
    if main.empty:
        lines.append("_Not enough data to fit any year model yet._")
        return "\n".join(lines) + "\n"

    trend_view = pd.DataFrame(
        {
            "pool": [_pool_label(p, t) for p, t in zip(main["platform"], main["time_class"])],
            "years": [f"{a}-{b} ({n})" for a, b, n in zip(main["first_year"], main["last_year"], main["years"])],
            "trend per year": [_signed_ci(r.trend_per_year, r.ci_low, r.ci_high, 3) for r in main.itertuples()],
            "p": [_p(p) for p in main["p_value"]],
            "slope per 100 rating": [f"{v:.2f}" for v in main["slope_per_100"]],
            "≈ rating points per year": [_rating(v, signed=True) for v in main["rating_points_per_year"]],
            "players": main["players"],
        }
    )
    lines += ["**Trend** (accuracy points per year at a fixed rating):\n", _markdown(trend_view)]
    lines.append(
        "\nThe last-but-one column divides the trend by the accuracy-per-rating slope. That slope is a few tenths of "
        "a point per 100 rating, so it sits in the denominator of a ratio far less certain than the trend itself: "
        "read it as an order of magnitude, never as a figure."
    )

    if not results.effects.empty:
        effects = results.effects.assign(
            cell=[_pool_label(p, t) for p, t in zip(results.effects["platform"], results.effects["time_class"])],
            text=[
                "reference" if r.year == r.reference_year else _signed_ci(r.effect, r.ci_low, r.ci_high)
                for r in results.effects.itertuples()
            ],
        )
        wide = effects.pivot(index="year", columns="cell", values="text").fillna("")
        lines += [
            "\n**Each year against the reference year** (accuracy points at the same rating and clock; positive = "
            "that year's players were more accurate than today's at the same rating):\n",
            _markdown(wide.reset_index()),
        ]

    if not results.slopes.empty:
        lines.append(
            "\n**Does the rating slope itself change?** If it does, the year gap depends on the rating it is read at. "
            "The drift column is the change in the accuracy-per-100-rating slope per year; the joint test asks whether "
            "one slope fits every year.\n"
        )
        slopes = results.slopes
        view = pd.DataFrame(
            {
                "pool": [_pool_label(p, t) for p, t in zip(slopes["platform"], slopes["time_class"])],
                "slope first year": [f"{s:.2f} ({y})" for s, y in zip(slopes["slope_first_year"], slopes["first_year"])],
                "slope last year": [f"{s:.2f} ({y})" for s, y in zip(slopes["slope_last_year"], slopes["last_year"])],
                "drift per year": [_signed_ci(r.slope_change_per_year, r.ci_low, r.ci_high, 3) for r in slopes.itertuples()],
                "one slope fits all years (p)": [_p(p) for p in slopes["equal_slopes_p_value"]],
            }
        )
        lines.append(_markdown(view))
        drifting = slopes[slopes["equal_slopes_p_value"] < 0.05]
        if drifting.empty:
            lines.append(
                "\nNo pool shows a detectable change of slope, so the year gap can be read as the same at every rating."
            )
        else:
            names = ", ".join(_pool_label(p, t) for p, t in zip(drifting["platform"], drifting["time_class"]))
            lines.append(
                f"\nThe slope changes over the years in {names}: there, the year gap differs by rating level, and "
                "the figure above (which lets every year have its own slope) is the better guide than a single number."
            )
    return "\n".join(lines) + "\n"


def _sites_section(results: Results, figures: dict[str, Path]) -> str:
    lines = [
        "## chess.com against Lichess\n",
        "For each year and cadence with at least "
        f"{sites.MIN_SITE_OBSERVATIONS} rows on both sites, each site's `accuracy ~ rating + rating² + log(clock)` "
        "curve is fitted separately, and a chess.com rating is matched to the Lichess rating with the same "
        f"predicted accuracy at the same clock. Intervals: 95% bootstrap over players ({results.site_replicates} "
        "replicates). A blank means the match falls outside the ratings sampled on the other site; an offset with no "
        f"interval means more than {sites.MAX_FAILED_SHARE:.0%} of replicates found no match.\n",
        f"![site offsets]({_relative(figures['site_offsets'])})\n",
    ]
    at = sites.offsets_at(results.offsets)
    if at.empty:
        lines.append("_No year has enough games on both sites yet._")
        return "\n".join(lines) + "\n"
    at = at.assign(
        column=[f"{t} {r}" for t, r in zip(at["time_class"], at["rating"])],
        text=[_signed_ci(r.offset, r.offset_ci_low, r.offset_ci_high, 0) for r in at.itertuples()],
    )
    wide = at.pivot(index="year", columns="column", values="text").fillna("")
    lines += ["Lichess minus chess.com rating at equal accuracy, by chess.com rating:\n", _markdown(wide.reset_index())]
    lines.append(_skipped_years_text(sites.skipped_cells(results.sample)))
    lines.append(
        "\nEqual accuracy is not equal standing in the pool: the claim is only that the two groups make moves of the "
        "same engine-judged quality at the same clock. The reverse direction (Lichess to chess.com) is in "
        "`site_offsets.csv`."
    )
    return "\n".join(lines) + "\n"


def _skipped_years_text(skipped: pd.DataFrame) -> str:
    """Name the years left out of the table for being too thin, with what each site had."""
    if skipped.empty:
        return ""
    by_cadence = []
    for time_class, cell in skipped.groupby("time_class"):
        years = ", ".join(
            f"{int(r.year)} (chess.com {int(r.chesscom)}, Lichess {int(r.lichess)})" for r in cell.sort_values("year").itertuples()
        )
        by_cadence.append(f"{time_class}: {years}")
    return (
        f"\nSkipped because one site had fewer than {sites.MIN_SITE_OBSERVATIONS} usable rows (rows on each site in "
        f"brackets): {'; '.join(by_cadence)}. Years with no games at all are not listed."
    )


def _conversion_section(results: Results, figures: dict[str, Path]) -> str:
    year = config.CONVERSION_YEAR
    lines = [
        f"## {year} conversion: details\n",
        f"![conversions]({_relative(figures['conversion'])})\n",
        "**Who counts.** A rating enters only when it is established: at least "
        f"{MIN_POOL_GAMES} games in the pool, a rating deviation of at most {MAX_ESTABLISHED_RD} (Lichess's own "
        f"provisional threshold, applied to both Glicko sites), last played within {RECENT_DAYS} days of the "
        f"survey on chess.com or seen in {year} on Lichess, and a Lichess account neither closed nor flagged. "
        f"Same-month snapshots need {SNAPSHOT_MIN_GAMES}+ games of each cadence that month. Pairs outside "
        f"{conversion.FIT_WINDOW[0]}-{conversion.FIT_WINDOW[1]} are left out of the fits (rating floors and titled "
        "outliers bend the line there).\n",
        "**Sources:**\n",
        _markdown(_source_view(results.sources)),
        "\n**Why this method.** Regressing one rating on the other answers a different question -- the *average* "
        "rating in the other pool of people at a given rating -- and is pulled toward the mean by everything that "
        "makes a person's two ratings disagree; the two regressions do not invert, so converting A to B and back "
        "does not return to A. That answer is still shown (\"OLS slope\", with its residual spread) as the "
        "typical-player prediction. The published conversion instead matches *percentiles* among the people who "
        "hold both ratings: the table matches them all (equipercentile linking, presmoothed, straight beyond the "
        "central 95%), the formula matches only the mean and spread (linear equating). Both are exact inverses of "
        "their reverse. A Deming fit with the RDs as error variances was the alternative; it assumes every "
        "difference between a person's two ratings is rating noise, but the residual spread below is several "
        "times the RDs -- most of it is genuine preference for a cadence or site -- so its slope is shown only as a "
        "sensitivity check.\n",
        f"**Outliers.** Before fitting, pairs further than {conversion.OUTLIER_ROBUST_SDS:.0f} robust standard "
        "deviations from a median/MAD line are dropped (mostly wrong or stale links: ratings 800+ points apart "
        "where nearly everyone is within 200). The count per pair is in the table below; the dropped points are "
        "drawn as open circles in the figure.\n",
    ]
    if results.formulas.empty:
        lines.append("_No pair had enough people to fit._")
        return "\n".join(lines) + "\n"

    forward = results.formulas.iloc[::2]
    fit_view = pd.DataFrame(
        {
            "pair": [f"{f} → {t}" for f, t in zip(forward["from"], forward["to"])],
            "source": forward["source"],
            "people": forward["people"],
            "outliers dropped": forward["outliers_dropped"],
            "r": forward["correlation"].round(3),
            "formula slope (95% CI)": [
                f"{s:.3f} ({lo:.3f} to {hi:.3f})" for s, lo, hi in zip(forward["slope"], forward["slope_ci_low"], forward["slope_ci_high"])
            ],
            "Deming slope": forward["deming_slope"].round(3),
            "OLS slope": forward["ols_slope"].round(3),
            "residual SD": forward["ols_residual_sd"].round(0).astype(int),
            "straight line OK": [
                f"{'yes' if ok else 'no'} (max bend {_rating(dev)})"
                for ok, dev in zip(forward["straight_line_adequate"], forward["curvature_max_deviation"])
            ],
        }
    )
    lines += ["**The fits:**\n", _markdown(fit_view)]
    bent = forward[~forward["straight_line_adequate"].astype(bool)]
    if bent.empty:
        lines.append(
            f"\nA squared term never moves the fit by more than {conversion.CURVATURE_TOLERANCE:.0f} points over "
            f"{conversion.TABLE_RATINGS[0]}-{conversion.TABLE_RATINGS[-1]} (or is not detectable), so a straight line is adequate."
        )
    else:
        names = "; ".join(f"{f} → {t}" for f, t in zip(bent["from"], bent["to"]))
        lines.append(
            f"\nThe relation bends noticeably for: {names}. There the straight formula is a compromise and the "
            "equipercentile table, which follows the bend, is the conversion to use."
        )

    lines.append(_chain_text(results.chains))
    lines.append(_agreement_text(results.agreement))
    lines.append(_accuracy_check_text(results.accuracy_check))
    lines.append(_benchmark_text(results.benchmark))
    return "\n".join(lines) + "\n"


def _benchmark_text(benchmark: pd.DataFrame) -> str:
    """The outside comparison, kept visibly apart from this study's numbers."""
    view = benchmark.assign(
        pair=[f"{f} → {t}" for f, t in zip(benchmark["from"], benchmark["to"])],
        this_study=[
            f"{_rating(v)} ({_range(lo, hi)})" if np.isfinite(v) else "n/a"
            for v, lo, hi in zip(
                benchmark["this_study"], benchmark["this_study_ci_low"], benchmark["this_study_ci_high"]
            )
        ],
        formula=[_rating(v) for v in benchmark["this_study_formula"]],
        typical=[_rating(v) for v in benchmark["this_study_typical"]],
        difference=[_rating(v, signed=True) for v in benchmark["difference"]],
    )
    view = view.rename(
        columns={
            "benchmark": "ChessGoals",
            "this_study": "this study (table)",
            "formula": "this study (formula)",
            "typical": "this study (typical)",
            "difference": "table minus ChessGoals",
        }
    )
    return (
        "\n**Published comparison (not from this study's data).** "
        f"{conversion.BENCHMARK_SOURCE}: a voluntary survey with usernames matched across sites and Lichess RD "
        "below 150; its values are typed in from the site, not recomputed. \"Typical\" is this study's OLS "
        "average for people at that rating, the closer analogue if the benchmark averages respondents.\n\n"
        + _markdown(
            view[["pair", "rating", "ChessGoals", "this study (table)", "this study (formula)", "this study (typical)", "table minus ChessGoals"]]
        )
    )


def _source_view(sources: pd.DataFrame) -> pd.DataFrame:
    if sources.empty:
        return sources
    view = sources.copy()
    view["file_present"] = view["file_present"].map({True: "yes", False: "**missing**"})
    view = view.rename(columns={"file_present": "file", "established": "established ratings"})
    return view[["source", "file", "records", "established ratings", "notes"]]


def _chain_text(chains: pd.DataFrame) -> str:
    if chains.empty:
        return "\n**Consistency.** _Not enough pairs fitted to chain conversions through a third pool._"
    fitted = {(x.label, y.label) for x, y in conversion.PAIRS}
    is_fitted_pair = pd.Series([(f, t) in fitted for f, t in zip(chains["from"], chains["to"])], index=chains.index)
    middle = chains[(chains["rating"] == conversion.CHECK_RATINGS[1]) & is_fitted_pair]
    worst = chains.loc[chains["discrepancy"].abs().idxmax()]
    view = middle.assign(discrepancy=middle["discrepancy"].round(0).astype(int), direct=middle["direct"].round(0).astype(int), chained=middle["chained"].round(0).astype(int))
    return (
        "\n**Consistency.** Converting through a third pool (for example chess.com rapid → Lichess rapid → Lichess "
        "blitz) should land near the direct formula. At "
        f"{conversion.CHECK_RATINGS[1]}, the routes compare as follows; over "
        f"{conversion.CHECK_RATINGS[0]}-{conversion.CHECK_RATINGS[-1]} the largest gap is "
        f"{_rating(worst['discrepancy'], signed=True)} points ({worst['from']} → {worst['to']} via {worst['via']}, at "
        f"{int(worst['rating'])}). Gaps are expected: each pair is fitted on the people rated in *those two* pools. "
        "Each pair is shown once; the reverse direction is in `conversion_chains.csv`.\n\n"
        + _markdown(view[["from", "to", "via", "direct", "chained", "discrepancy"]].reset_index(drop=True))
    )


def _agreement_text(agreement: pd.DataFrame) -> str:
    if agreement.empty:
        return "\n**Second source.** _No pair had a second source large enough to cross-check the published fit._"
    view = agreement.assign(
        pair=[f"{f} → {t}" for f, t in zip(agreement["from"], agreement["to"])],
        published=agreement["published"].round(0).astype(int),
        check=agreement["check"].round(0).astype(int),
        difference=agreement["difference"].round(0).astype(int),
    )
    worst = float(agreement["difference"].abs().max())
    verdict = "agree" if worst <= conversion.AGREEMENT_TOLERANCE else "disagree in places"
    return (
        f"\n**Second source.** The same pair fitted on a different set of people; the sources {verdict} "
        f"(largest gap {worst:.0f} points).\n\n"
        + _markdown(
            view[["pair", "rating", "published_source", "published", "check_source", "check", "check_people", "difference"]].rename(
                columns={
                    "published_source": "published from", "check_source": "check from",
                    "check_people": "check people", "difference": "check minus published",
                }
            )
        )
    )


def _accuracy_check_text(check: pd.DataFrame) -> str:
    if check.empty or check[["linked_accounts", "accuracy_equating"]].isna().all().all():
        return "\n**Accuracy cross-check.** _Not available: needs both linked accounts and engine data for the conversion year._"
    view = check.assign(
        linked_accounts=[_rating(v) for v in check["linked_accounts"]],
        accuracy_equating=[
            f"{_rating(v)} ({_range(lo, hi)})" if np.isfinite(v) else "n/a"
            for v, lo, hi in zip(check["accuracy_equating"], check["accuracy_ci_low"], check["accuracy_ci_high"])
        ],
        difference=[_rating(v, signed=True) for v in check["difference"]],
    )
    return (
        f"\n**Accuracy cross-check.** The same-cadence cross-site conversion from linked accounts, against the Lichess "
        f"rating that plays as accurately in {config.CONVERSION_YEAR} (part 2). The two use disjoint evidence -- "
        "people who play on both sites, versus how well each site's players move -- so agreement is meaningful and "
        "disagreement is a warning about one of them.\n\n"
        + _markdown(
            view.rename(
                columns={
                    "time_class": "cadence", "chesscom_rating": "chess.com rating",
                    "linked_accounts": "Lichess, linked accounts", "accuracy_equating": "Lichess, accuracy equating",
                    "difference": "equating minus linked",
                }
            )[["cadence", "chess.com rating", "Lichess, linked accounts", "Lichess, accuracy equating", "equating minus linked"]]
        )
    )


def _robustness_section(results: Results, figures: dict[str, Path]) -> str:
    lines = [
        "## Robustness\n",
        "The trend (accuracy points per year at a fixed rating) under alternative choices. Without the clock "
        "control, a shift in the time-control mix is read as a change in skill; the stable-controls row keeps only "
        "chess.com time controls whose class never changed; one game per player removes any weight from prolific "
        "accounts; ACPL is a second metric that does not depend on the accuracy formula (lower is better, so its "
        "sign is reversed).\n",
        f"![trend robustness]({_relative(figures['trend_robustness'])})\n",
    ]
    if results.trends.empty:
        lines.append("_No trend could be estimated yet._")
        return "\n".join(lines) + "\n"
    trends = results.trends.assign(
        pool=[_pool_label(p, t) for p, t in zip(results.trends["platform"], results.trends["time_class"])],
        text=[_signed_ci(r.trend_per_year, r.ci_low, r.ci_high, 3) for r in results.trends.itertuples()],
    )
    wide = trends.pivot(index="label", columns="pool", values="text").fillna("n/a")
    wide = wide.reindex([s.label for s in yearly.SPECIFICATIONS if s.label in wide.index])
    lines.append(_markdown(wide.reset_index().rename(columns={"label": "specification"})))
    return "\n".join(lines) + "\n"


def _bands_section(results: Results) -> str:
    lines = [
        "## Shape check: band by year\n",
        f"Mean accuracy per {config.REPORT_BAND_WIDTH}-point band and year, with no model at all. A shape check on "
        "the regressions, not a set of findings: cells are small and players sit at different points within a band.\n",
    ]
    if results.bands.empty:
        lines.append("_No rows yet._")
        return "\n".join(lines) + "\n"
    for (platform, time_class), cell in results.bands.groupby(["platform", "time_class"]):
        wide = cell.pivot(index="band_label", columns="year", values="mean")
        wide = wide.loc[sorted(wide.index, key=lambda label: int(label.split("-")[0]))]
        lines += [f"**{_pool_label(platform, time_class)}**\n", _markdown(wide.reset_index().rename(columns={"band_label": "band"})), ""]
    return "\n".join(lines) + "\n"


def _instrument_section(results: Results, figures: dict[str, Path]) -> str:
    lines = [
        "## Why accuracy was recomputed\n",
        "chess.com's own accuracy exists only for games somebody ran a review on, its formula has changed over the "
        "years, and Lichess publishes none through the same archive. Comparing years or sites on it would compare "
        "the games people chose to review, across a moving formula. Every game is therefore re-scored with one "
        "engine at one depth.\n",
        f"![coverage]({_relative(figures['reported_accuracy_coverage'])})\n",
    ]
    if not results.coverage.empty:
        coverage = results.coverage.assign(text=[f"{c:.0%} of {n}" for c, n in zip(results.coverage["coverage"], results.coverage["observations"])])
        wide = coverage.pivot(index="year", columns="time_class", values="text").fillna("")
        lines += ["Share of chess.com sides carrying a reported accuracy:\n", _markdown(wide.reset_index())]
    validation = results.validation
    if "pearson_r" in validation:
        lines.append(
            f"\nWhere both numbers exist (n = {validation['n']:.0f}), the recomputed accuracy tracks chess.com's "
            f"(Pearson r = {validation['pearson_r']:.2f}, Spearman rho = {validation['spearman_rho']:.2f}): the "
            f"instrument measures the same thing on a different scale (mean {validation['mean_recomputed']:.1f} here "
            f"against {validation['mean_reported']:.1f} reported)."
        )
    else:
        lines.append(f"\n_Too few games carry both numbers (n = {validation.get('n', 0):.0f}) to validate the recomputed accuracy against chess.com's._")
    return "\n".join(lines) + "\n"


def _caveats_section() -> str:
    return "\n".join(
        [
            "## What this cannot tell you\n",
            "- **Survivorship, worse the further back.** Old games can only be read from accounts that still exist. "
            "Players who quit and closed their account are invisible, and a 2014 sample is more filtered than a "
            "2025 one.",
            "- **Activity weighting.** The crawler walks the opponent graph, so a player with a hundred games that "
            "month is likelier to be sampled than one with three. The sample describes the opposition you would "
            "actually meet at a rating, not a uniform draw over accounts.",
            "- **Clustered sampling.** Each account's month is read newest first, so games bunch towards the end of "
            "the month and around a few very active accounts and their opponents. Errors are clustered by player, "
            "which covers repeated players but not whole communities, so a single year can move by a point or two "
            "for no reason; read the trend, not one year.",
            "- **Accuracy is not strength.** It rewards quiet, forcing and simplified positions. If the *style* of "
            "play at a rating changed -- more theory, sharper openings, more time trouble -- accuracy moves without "
            "strength moving.",
            f"- **One engine, one depth.** Stockfish at depth {config.ENGINE_DEPTH} makes the scale identical across "
            "years and sites, which is what the comparison needs; the absolute numbers are not chess.com accuracies "
            "and should not be quoted as such.",
            f"- **Cadence boundaries moved.** chess.com reclassified 10|0 from blitz to rapid on "
            f"{yearly.CHESSCOM_RECLASSIFICATION}, seeding movers' rapid ratings from blitz. The clock control, the "
            "stable-controls check and the before/after trends address the time controls, but the *players* who "
            "followed 10|0 into rapid changed that pool's composition, which no control removes.",
            "- **Lichess rapid only exists from 2018,** and began "
            "([2017-12-01](https://lichess.org/blog/Wh9KWiQAAI5JrKVn/introducing-rapid-ratings)) as a copy of each "
            "player's Classical rating. Before it, the Lichess API labels 10+0 games \"rapid\" by today's speed "
            "rules while their rating is a Classical one; those rows are excluded, not relabelled. Lichess also "
            "estimated game length as base + 30 × increment before 2015-04-24 (40 × since), so a few early games "
            "sit in a different cadence than they were rated in.",
            "- **Rating systems changed.** Lichess's rating floor fell from 800 to 600 (2019-06-28) and to 400 "
            "(2023-03); its new accounts start with a larger RD since September 2020; and a first-move advantage "
            "entered its Glicko-2 in November 2025 ([changelog](https://lichess.org/page/changelog-2025)). "
            "chess.com lets new accounts pick a starting rating. A year effect is the combined result of pool "
            "composition and such administrative changes, which this design cannot separate; the "
            "ratings-1000-and-up check keeps clear of the floors.",
            "- **Linked accounts are self-selected.** Only people who declare their chess.com account on Lichess enter "
            "the cross-site conversion, and they are likely more serious and more active than average.",
            "- **The rating survey is a cross-section, not a panel.** It pairs each person's current ratings; it says "
            "how the pools relate in the survey year, not how any person's ratings evolved.",
            "- **Equal accuracy is not equal rating.** The site comparison equates move quality, not results against "
            "a common opponent; the conversion formulas equate people, not moves. Where the two agree, both are "
            "more credible.",
        ]
    )


def _files_section(tables: dict[str, Path], figures: dict[str, Path]) -> str:
    """Every artefact, relative to this note (which sits beside `tables/` and `figures/`)."""
    lines = ["## Files\n", "Tables (CSV, full precision):\n"]
    lines += [f"- `{_relative(path)}`" for path in tables.values()]
    lines += ["\nFigures:\n"]
    lines += [f"- `{_relative(path)}`" for path in figures.values()]
    return "\n".join(lines)


# -- formatting -------------------------------------------------------------


def _pool_label(platform: str, time_class: str) -> str:
    return f"{config.PLATFORM_LABELS.get(platform, platform)} {time_class}"


def _signed_ci(value: float, low: float, high: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return ""
    if round(value, digits) == 0:
        value = 0.0
    if digits == 0:
        text = _rating(value, signed=True)
        return f"{text} ({_rating(low, signed=True)} to {_rating(high, signed=True)})" if np.isfinite(low) and np.isfinite(high) else text
    text = f"{value:+.{digits}f}"
    if np.isfinite(low) and np.isfinite(high):
        text += f" ({low:+.{digits}f} to {high:+.{digits}f})"
    return text


def _rating(value: float, signed: bool = False) -> str:
    """Ratings to the nearest 5 points: finer than that is noise dressed as precision."""
    if value is None or not np.isfinite(value):
        return "n/a"
    rounded = int(5 * round(float(value) / 5))
    return f"{rounded:+d}" if signed else f"{rounded:d}"


def _range(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "n/a"
    return f"{_rating(low)}–{_rating(high)}"


def _p(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.2g}"


def _relative(path: Path) -> str:
    """Figures are linked relative to findings.md, which sits beside `figures/`."""
    return f"{path.parent.name}/{path.name}"


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    numeric = frame.select_dtypes(include="number").columns
    frame.assign(**{column: frame[column].round(4) for column in numeric}).to_csv(path, index=False)
    return path


def _markdown(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_(no rows)_"
    return frame.to_markdown(index=False)
