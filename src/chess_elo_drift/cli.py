"""Command line entry point for the study.

The pipeline is four commands, each resumable and each writing its output where
the next one expects it:

    collect   both sites' APIs -> data/raw/<platform>/games_<year>.jsonl (+ rating snapshots)
    survey    both sites' APIs -> data/raw/conversion/ (current ratings, linked accounts)
    evaluate  stored PGNs      -> data/processed/evaluations.csv
    report    everything above -> reports/ (tables, figures, findings)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from contextlib import ExitStack
from pathlib import Path

from chess_elo_drift import config
from chess_elo_drift.collection.sampler import CoverageTarget, StratifiedSnowballCrawler
from chess_elo_drift.collection.store import GameStore, SnapshotStore, VisitedLog
from chess_elo_drift.engine.results import EvaluationWriter
from chess_elo_drift.engine.runner import evaluate_corpus
from chess_elo_drift.selection import balanced_subsample, summarise_cells


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_years(text: str) -> tuple[int, ...]:
    """Read `2014-2026`, `2018,2026` or a mix of both into a tuple of study years."""
    years: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(bound) for bound in part.split("-"))
            years.update(range(low, high + 1))
        else:
            years.add(int(part))
    unknown = years - set(config.YEARS)
    if unknown:
        raise argparse.ArgumentTypeError(f"outside the study period: {sorted(unknown)}")
    return tuple(sorted(years))


def parse_platforms(text: str) -> tuple[str, ...]:
    if text == "both":
        return config.PLATFORMS
    if text not in config.PLATFORMS:
        raise argparse.ArgumentTypeError(f"unknown platform {text!r}")
    return (text,)


def _open_platform(platform: str, stack: ExitStack, min_interval: float):
    """Return the site's `GameSource` and a seed loader for it."""
    if platform == "chesscom":
        from chess_elo_drift.chesscom import ChessComClient
        from chess_elo_drift.collection.seeds import load_seeds_for_era
        from chess_elo_drift.collection.sources import ChessComSource

        client = stack.enter_context(ChessComClient(min_interval_seconds=min_interval))
        return ChessComSource(client), lambda era: load_seeds_for_era(
            client, era, config.raw_dir(platform)
        )

    from chess_elo_drift.lichess import LichessClient
    from chess_elo_drift.lichess.seeds import load_lichess_seeds
    from chess_elo_drift.lichess.source import LichessSource

    client = stack.enter_context(LichessClient(min_interval_seconds=min_interval))
    # The crawler keeps at most a few games per player and cell, and anonymous
    # exports stream at ~11 games/s, so reading a whole busy month is wasted time.
    return LichessSource(client, max_games_per_month=40), lambda era: load_lichess_seeds(
        client, era, config.raw_dir(platform)
    )


def run_collect(args: argparse.Namespace) -> int:
    """Crawl each requested site and year until every rating band meets its quota."""
    logger = logging.getLogger("collect")

    for platform in args.platform:
        snapshots = SnapshotStore(config.snapshots_path(platform))
        with ExitStack() as stack:
            source, load_seeds = _open_platform(platform, stack, args.min_interval)
            for year in args.years:
                era = config.era_for_year(year)
                time_classes = tuple(
                    tc for tc in config.TIME_CLASSES
                    if config.pool_exists(platform, tc, min(era.months))
                )
                store = GameStore(config.games_path(platform, year))
                visited = VisitedLog(config.visited_path(platform, year))
                target = CoverageTarget(args.per_cell, time_classes=time_classes, months=era.months)
                target.prime(store.read_all())
                if target.is_complete:
                    logger.info("%s %s already complete (%s)", platform, year, target.progress)
                    continue

                crawler = StratifiedSnowballCrawler(
                    source,
                    store,
                    era,
                    target,
                    rng=random.Random(args.seed + year),
                    max_api_requests=args.max_requests,
                    visited=visited,
                    snapshots=snapshots,
                )
                logger.info(
                    "crawling %s %s (%s): target %d per cell, resuming from %d games "
                    "and %d visited accounts (coverage %s)",
                    platform, year, "/".join(time_classes), args.per_cell, len(store),
                    len(visited), target.progress,
                )
                try:
                    stats = crawler.crawl(load_seeds(era))
                finally:
                    visited.save()
                logger.info("%s %s done: %s", platform, year, stats.as_dict())
                config.crawl_stats_path(platform, year).write_text(
                    json.dumps(stats.as_dict(), indent=2), encoding="utf-8"
                )
    return 0


def run_survey(args: argparse.Namespace) -> int:
    """Collect current ratings and cross-site account links for the conversion study."""
    from chess_elo_drift.conversion.survey import collect_survey

    summary = collect_survey(
        platforms=args.platform,
        max_chesscom_requests=args.max_chesscom_requests,
        max_lichess_users=args.max_lichess_users,
        min_interval=args.min_interval,
    )
    logging.getLogger("survey").info("survey done: %s", summary)
    return 0


def load_corpus(platforms: tuple[str, ...] = config.PLATFORMS, years: tuple[int, ...] = config.YEARS):
    return [
        record
        for platform in platforms
        for year in years
        for record in GameStore(config.games_path(platform, year)).read_all()
    ]


def run_evaluate(args: argparse.Namespace) -> int:
    """Re-analyse every collected game with the engine and store move quality."""
    logger = logging.getLogger("evaluate")

    records = load_corpus()
    logger.info("corpus holds %d games", len(records))

    selected = balanced_subsample(records, args.limit)
    if len(selected) < len(records):
        logger.info("selected %d games, balanced across the design", len(selected))
    logger.debug("cell sizes: %s", summarise_cells(selected))

    writer = EvaluationWriter(config.EVALUATIONS_PATH)
    evaluate_corpus(
        selected,
        writer,
        engine_path=args.engine,
        depth=args.depth,
        workers=args.workers,
    )
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Build the estimation samples, run every comparison and write the report."""
    from chess_elo_drift.analysis.report import generate_report

    artifacts = generate_report(
        evaluations_path=args.evaluations or config.EVALUATIONS_PATH,
        max_per_player=args.max_per_player,
    )
    logging.getLogger("report").info("findings: %s", artifacts.findings)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-elo-drift", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)

    collect = subcommands.add_parser("collect", help="sample games from both sites' APIs")
    collect.add_argument(
        "--platform", type=parse_platforms, default=config.PLATFORMS,
        help="chesscom, lichess or both (default both, one after the other)",
    )
    collect.add_argument(
        "--years", type=parse_years, default=config.YEARS,
        help="years to crawl, e.g. 2014-2026 or 2018,2026 (default: all)",
    )
    collect.add_argument(
        "--per-cell", type=int, default=48,
        help="target player-observations per (time class, 200-point band) cell, per year, split evenly across the sampled months",
    )
    collect.add_argument(
        "--max-requests", type=int, default=1500,
        help="API requests allowed per site and year before giving up on the quotas",
    )
    collect.add_argument(
        "--min-interval", type=float, default=0.0,
        help="minimum seconds between two API calls (raise it if throttled)",
    )
    collect.add_argument("--seed", type=int, default=20142026, help="RNG seed")
    collect.set_defaults(handler=run_collect)

    survey = subcommands.add_parser(
        "survey", help="current ratings and linked accounts for the 2026 conversion"
    )
    survey.add_argument("--platform", type=parse_platforms, default=config.PLATFORMS)
    survey.add_argument(
        "--max-chesscom-requests", type=int, default=6000,
        help="cap on chess.com player-stats requests",
    )
    survey.add_argument(
        "--max-lichess-users", type=int, default=60000,
        help="cap on Lichess accounts looked up (300 per request)",
    )
    survey.add_argument("--min-interval", type=float, default=0.0)
    survey.set_defaults(handler=run_survey)

    evaluate = subcommands.add_parser("evaluate", help="score collected games with the engine")
    evaluate.add_argument("--depth", type=int, default=config.ENGINE_DEPTH, help="fixed search depth")
    evaluate.add_argument(
        "--workers", type=int, default=config.DEFAULT_ENGINE_WORKERS,
        help="engine processes to run in parallel",
    )
    evaluate.add_argument(
        "--limit", type=int, default=0,
        help="cap on games to score, spread evenly across the design (0 = all)",
    )
    evaluate.add_argument("--engine", type=Path, default=config.ENGINE_PATH, help="UCI engine binary")
    evaluate.set_defaults(handler=run_evaluate)

    report = subcommands.add_parser("report", help="compare years and sites and write the findings")
    report.add_argument(
        "--evaluations", type=Path, default=None, help="engine output CSV to analyse"
    )
    report.add_argument(
        "--max-per-player", type=int, default=3,
        help="cap on rows one player contributes per site, year and cadence",
    )
    report.set_defaults(handler=run_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
