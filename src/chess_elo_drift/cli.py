"""Command line entry point for the study.

The pipeline is three commands, each resumable and each writing its output where
the next one expects it:

    collect   chess.com API   -> data/raw/games_<era>.jsonl
    evaluate  stored PGNs     -> data/processed/evaluations.csv
    report    evaluations     -> reports/ (tables and figures)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from chess_elo_drift import config
from chess_elo_drift.chesscom import ChessComClient
from chess_elo_drift.collection.sampler import CoverageTarget, StratifiedSnowballCrawler
from chess_elo_drift.collection.seeds import load_seeds_for_era
from chess_elo_drift.collection.store import GameStore, VisitedLog
from chess_elo_drift.analysis.dataset import build_estimation_sample, load_evaluations
from chess_elo_drift.analysis.report import generate_report
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


def games_path(era_name: str):
    return config.DATA_RAW / f"games_{era_name}.jsonl"


def run_collect(args: argparse.Namespace) -> int:
    """Crawl both eras until every rating band meets its quota."""
    logger = logging.getLogger("collect")
    client = ChessComClient(min_interval_seconds=args.min_interval)

    with client:
        for era in config.ERAS:
            seeds = load_seeds_for_era(client, era, config.DATA_RAW)
            store = GameStore(games_path(era.name))
            visited = VisitedLog(config.DATA_RAW / f"visited_{era.name}.json")
            target = CoverageTarget(args.per_cell)
            target.prime(store.read_all())

            crawler = StratifiedSnowballCrawler(
                client,
                store,
                era,
                target,
                rng=random.Random(args.seed),
                max_api_requests=args.max_requests,
                visited=visited,
            )
            logger.info(
                "crawling %s: %d months, target %d per cell, resuming from %d games "
                "and %d visited accounts (coverage %s)",
                era.name, len(era.months), args.per_cell, len(store), len(visited), target.progress,
            )
            try:
                stats = crawler.crawl(seeds)
            finally:
                visited.save()
            logger.info("era %s done: %s", era.name, stats.as_dict())

            stats_path = config.DATA_RAW / f"crawl_stats_{era.name}.json"
            stats_path.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")

    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Re-analyse every collected game with the engine and store move quality."""
    logger = logging.getLogger("evaluate")

    records = [
        record
        for era in config.ERAS
        for record in GameStore(games_path(era.name)).read_all()
    ]
    logger.info("corpus holds %d games", len(records))

    selected = balanced_subsample(records, args.limit)
    if len(selected) < len(records):
        logger.info("selected %d games, balanced across the design", len(selected))
    logger.debug("cell sizes: %s", summarise_cells(selected))

    writer = EvaluationWriter(config.DATA_PROCESSED / "evaluations.csv")
    evaluate_corpus(
        selected,
        writer,
        engine_path=args.engine,
        depth=args.depth,
        workers=args.workers,
    )
    return 0


def run_report(args: argparse.Namespace) -> int:
    """Build the estimation sample, run the comparisons and write the report."""
    logger = logging.getLogger("report")

    evaluations = load_evaluations(args.evaluations)
    sample = build_estimation_sample(evaluations, max_per_player=args.max_per_player)
    artifacts = generate_report(sample, evaluations)

    logger.info("findings: %s", artifacts.findings)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-elo-drift", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)

    collect = subcommands.add_parser("collect", help="sample games from the chess.com API")
    collect.add_argument(
        "--per-cell", type=int, default=60,
        help="target player-observations per (time class, 200-point band) cell",
    )
    collect.add_argument(
        "--max-requests", type=int, default=2500,
        help="API requests allowed per era before giving up on the quotas",
    )
    collect.add_argument(
        "--min-interval", type=float, default=0.0,
        help="minimum seconds between two API calls (raise it if throttled)",
    )
    collect.add_argument("--seed", type=int, default=20182025, help="RNG seed")
    collect.set_defaults(handler=run_collect)

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

    report = subcommands.add_parser("report", help="compare the eras and write the findings")
    report.add_argument(
        "--evaluations", type=Path, default=None, help="engine output CSV to analyse"
    )
    report.add_argument(
        "--max-per-player", type=int, default=3,
        help="cap on rows one player contributes per era and cadence",
    )
    report.set_defaults(handler=run_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
