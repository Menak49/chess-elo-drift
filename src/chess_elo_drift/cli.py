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

from chess_elo_drift import config
from chess_elo_drift.chesscom import ChessComClient
from chess_elo_drift.collection.sampler import CoverageTarget, StratifiedSnowballCrawler
from chess_elo_drift.collection.seeds import load_seeds_for_era
from chess_elo_drift.collection.store import GameStore


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
            target = CoverageTarget(args.per_cell)
            crawler = StratifiedSnowballCrawler(
                client,
                store,
                era,
                target,
                rng=random.Random(args.seed),
                max_api_requests=args.max_requests,
            )
            logger.info(
                "crawling era %s (%d months, target %d games per cell, store holds %d)",
                era.name, len(era.months), args.per_cell, len(store),
            )
            stats = crawler.crawl(seeds)
            logger.info("era %s done: %s", era.name, stats.as_dict())

            stats_path = config.DATA_RAW / f"crawl_stats_{era.name}.json"
            stats_path.write_text(json.dumps(stats.as_dict(), indent=2), encoding="utf-8")

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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
