"""Running the analyser across a corpus on every available core.

An engine process is expensive to start and cheap to reuse, so each worker
builds one analyser on start-up and keeps it for the whole run. Results are
written as they arrive: a run that is interrupted loses nothing already scored.
"""

from __future__ import annotations

import atexit
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

from chess_elo_drift import config
from chess_elo_drift.engine.evaluator import GameAnalyzer, GameEvaluation
from chess_elo_drift.engine.results import EvaluationWriter, to_rows
from chess_elo_drift.records import GameRecord

logger = logging.getLogger(__name__)

#: One analyser per worker process, created by the pool initialiser.
_ANALYZER: GameAnalyzer | None = None


def _init_worker(engine_path: str, depth: int, opening_plies: int, max_plies: int) -> None:
    global _ANALYZER
    _ANALYZER = GameAnalyzer(
        Path(engine_path),
        depth=depth,
        opening_plies_skipped=opening_plies,
        max_plies_scored=max_plies,
    ).open()
    atexit.register(_ANALYZER.close)


def _analyse_task(task: tuple[str, str]) -> GameEvaluation | None:
    """Score one game inside a worker. Never raises: a bad game is just skipped."""
    game_id, pgn = task
    if _ANALYZER is None:  # pragma: no cover - the pool always initialises first
        return None
    try:
        return _ANALYZER.analyse(game_id, pgn)
    except Exception:  # noqa: BLE001 - an unreadable game must not kill the run
        logger.exception("failed to analyse game %s", game_id)
        return None


def evaluate_corpus(
    records: Sequence[GameRecord],
    writer: EvaluationWriter,
    *,
    engine_path: Path = config.ENGINE_PATH,
    depth: int = config.ENGINE_DEPTH,
    workers: int = config.DEFAULT_ENGINE_WORKERS,
    opening_plies: int = config.OPENING_PLIES_SKIPPED,
    max_plies: int = config.MAX_PLIES_SCORED_PER_GAME,
    report_every: int = 100,
) -> int:
    """Evaluate every record not already present in the writer's output.

    Returns the number of games newly scored.
    """
    pending = _pending(records, writer.completed_game_ids())
    if not pending:
        logger.info("nothing to evaluate: every game is already scored")
        return 0

    by_id = {record.game_id: record for record in pending}
    tasks = [(record.game_id, record.pgn) for record in pending]
    logger.info("evaluating %d games at depth %d on %d workers", len(tasks), depth, workers)

    scored = 0
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(engine_path), depth, opening_plies, max_plies),
    ) as pool:
        for evaluation in pool.map(_analyse_task, tasks, chunksize=1):
            if evaluation is None:
                continue
            writer.append(to_rows(by_id[evaluation.game_id], evaluation))
            scored += 1
            if scored % report_every == 0:
                _log_progress(scored, len(tasks), started)

    logger.info("scored %d/%d games in %.1fs", scored, len(tasks), time.monotonic() - started)
    return scored


def _pending(records: Iterable[GameRecord], done: set[str]) -> list[GameRecord]:
    seen: set[str] = set()
    pending: list[GameRecord] = []
    for record in records:
        if record.game_id in done or record.game_id in seen:
            continue
        seen.add(record.game_id)
        pending.append(record)
    return pending


def _log_progress(scored: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = scored / elapsed if elapsed else 0.0
    remaining = (total - scored) / rate if rate else 0.0
    logger.info(
        "%d/%d games (%.1f games/s, ~%.0fs left)", scored, total, rate, remaining
    )
