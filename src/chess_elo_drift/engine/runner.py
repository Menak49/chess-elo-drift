"""Running the analyser across a corpus on every available core.

An engine process is expensive to start and cheap to reuse, so each worker
builds one analyser on start-up and keeps it for the whole run. Results are
written as they arrive: a run that is interrupted loses nothing already scored.
"""

from __future__ import annotations

import atexit
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Sequence

import chess.engine

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
    except chess.engine.EngineTerminatedError:
        # A dead engine would otherwise fail every game this worker is handed
        # for the rest of the run. Start a fresh one; the game is retried by the
        # next run, since nothing was written for it.
        logger.exception("engine died on game %s; restarting it", game_id)
        _ANALYZER.close()
        _ANALYZER.open()
        return None
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
    stall_seconds: float = 600.0,
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
        # Completion order, not submission order. Games differ wildly in cost --
        # a 200-ply middlegame can take a hundred times as long as a short one --
        # and an ordered map would hold every finished result in memory behind
        # the slowest one, so an interrupted run would lose work it had done.
        pending_futures = {pool.submit(_analyse_task, task) for task in tasks}
        while pending_futures:
            done, pending_futures = wait(
                pending_futures, timeout=stall_seconds, return_when=FIRST_COMPLETED
            )
            if not done:
                # No game in `stall_seconds` means a wedged worker, not a slow
                # game (the longest take seconds). Stop rather than hang: what
                # is left stays unscored and the next run picks it up.
                logger.error(
                    "no game finished in %.0fs; stopping with %d games unscored",
                    stall_seconds, len(pending_futures),
                )
                _abandon(pool)
                break
            for future in done:
                evaluation = future.result()
                if evaluation is None:
                    continue
                writer.append(to_rows(by_id[evaluation.game_id], evaluation))
                scored += 1
                if scored % report_every == 0:
                    _log_progress(scored, len(tasks), started)

    logger.info("scored %d/%d games in %.1fs", scored, len(tasks), time.monotonic() - started)
    return scored


def _abandon(pool: ProcessPoolExecutor) -> None:
    """Stop a pool whose workers may never return, without waiting on them."""
    pool.shutdown(wait=False, cancel_futures=True)
    # The executor offers no public way to stop a worker mid-task.
    for process in list(getattr(pool, "_processes", {}).values()):
        process.terminate()


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
