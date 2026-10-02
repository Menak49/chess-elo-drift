"""Scoring one game with a UCI engine.

The analyser walks a game once, evaluating every position from a fixed depth,
and reads both players' move quality off that single pass: the evaluation after
one player's move is also the evaluation before their opponent's reply.
"""

from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chess_elo_drift import config
from chess_elo_drift.engine.accuracy import (
    BLUNDER_THRESHOLD,
    INACCURACY_THRESHOLD,
    MISTAKE_THRESHOLD,
    centipawn_loss,
    count_severe_errors,
    mean_accuracy,
    move_accuracy,
    win_percentage,
)

logger = logging.getLogger(__name__)

#: Centipawn value standing in for a forced mate. Large enough that the expected
#: score saturates, finite so the arithmetic stays defined.
MATE_SCORE = 10_000


@dataclass(frozen=True)
class SideEvaluation:
    """How well one player played the scored part of one game."""

    moves_scored: int
    accuracy: float | None
    acpl: float | None
    inaccuracies: int
    mistakes: int
    blunders: int

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GameEvaluation:
    """Both players' scores for a single game, plus how it was measured."""

    game_id: str
    engine_depth: int
    plies_scored: int
    white: SideEvaluation
    black: SideEvaluation

    def side(self, colour: str) -> SideEvaluation:
        return self.white if colour == "white" else self.black


class GameAnalyzer:
    """A single engine process, reused across games.

    Starting an engine costs far more than evaluating a position, so one
    analyser is created per worker and kept alive for the whole run.
    """

    def __init__(
        self,
        engine_path: Path,
        *,
        depth: int = config.ENGINE_DEPTH,
        opening_plies_skipped: int = config.OPENING_PLIES_SKIPPED,
        max_plies_scored: int = config.MAX_PLIES_SCORED_PER_GAME,
        hash_mb: int = 16,
        threads: int = 1,
    ) -> None:
        self._engine_path = Path(engine_path)
        self._depth = depth
        self._opening_plies_skipped = opening_plies_skipped
        self._max_plies_scored = max_plies_scored
        self._hash_mb = hash_mb
        self._threads = threads
        self._engine: chess.engine.SimpleEngine | None = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "GameAnalyzer":
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self._engine_path.resolve()))
            self._engine.configure({"Threads": self._threads, "Hash": self._hash_mb})
        return self

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except chess.engine.EngineError:
                pass
            self._engine = None

    def __enter__(self) -> "GameAnalyzer":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- analysis -----------------------------------------------------------

    def analyse(self, game_id: str, pgn: str) -> GameEvaluation | None:
        """Score a game, or return None if its moves cannot be read."""
        if self._engine is None:
            self.open()

        moves = _read_mainline(pgn)
        if moves is None:
            return None

        first_ply = self._opening_plies_skipped
        last_ply = min(len(moves), first_ply + self._max_plies_scored)
        if last_ply <= first_ply:
            return None

        evaluations, turns = self._evaluate_positions(moves, first_ply, last_ply)
        return self._score(game_id, evaluations, turns, first_ply, last_ply)

    def _evaluate_positions(
        self, moves: list[chess.Move], first_ply: int, last_ply: int
    ) -> tuple[dict[int, float], dict[int, bool]]:
        """Evaluate every position bounding a scored ply, from White's side.

        Terminal positions are left out: an engine has nothing to say about a
        finished game, so the mating move itself goes unscored. The same rule
        applies to every game of every year and site, so comparisons are unaffected.
        """
        board = chess.Board()
        evaluations: dict[int, float] = {}
        turns: dict[int, bool] = {}

        for index in range(last_ply + 1):
            if index >= first_ply and not board.is_game_over():
                evaluations[index] = self._evaluate(board)
                turns[index] = board.turn
            if index < len(moves):
                board.push(moves[index])

        return evaluations, turns

    def _evaluate(self, board: chess.Board) -> float:
        assert self._engine is not None
        info = self._engine.analyse(board, chess.engine.Limit(depth=self._depth))
        score = info["score"].white()
        return float(score.score(mate_score=MATE_SCORE))

    def _score(
        self,
        game_id: str,
        evaluations: dict[int, float],
        turns: dict[int, bool],
        first_ply: int,
        last_ply: int,
    ) -> GameEvaluation:
        accuracies: dict[bool, list[float]] = {chess.WHITE: [], chess.BLACK: []}
        losses: dict[bool, list[float]] = {chess.WHITE: [], chess.BLACK: []}
        cp_losses: dict[bool, list[float]] = {chess.WHITE: [], chess.BLACK: []}

        for ply in range(first_ply, last_ply):
            if ply not in evaluations or ply + 1 not in evaluations:
                continue
            mover = turns[ply]
            before_white, after_white = evaluations[ply], evaluations[ply + 1]

            if mover == chess.WHITE:
                before, after = before_white, after_white
            else:
                before, after = -before_white, -after_white

            lost_expected_score = max(
                0.0, win_percentage(before) - win_percentage(after)
            )
            accuracies[mover].append(move_accuracy(win_percentage(before), win_percentage(after)))
            losses[mover].append(lost_expected_score)
            cp_losses[mover].append(centipawn_loss(before, after))

        return GameEvaluation(
            game_id=game_id,
            engine_depth=self._depth,
            plies_scored=len(accuracies[chess.WHITE]) + len(accuracies[chess.BLACK]),
            white=_summarise(accuracies[chess.WHITE], losses[chess.WHITE], cp_losses[chess.WHITE]),
            black=_summarise(accuracies[chess.BLACK], losses[chess.BLACK], cp_losses[chess.BLACK]),
        )


def _summarise(
    accuracies: list[float], losses: list[float], cp_losses: list[float]
) -> SideEvaluation:
    return SideEvaluation(
        moves_scored=len(accuracies),
        accuracy=mean_accuracy(accuracies),
        acpl=(sum(cp_losses) / len(cp_losses)) if cp_losses else None,
        inaccuracies=count_severe_errors(losses, INACCURACY_THRESHOLD),
        mistakes=count_severe_errors(losses, MISTAKE_THRESHOLD),
        blunders=count_severe_errors(losses, BLUNDER_THRESHOLD),
    )


def _read_mainline(pgn: str) -> list[chess.Move] | None:
    """Parse a PGN into its mainline moves, or None if it is unreadable.

    A game that starts from a set-up position is refused too: its moves are only
    legal from that position, and replaying them from the normal start sends the
    engine an illegal move sequence, which kills the engine process.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except (ValueError, RuntimeError):
        return None
    if game is None or game.board() != chess.Board():
        return None
    moves = list(game.mainline_moves())
    return moves or None
