"""Turning engine evaluations into a per-move quality score.

chess.com's own accuracy number cannot be used for this study: it is absent from
essentially every 2018-2019 game of an ordinary player, present on only a small,
self-selected slice of recent ones, and its formula changed in between. Whatever
that number would show, it would be a fact about chess.com's analysis coverage
rather than about how people play.

So accuracy is recomputed here, from the moves, with one fixed engine at one
fixed depth for every game of both eras. The scale is the one popularised by
Lichess -- itself a published reconstruction of the same idea as chess.com's
CAPS -- which maps an evaluation to an expected score and then a loss of
expected score to a 0-100 accuracy. The absolute numbers are therefore not
directly comparable to a chess.com accuracy; the *difference between the two
eras* is what the study measures, and it is measured with a single instrument.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

#: Logistic slope mapping centipawns to an expected score. Fitted on real games
#: by Lichess; reused verbatim so the scale is a documented one.
_WIN_PROBABILITY_SLOPE = 0.00368208

#: Exponential decay from lost expected score to accuracy percentage.
_ACCURACY_SCALE = 103.1668
_ACCURACY_DECAY = 0.04354
_ACCURACY_OFFSET = 3.1669

#: Beyond a rook of advantage, further material tells us nothing about the
#: quality of a move: capping keeps one lost position from dominating a mean.
CENTIPAWN_CAP = 1000

#: Expected-score losses that name a mistake, in percentage points.
INACCURACY_THRESHOLD = 5.0
MISTAKE_THRESHOLD = 10.0
BLUNDER_THRESHOLD = 20.0


def win_percentage(centipawns: float) -> float:
    """Expected score, in percent, for the side the evaluation favours.

    Centipawns are not linear in anything a player cares about: going from +100
    to +200 changes the result far more than going from +900 to +1000. Mapping
    to expected score first is what makes a "loss" comparable across positions.
    """
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-_WIN_PROBABILITY_SLOPE * centipawns)) - 1.0)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Score a single move from the expected score it gave away.

    Both arguments are expected scores for the player who is moving, before and
    after their move. A move that concedes nothing scores 100.
    """
    lost = max(0.0, win_before - win_after)
    accuracy = _ACCURACY_SCALE * math.exp(-_ACCURACY_DECAY * lost) - _ACCURACY_OFFSET
    return min(100.0, max(0.0, accuracy))


def centipawn_loss(before: float, after: float) -> float:
    """Centipawns given away by one move, capped and floored at zero.

    Both arguments are evaluations from the mover's point of view. Negative
    values -- the engine preferring the position *after* the move, which happens
    when a deeper search overturns a shallower one -- are treated as zero loss.
    """
    capped_before = _clamp(before, -CENTIPAWN_CAP, CENTIPAWN_CAP)
    capped_after = _clamp(after, -CENTIPAWN_CAP, CENTIPAWN_CAP)
    return max(0.0, capped_before - capped_after)


def mean_accuracy(accuracies: Sequence[float]) -> float | None:
    """Average move accuracy over a game, or None if nothing was scored.

    A plain mean is used rather than the weighted blend some sites apply,
    because the study compares two groups measured the same way; a weighting
    designed to flatter a player's headline number would only add noise.
    """
    if not accuracies:
        return None
    return sum(accuracies) / len(accuracies)


def count_severe_errors(win_losses: Iterable[float], threshold: float) -> int:
    """How many moves gave away at least `threshold` points of expected score."""
    return sum(1 for loss in win_losses if loss >= threshold)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
