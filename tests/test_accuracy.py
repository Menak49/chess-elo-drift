import math

import pytest

from chess_elo_drift.engine.accuracy import (
    BLUNDER_THRESHOLD,
    CENTIPAWN_CAP,
    centipawn_loss,
    count_severe_errors,
    mean_accuracy,
    move_accuracy,
    win_percentage,
)


def test_a_balanced_position_is_an_even_game():
    assert win_percentage(0) == pytest.approx(50.0)


def test_win_percentage_is_antisymmetric():
    for centipawns in (50, 200, 700, 3000):
        assert win_percentage(centipawns) + win_percentage(-centipawns) == pytest.approx(100.0)


def test_win_percentage_is_monotonic_and_bounded():
    values = [win_percentage(cp) for cp in range(-2000, 2001, 100)]
    assert values == sorted(values)
    assert 0.0 <= values[0] and values[-1] <= 100.0


def test_win_percentage_saturates_on_a_forced_mate():
    assert win_percentage(10_000) == pytest.approx(100.0, abs=1e-6)


def test_a_move_that_concedes_nothing_scores_full_marks():
    """The published constants top out a hair under 100, by design."""
    assert move_accuracy(62.0, 62.0) == pytest.approx(100.0, abs=1e-3)


def test_improving_on_the_engine_is_not_rewarded_beyond_full_marks():
    """A deeper search can favour the played move; that is not a 110% move."""
    assert move_accuracy(40.0, 55.0) == move_accuracy(40.0, 40.0) <= 100.0


def test_accuracy_falls_as_expected_score_is_given_away():
    scores = [move_accuracy(80.0, 80.0 - lost) for lost in (0, 5, 10, 20, 40)]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] < 20.0


def test_accuracy_never_leaves_the_percentage_scale():
    assert move_accuracy(100.0, 0.0) >= 0.0
    assert move_accuracy(100.0, 0.0) <= 100.0


def test_centipawn_loss_is_the_evaluation_given_away():
    assert centipawn_loss(120, -30) == pytest.approx(150)


def test_centipawn_loss_is_capped_so_one_move_cannot_dominate():
    assert centipawn_loss(9000, -9000) == pytest.approx(2 * CENTIPAWN_CAP)


def test_centipawn_loss_is_never_negative():
    assert centipawn_loss(-50, 120) == 0.0


def test_mean_accuracy_of_nothing_is_undefined():
    assert mean_accuracy([]) is None


def test_mean_accuracy_averages_the_moves():
    assert mean_accuracy([90.0, 80.0, 100.0]) == pytest.approx(90.0)


def test_severe_errors_are_counted_at_the_threshold():
    losses = [0.0, 4.9, 5.0, 19.9, 20.0, 55.0]
    assert count_severe_errors(losses, BLUNDER_THRESHOLD) == 2


def test_formula_matches_its_published_constants():
    """Guards the scale itself: a silent constant change would shift every result."""
    lost = 12.5
    expected = 103.1668 * math.exp(-0.04354 * lost) - 3.1669
    assert move_accuracy(70.0, 70.0 - lost) == pytest.approx(expected)
