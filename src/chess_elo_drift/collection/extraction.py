"""Turning raw chess.com archive payloads into validated `GameRecord`s.

This module owns one question only: *is this payload a usable game, and what
does it say?* Which games the study wants is decided by the sampler.
"""

from __future__ import annotations

from typing import Any

from chess_elo_drift import config
from chess_elo_drift.config import YearMonth
from chess_elo_drift.records import GameRecord, classify_result, count_plies

STANDARD_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def extract_game(raw: dict[str, Any], *, month: YearMonth) -> GameRecord | None:
    """Build a `GameRecord`, or return `None` if the payload is not usable.

    Rejected: variants, unrated games, time classes outside the study, and
    games too short to say anything about playing strength (aborts, instant
    resignations, disconnections).

    The time class is taken as chess.com recorded it when the game was played,
    which is the pool the printed rating belongs to -- even for a 10|0 game from
    before September 2020, when chess.com still rated those as blitz.
    """
    if raw.get("rules") != "chess" or not raw.get("rated"):
        return None
    time_class = raw.get("time_class")
    if time_class not in config.TIME_CLASSES:
        return None
    if not config.pool_exists("chesscom", time_class, month):
        return None

    white, black = raw.get("white"), raw.get("black")
    if not isinstance(white, dict) or not isinstance(black, dict):
        return None
    if not (white.get("rating") and black.get("rating")):
        return None

    # Thematic tournaments start from a set position (a gambit already played);
    # those moves were not chosen by the players and the game is not comparable.
    if raw.get("initial_setup", STANDARD_START) != STANDARD_START:
        return None

    pgn = raw.get("pgn")
    if not pgn:
        return None
    ply_count = count_plies(pgn)
    if ply_count < config.MIN_PLIES:
        return None

    accuracies = raw.get("accuracies") or {}
    return GameRecord(
        game_id=raw.get("uuid") or raw["url"],
        url=raw["url"],
        platform="chesscom",
        year=month.year,
        month=str(month),
        time_class=time_class,
        time_control=str(raw.get("time_control", "")),
        ply_count=ply_count,
        white_username=white["username"].lower(),
        white_rating=int(white["rating"]),
        white_result=classify_result(white.get("result", "")),
        black_username=black["username"].lower(),
        black_rating=int(black["rating"]),
        black_result=classify_result(black.get("result", "")),
        pgn=pgn,
        reported_accuracy_white=accuracies.get("white"),
        reported_accuracy_black=accuracies.get("black"),
    )
