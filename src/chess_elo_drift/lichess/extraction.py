"""Turning raw Lichess game exports into validated `GameRecord`s.

Like its chess.com counterpart this module answers one question only: *is this
export a usable game, and what does it say?* Which games the study wants is the
sampler's business.

Two things about Lichess are worth knowing when reading the rules below. Its
API labels every game with the speed *today's* rules would give it, while the
rating printed on the game belongs to the pool of the time; and a rated game
carries no explicit "human" flag, so bots and engine opponents are recognised by
the fields they lack or the title they hold.
"""

from __future__ import annotations

from typing import Any

from chess_elo_drift import config
from chess_elo_drift.config import YearMonth
from chess_elo_drift.records import GameRecord

#: Games that never got going, or whose outcome says nothing about strength.
#: `cheat` is a game annulled after a player was flagged for engine use.
REJECTED_STATUSES = frozenset(
    {"aborted", "noStart", "created", "started", "unknownFinish", "cheat"}
)

COLOURS = ("white", "black")


def is_rated_standard_game(raw: dict[str, Any]) -> bool:
    """Whether `raw` is a finished, rated game of ordinary chess."""
    return (
        raw.get("variant") == "standard"
        and bool(raw.get("rated"))
        and raw.get("status") not in REJECTED_STATUSES
    )


def study_time_class(raw: dict[str, Any], month: YearMonth) -> str | None:
    """The time class of `raw` if its rating belongs to a pool the study uses.

    Lichess only split Rapid out of Classical in December 2017 and relabels old
    games retroactively, so a 2016 `10+0` game comes back as "rapid" while the
    rating on it is a Classical one. Those are dropped, not relabelled. Requiring
    `perf` to agree with `speed` is a second guard against a rating that does
    not belong to the pool named on the game.
    """
    speed = raw.get("speed")
    if speed not in config.TIME_CLASSES or raw.get("perf") != speed:
        return None
    if not config.pool_exists("lichess", speed, month):
        return None
    return speed


def human_side(raw: dict[str, Any], colour: str) -> dict[str, Any] | None:
    """The player block of `colour` if it is a rated human account, else None.

    Engine opponents come back as `{"aiLevel": n}` with no user at all, and bot
    accounts hold the BOT title. Neither plays at a strength a rating measures
    in the way the study needs.
    """
    side = (raw.get("players") or {}).get(colour)
    if not isinstance(side, dict) or "aiLevel" in side:
        return None
    user = side.get("user")
    if not isinstance(user, dict) or user.get("title") == "BOT":
        return None
    if not (user.get("id") or user.get("name")) or not side.get("rating"):
        return None
    return side


def username_of(side: dict[str, Any]) -> str:
    user = side["user"]
    return str(user.get("id") or user["name"]).lower()


def extract_game(raw: dict[str, Any], *, month: YearMonth) -> GameRecord | None:
    """Build a `GameRecord`, or return `None` if the export is not usable.

    Rejected: variants, unrated and from-position games, unfinished games,
    time classes (or pre-2018 "rapid") outside the study, games against bots or
    the engine, and games too short to say anything about playing strength.
    """
    if not is_rated_standard_game(raw) or raw.get("initialFen"):
        return None
    time_class = study_time_class(raw, month)
    if time_class is None:
        return None

    white, black = human_side(raw, "white"), human_side(raw, "black")
    if white is None or black is None:
        return None

    moves, pgn, clock = raw.get("moves"), raw.get("pgn"), raw.get("clock")
    if not moves or not pgn or not isinstance(clock, dict):
        return None
    ply_count = len(moves.split())
    if ply_count < config.MIN_PLIES:
        return None
    if "initial" not in clock or "increment" not in clock:
        return None

    game_id = str(raw["id"])
    winner = raw.get("winner")
    return GameRecord(
        game_id=f"lichess:{game_id}",
        url=f"{config.LICHESS_ROOT}/{game_id}",
        platform="lichess",
        year=month.year,
        month=str(month),
        time_class=time_class,
        time_control=f"{int(clock['initial'])}+{int(clock['increment'])}",
        ply_count=ply_count,
        white_username=username_of(white),
        white_rating=int(white["rating"]),
        white_result=_result_for("white", winner),
        black_username=username_of(black),
        black_rating=int(black["rating"]),
        black_result=_result_for("black", winner),
        pgn=pgn,
        white_provisional=bool(white.get("provisional", False)),
        black_provisional=bool(black.get("provisional", False)),
    )


def _result_for(colour: str, winner: str | None) -> str:
    if winner is None:
        return "draw"
    return "win" if winner == colour else "loss"
