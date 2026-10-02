"""Reading a rating pool out of each site's player payload.

The two sites describe a rating differently, and both describe it as of the
player's *latest* game, not of a shared moment. What each keeps that lets the
analysis judge how far a rating can be trusted:

* chess.com: the rating deviation `rd`, the date of the last game in the pool
  and the number of games played in it. A pool untouched for years keeps its old
  rating and a large `rd`.
* Lichess: the rating deviation `rd`, the number of games and the
  `provisional` flag, which Lichess raises while `rd` is still above 110.
  Lichess exposes no per-pool date, only the account's `seenAt`.

The pools are stored as they were reported. Deciding who is established enough
to enter a conversion formula is the analysis' job, not the collector's.
"""

from __future__ import annotations

from typing import Any, Mapping

from chess_elo_drift.config import TIME_CLASSES


def chesscom_pool(stats: Mapping[str, Any], time_class: str) -> dict[str, Any] | None:
    """One chess.com pool from a `/pub/player/{name}/stats` payload, or None."""
    block = stats.get(f"chess_{time_class}")
    if not isinstance(block, Mapping):
        return None
    last = block.get("last")
    if not isinstance(last, Mapping) or last.get("rating") is None:
        return None
    record = block.get("record") or {}
    best = block.get("best") or {}
    return {
        "rating": int(last["rating"]),
        "rd": _int_or_none(last.get("rd")),
        "last_date": _int_or_none(last.get("date")),
        "games": sum(int(record.get(key) or 0) for key in ("win", "loss", "draw")),
        "best": _int_or_none(best.get("rating")),
    }


def chesscom_pools(stats: Mapping[str, Any]) -> dict[str, dict[str, Any] | None]:
    return {time_class: chesscom_pool(stats, time_class) for time_class in TIME_CLASSES}


def lichess_pool(user: Mapping[str, Any], time_class: str) -> dict[str, Any] | None:
    """One Lichess pool from a user object of `POST /api/users`, or None."""
    perf = (user.get("perfs") or {}).get(time_class)
    if not isinstance(perf, Mapping) or perf.get("rating") is None:
        return None
    return {
        "rating": int(perf["rating"]),
        "rd": _int_or_none(perf.get("rd")),
        "games": int(perf.get("games") or 0),
        "provisional": bool(perf.get("prov", False)),
    }


def lichess_pools(user: Mapping[str, Any]) -> dict[str, dict[str, Any] | None]:
    return {time_class: lichess_pool(user, time_class) for time_class in TIME_CLASSES}


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)
