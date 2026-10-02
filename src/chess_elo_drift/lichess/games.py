"""Typed wrappers over the Lichess endpoints the study reads.

Each function is one request. They return plain data and leave every decision
about it (which games count, which months are sampled) to the callers, so this
module is the only place that knows the wire format.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, NamedTuple

from chess_elo_drift.config import YearMonth
from chess_elo_drift.http import NotFoundError
from chess_elo_drift.lichess.client import LichessClient

#: `POST /api/users` accepts at most this many comma-separated ids.
BULK_USERS_LIMIT = 300


class RatingPoint(NamedTuple):
    """One day on which a player's rating in a cadence changed."""

    month: YearMonth
    day: int
    rating: int


def month_bounds_ms(month: YearMonth) -> tuple[int, int]:
    """Epoch milliseconds (UTC) of the start of `month` and of the month after."""
    following = YearMonth(month.year + month.month // 12, month.month % 12 + 1)
    return _epoch_ms(month), _epoch_ms(following)


def fetch_rating_history(client: LichessClient, username: str) -> dict[str, list[RatingPoint]]:
    """A player's rating history, keyed by the cadence name Lichess prints ("Blitz").

    The wire format is `[year, month, day, rating]` where *month is zero-based*,
    the JavaScript convention; it is converted here so that nothing downstream
    has to remember. An account that does not exist or was closed yields {}.
    """
    try:
        payload = client.get(f"/api/user/{username}/rating-history")
    except NotFoundError:
        return {}
    if not isinstance(payload, list):
        return {}

    history: dict[str, list[RatingPoint]] = {}
    for series in payload:
        history[str(series.get("name"))] = [
            RatingPoint(YearMonth(year, month0 + 1), day, rating)
            for year, month0, day, rating in series.get("points", [])
        ]
    return history


def fetch_month_games(
    client: LichessClient, username: str, month: YearMonth, *, max_games: int
) -> list[dict[str, Any]]:
    """Rated blitz and rapid games `username` started in `month`, newest first.

    `max_games` bounds the cost of one busy account: an uncapped read of a
    prolific account can take minutes. Moves come as SAN in the JSON and the PGN
    is embedded, which saves a second request per game; clocks, evaluations and
    openings are left out because the study recomputes everything from the moves.
    """
    since, until = month_bounds_ms(month)
    params = {
        "since": since,
        "until": until,
        "rated": "true",
        "perfType": "blitz,rapid",
        "pgnInJson": "true",
        "clocks": "false",
        "evals": "false",
        "opening": "false",
        "moves": "true",
        "max": max_games,
    }
    try:
        return client.get_ndjson(f"/api/games/user/{username}", params=params)
    except NotFoundError:
        return []


def fetch_popular_team_leaders(client: LichessClient, page: int) -> list[str]:
    """Account ids of the leaders of the teams on one page of the popularity ranking.

    Fifteen teams per page, and the page already lists their leaders, so this is
    the cheapest way to meet people who have run something on Lichess for years.
    """
    payload = client.get("/api/team/all", params={"page": page})
    return [
        leader["id"]
        for team in payload.get("currentPageResults", [])
        for leader in team.get("leaders", [])
        if leader.get("id")
    ]


def fetch_leaderboard(client: LichessClient, perf: str, count: int) -> list[str]:
    """Account ids on the current leaderboard of `perf` (`blitz`, `rapid`, ...)."""
    payload = client.get(f"/api/player/top/{count}/{perf}")
    return [user["id"] for user in payload.get("users", []) if user.get("id")]


def fetch_users(client: LichessClient, usernames: list[str]) -> list[dict[str, Any]]:
    """Public profiles (with `createdAt`) of up to `BULK_USERS_LIMIT` accounts, in one request."""
    if len(usernames) > BULK_USERS_LIMIT:
        raise ValueError(f"at most {BULK_USERS_LIMIT} users per request, got {len(usernames)}")
    if not usernames:
        return []
    return client.post_text("/api/users", ",".join(usernames))


def _epoch_ms(month: YearMonth) -> int:
    return int(datetime(month.year, month.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
