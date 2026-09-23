"""Typed wrappers over the handful of chess.com endpoints this study needs."""

from __future__ import annotations

import re
from typing import Any

from chess_elo_drift.chesscom.client import ChessComClient, NotFoundError
from chess_elo_drift.config import YearMonth

_ARCHIVE_URL_PATTERN = re.compile(r"/games/(\d{4})/(\d{2})$")


def fetch_available_months(client: ChessComClient, username: str) -> list[YearMonth]:
    """Return every month in which `username` has an archive.

    One call to this index replaces guessing months blind. It is what makes the
    older era affordable to crawl at all: the overwhelming majority of accounts
    simply did not exist in 2018, and this endpoint says so in a single request
    instead of one wasted download per month tried.
    """
    try:
        payload = client.get(f"player/{username}/games/archives")
    except NotFoundError:
        return []

    months: list[YearMonth] = []
    for url in payload.get("archives", []):
        match = _ARCHIVE_URL_PATTERN.search(url)
        if match:
            months.append(YearMonth(int(match.group(1)), int(match.group(2))))
    return months


def fetch_monthly_archive(
    client: ChessComClient, username: str, month: YearMonth
) -> list[dict[str, Any]]:
    """Return every game `username` finished during `month`.

    An account with no games that month, or one that has since been closed,
    yields an empty list rather than an error: both are normal during a crawl.
    """
    try:
        payload = client.get(f"player/{username}/games/{month.path_fragment}")
    except NotFoundError:
        return []
    return payload.get("games", [])


def fetch_country_players(client: ChessComClient, country_code: str) -> list[str]:
    """Return usernames registered under an ISO country code (capped by the API)."""
    try:
        payload = client.get(f"country/{country_code.upper()}/players")
    except NotFoundError:
        return []
    return list(payload.get("players", []))


def fetch_club_members_with_dates(client: ChessComClient, club_id: str) -> list[tuple[str, int]]:
    """Return `(username, joined_epoch)` for every member of a club.

    The join date is the club join date, not the account creation date, but it
    bounds it from above: nobody joins a club before registering. That is enough
    to select accounts that already existed at the start of an era.
    """
    try:
        payload = client.get(f"club/{club_id}/members")
    except NotFoundError:
        return []

    members: dict[str, int] = {}
    for bucket in ("weekly", "monthly", "all_time"):
        for member in payload.get(bucket, []):
            if not isinstance(member, dict) or not member.get("username"):
                continue
            username = member["username"].lower()
            joined = int(member.get("joined", 0))
            members[username] = min(members.get(username, joined), joined)
    return sorted(members.items())
