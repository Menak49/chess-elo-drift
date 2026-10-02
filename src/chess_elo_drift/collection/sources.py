"""What the crawler needs from a chess site, and chess.com's answer to it.

The snowball walk is the same on both sites: find out whether an account played
during the year, read one of its months, keep the games and queue the opponents.
Only the two reads differ, so they sit behind `GameSource` and the crawler never
learns which site it is walking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from chess_elo_drift import config
from chess_elo_drift.chesscom import ChessComClient, fetch_available_months, fetch_monthly_archive
from chess_elo_drift.collection.extraction import extract_game
from chess_elo_drift.config import Era, YearMonth
from chess_elo_drift.records import GameRecord, RatingSnapshot


@dataclass(frozen=True)
class MonthHarvest:
    """Everything one read of one player's month produced.

    `games` are the usable, study-eligible games; `snapshot` summarises the
    player's own ratings that month from *every* rated game of both cadences,
    including the short ones `games` leaves out, or None if they played none.
    """

    games: list[GameRecord]
    snapshot: RatingSnapshot | None


class GameSource(Protocol):
    """One chess site, as seen by the crawler.

    A source whose `active_months` answers from memory rather than the network
    sets `index_is_free`, so the crawler's request budget counts real requests.
    """

    platform: str
    index_is_free: bool

    def active_months(self, username: str, era: Era) -> list[YearMonth]:
        """The era's sampled months in which `username` played blitz or rapid.

        Costs one request. An account that does not exist, or did not play
        during the era, yields an empty list rather than an error.
        """
        ...

    def month_games(self, username: str, month: YearMonth) -> MonthHarvest:
        """Read `username`'s games of `month`. Costs one request."""
        ...


class ChessComSource:
    """chess.com, through the public archive endpoints."""

    platform = "chesscom"
    index_is_free = False

    def __init__(self, client: ChessComClient) -> None:
        self._client = client

    def active_months(self, username: str, era: Era) -> list[YearMonth]:
        # The archive index lists months with *any* game, daily included, so
        # this over-reports slightly; an empty month read costs one request.
        return [month for month in fetch_available_months(self._client, username) if month in era]

    def month_games(self, username: str, month: YearMonth) -> MonthHarvest:
        raw_games = fetch_monthly_archive(self._client, username, month)
        games = [
            record
            for record in (extract_game(raw, month=month) for raw in raw_games)
            if record is not None
        ]
        return MonthHarvest(games, chesscom_snapshot(raw_games, username, month))


def chesscom_snapshot(
    raw_games: Iterable[dict[str, Any]], username: str, month: YearMonth
) -> RatingSnapshot | None:
    """Summarise `username`'s blitz and rapid ratings over one archive month.

    chess.com prints each side's rating after the game on the game itself, so
    the last game of a cadence carries the rating the month ended on.
    """
    name = username.lower()
    latest: dict[str, tuple[int, int]] = {}
    counts = {time_class: 0 for time_class in config.TIME_CLASSES}

    for raw in raw_games:
        time_class = raw.get("time_class")
        if raw.get("rules") != "chess" or not raw.get("rated") or time_class not in counts:
            continue
        for colour in ("white", "black"):
            side = raw.get(colour)
            if not isinstance(side, dict) or str(side.get("username", "")).lower() != name:
                continue
            if not side.get("rating"):
                continue
            counts[time_class] += 1
            ended = int(raw.get("end_time") or 0)
            if time_class not in latest or ended >= latest[time_class][0]:
                latest[time_class] = (ended, int(side["rating"]))

    if not any(counts.values()):
        return None
    return RatingSnapshot(
        platform="chesscom",
        username=name,
        month=str(month),
        blitz_rating=latest["blitz"][1] if "blitz" in latest else None,
        blitz_games=counts["blitz"],
        rapid_rating=latest["rapid"][1] if "rapid" in latest else None,
        rapid_games=counts["rapid"],
    )
