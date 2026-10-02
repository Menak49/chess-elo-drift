"""Lichess as a `GameSource`.

Lichess has no per-month archive: games are exported per player over a time
window. It also has no reliable index of the months a player was active. The
obvious candidate, the rating-history endpoint, answers an empty list for most
accounts when called without authentication (it only serves what happens to be
cached), so an empty answer cannot be told apart from an idle account and using
it would either drop active players or cost a request to learn nothing.

The crawler does not need the index for the accounts it found through games,
though: a game in month M proves both players were active in M. The source
remembers that as it reads, so `active_months` is free and exact for every
opponent, and falls back to "any sampled month" for the seed accounts, where an
empty export costs one cheap request.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from chess_elo_drift import config
from chess_elo_drift.collection.sources import MonthHarvest
from chess_elo_drift.config import Era, YearMonth
from chess_elo_drift.lichess.client import LichessClient
from chess_elo_drift.lichess.extraction import (
    COLOURS,
    extract_game,
    human_side,
    is_rated_standard_game,
    study_time_class,
    username_of,
)
from chess_elo_drift.lichess.games import fetch_month_games, fetch_rating_history
from chess_elo_drift.records import RatingSnapshot

#: One account's games are read newest first and stop here. High enough that
#: an ordinary player's month is read whole, low enough that a bullet-grinding
#: account cannot hold the crawl for minutes.
DEFAULT_MAX_GAMES_PER_MONTH = 100


class LichessSource:
    """lichess.org, through the games-export endpoint.

    `use_rating_history` spends one request per account on the rating-history
    endpoint to prune months for accounts nothing is known about. It is off by
    default: it is empty for most accounts, so it saves little.
    """

    platform = "lichess"

    def __init__(
        self,
        client: LichessClient,
        *,
        max_games_per_month: int = DEFAULT_MAX_GAMES_PER_MONTH,
        use_rating_history: bool = False,
    ) -> None:
        self._client = client
        self._max_games = max_games_per_month
        self._use_history = use_rating_history
        self.index_is_free = not use_rating_history
        self._seen_playing: dict[str, set[YearMonth]] = defaultdict(set)

    def active_months(self, username: str, era: Era) -> list[YearMonth]:
        """The era's months in which `username` is known, or has to be assumed, to have played.

        Months where a game already proved the account active come first, but
        not alone: an opponent is only ever proven active in the month it was
        met in, and offering nothing else locks the whole snowball into the one
        month its first productive account was read in. The other sampled months
        follow, for the crawler to try when they are the ones its quotas need.
        With nothing proven, every sampled month is a candidate, narrowed by the
        rating history when that is switched on *and* not empty: an empty
        history is "unknown", never "idle".
        """
        proven = self._seen_playing.get(username.lower(), set())
        known = [month for month in era.months if month in proven]
        if known:
            return known + [month for month in era.months if month not in proven]
        if self._use_history:
            return self._months_from_history(username, era)
        return list(era.months)

    def month_games(self, username: str, month: YearMonth) -> MonthHarvest:
        raw_games = fetch_month_games(self._client, username, month, max_games=self._max_games)
        games = [
            record
            for record in (extract_game(raw, month=month) for raw in raw_games)
            if record is not None
        ]
        for record in games:
            for side in record.sides():
                self._seen_playing[side.username].add(month)
        return MonthHarvest(games, lichess_snapshot(raw_games, username, month))

    def _months_from_history(self, username: str, era: Era) -> list[YearMonth]:
        """Sampled months with a Blitz point, or a Rapid one once Rapid existed."""
        history = fetch_rating_history(self._client, username)
        if not any(history.values()):
            return list(era.months)
        played = {point.month for point in history.get("Blitz", [])}
        played |= {
            point.month
            for point in history.get("Rapid", [])
            if config.pool_exists("lichess", "rapid", point.month)
        }
        return [month for month in era.months if month in played]


def lichess_snapshot(
    raw_games: Iterable[dict[str, Any]], username: str, month: YearMonth
) -> RatingSnapshot | None:
    """Summarise `username`'s blitz and rapid ratings over one month of exports.

    Every rated standard game of either cadence counts, the short ones and the
    ones against bots included: they all moved the rating. Lichess prints the
    rating *before* the game and the change it caused, so the month ends on
    `rating + ratingDiff` of the last game. Where the export is capped, the
    games kept are the newest, so the closing rating stays right and the game
    count is a lower bound.
    """
    name = username.lower()
    latest: dict[str, tuple[int, int]] = {}
    counts = {time_class: 0 for time_class in config.TIME_CLASSES}

    for raw in raw_games:
        if not is_rated_standard_game(raw):
            continue
        time_class = study_time_class(raw, month)
        if time_class is None:
            continue
        for colour in COLOURS:
            side = _side_of(raw, colour, name)
            if side is None:
                continue
            counts[time_class] += 1
            ended = int(raw.get("lastMoveAt") or raw.get("createdAt") or 0)
            if time_class not in latest or ended >= latest[time_class][0]:
                closing = int(side["rating"]) + int(side.get("ratingDiff") or 0)
                latest[time_class] = (ended, closing)

    if not any(counts.values()):
        return None
    return RatingSnapshot(
        platform="lichess",
        username=name,
        month=str(month),
        blitz_rating=latest["blitz"][1] if "blitz" in latest else None,
        blitz_games=counts["blitz"],
        rapid_rating=latest["rapid"][1] if "rapid" in latest else None,
        rapid_games=counts["rapid"],
    )


def _side_of(raw: dict[str, Any], colour: str, name: str) -> dict[str, Any] | None:
    """The block of `colour` if it is the human account `name`, else None."""
    side = human_side(raw, colour)
    if side is None or username_of(side) != name:
        return None
    return side
