"""Where the Lichess usernames of the survey come from.

Lichess has no list of its players, so the survey has to find them where they
gather. Each source below yields `(username, source)` pairs and the source is
kept on every row, because each one has its own bias and the analysis may want to
check that a conversion formula does not depend on it:

    games_2026      accounts that appeared in the collected 2026 Lichess games
    snapshots_2026  accounts the crawler read a 2026 month of
    seed_pool       the crawler's seed accounts, seen in 2026
    leaderboard     the top 100 of each pool: the only source above 2600, and
                    the one whose profiles link chess.com most often
    team_leader     the leaders of Lichess teams, from the list of the biggest
                    teams and from team searches. Organisers, coaches and club
                    owners: engaged players who advertise themselves, so their
                    profiles link a chess.com account far more often (5 to 9 %
                    for teams found by searching "chess.com", "rating" or "coach")
                    than a tournament player's do (well under 1 %). They are also
                    the players least like everyone else, which the analysis
                    has to keep in mind when it reads a conversion off them.
    arena:<id>      everyone who played a finished or running blitz or rapid
                    arena, read from its standings. These players are active by
                    construction and lean towards regulars who like tournaments.

Only ordinary JSON endpoints are used. Lichess allows a caller one streaming
request at a time and locks it out of every streaming endpoint for about an hour
if one is dropped mid-read, and the Lichess game crawl of this study needs that
slot. Team member lists and tournament result streams, which would give far more
usernames, are therefore left alone.

Every source can fail without ending the harvest; a failed request is logged and
the next source is tried.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Protocol

from chess_elo_drift import config
from chess_elo_drift.http import ApiError

logger = logging.getLogger(__name__)

#: Arenas smaller than this add too little for the requests they cost.
MIN_TOURNAMENT_PLAYERS = 30

#: Arenas of any standard clock are read: a bullet arena's players have blitz and
#: rapid ratings too, and the analysis filters on those pools' own reliability.
TOURNAMENT_SPEEDS = ("blitz", "rapid", "bullet", "classical")

#: Every pool's leaderboard is read: a top player's blitz and rapid ratings are
#: both wanted, whichever board first surfaced the name.
LEADERBOARDS = ("blitz", "rapid", "bullet", "classical", "ultraBullet")

#: Lichess lists 15 teams per page, biggest first, each with its leaders, and
#: refuses any page beyond the fortieth, for the listing and for a search alike.
TEAM_PAGES = 40

#: Searches whose teams' leaders are read, most productive first. A search
#: matches a team's name and description, so each term reaches a different set.
TEAM_SEARCH_TERMS = (
    "chess.com", "chesscom", "chess com", "rating", "elo", "coach", "coaching",
    "academy", "school", "club", "blitz", "rapid", "tournament", "league", "learn",
    "improve", "beginners", "players", "online", "official", "community", "friends",
    "training", "tactics", "openings", "university", "students", "master",
    "grandmaster", "streamer", "youtube", "twitch", "discord", "fan club",
)

#: A standings page holds ten players.
STANDINGS_PAGE_SIZE = 10
MAX_STANDINGS_PAGES = 150

#: Earliest `seenAt` (ms) counted as activity in the conversion year.
YEAR_START_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z


class LichessJson(Protocol):
    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any: ...


class LichessHarvester:
    """Yields candidate Lichess usernames, most valuable sources first."""

    def __init__(
        self,
        client: LichessJson,
        *,
        games_path: Path | None = None,
        snapshots_path: Path | None = None,
        seed_pool_path: Path | None = None,
    ) -> None:
        self._client = client
        self._games_path = games_path
        self._snapshots_path = snapshots_path
        self._seed_pool_path = seed_pool_path
        self._seen: set[str] = set()

    def candidates(self) -> Iterator[tuple[str, str]]:
        sources = (
            self._from_games,
            self._from_snapshots,
            self._from_seed_pool,
            self._from_leaderboards,
            self._from_team_leaders,
            self._from_team_searches,
            self._from_tournaments,
        )
        for source in sources:
            try:
                for username, label in source():
                    key = str(username).lower()
                    if key and key not in self._seen:
                        self._seen.add(key)
                        yield key, label
            except ApiError as exc:
                logger.warning("harvest source %s failed: %s", source.__name__, exc)

    # -- files the crawler wrote ----------------------------------------------

    def _from_games(self) -> Iterator[tuple[str, str]]:
        for game in _read_jsonl(self._games_path):
            for side in ("white_username", "black_username"):
                if game.get(side):
                    yield game[side], "games_2026"

    def _from_snapshots(self) -> Iterator[tuple[str, str]]:
        for row in _read_jsonl(self._snapshots_path):
            if str(row.get("month", "")).startswith(str(config.CONVERSION_YEAR)) and row.get("username"):
                yield row["username"], "snapshots_2026"

    def _from_seed_pool(self) -> Iterator[tuple[str, str]]:
        if self._seed_pool_path is None or not self._seed_pool_path.exists():
            return
        try:
            pool = json.loads(self._seed_pool_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for account in pool.get("accounts", []):
            if (account.get("seen") or 0) >= YEAR_START_MS and account.get("id"):
                yield account["id"], "seed_pool"

    # -- Lichess endpoints ----------------------------------------------------

    def _from_leaderboards(self) -> Iterator[tuple[str, str]]:
        for perf in LEADERBOARDS:
            payload = self._client.get(f"api/player/top/200/{perf}")
            for user in payload.get("users", []):
                yield user.get("id") or user.get("username", ""), f"leaderboard:{perf}"

    def _from_team_leaders(self) -> Iterator[tuple[str, str]]:
        yield from self._leaders("api/team/all", {}, "team_leader:biggest")

    def _from_team_searches(self) -> Iterator[tuple[str, str]]:
        for term in TEAM_SEARCH_TERMS:
            try:
                yield from self._leaders("api/team/search", {"text": term}, f"team_leader:{term}")
            except ApiError as exc:
                logger.warning("team search %r failed: %s", term, exc)

    def _leaders(self, path: str, params: dict[str, Any], label: str) -> Iterator[tuple[str, str]]:
        for page in range(1, TEAM_PAGES + 1):
            payload = self._client.get(path, params={**params, "page": page})
            for team in payload.get("currentPageResults", []):
                for leader in team.get("leaders") or []:
                    yield leader.get("id") or leader.get("name", ""), label
            if not payload.get("nextPage"):
                break

    def _from_tournaments(self) -> Iterator[tuple[str, str]]:
        listing = self._client.get("api/tournament")
        tournaments = [*listing.get("finished", []), *listing.get("started", [])]
        usable = [
            item for item in tournaments
            if (item.get("variant") or {}).get("key", "standard") == "standard"
            and (item.get("perf") or {}).get("key") in TOURNAMENT_SPEEDS
            and (item.get("nbPlayers") or 0) >= MIN_TOURNAMENT_PLAYERS
        ]
        for item in sorted(usable, key=lambda entry: -(entry.get("nbPlayers") or 0)):
            label = f"arena:{item['id']}"
            try:
                yield from self._standings(item["id"], item.get("nbPlayers") or 0, label)
            except ApiError as exc:
                logger.warning("standings of %s unavailable: %s", item["id"], exc)

    def _standings(self, tournament_id: str, players: int, label: str) -> Iterator[tuple[str, str]]:
        pages = min(-(-players // STANDINGS_PAGE_SIZE), MAX_STANDINGS_PAGES)
        for page in range(1, pages + 1):
            payload = self._client.get(f"api/tournament/{tournament_id}", params={"page": page})
            rows = (payload.get("standing") or {}).get("players") or []
            if not rows:
                break
            for row in rows:
                yield row.get("name", ""), label
        logger.info("%s: read %d standings pages", label, pages)


def _read_jsonl(path: Path | None) -> Iterator[dict]:
    if path is None or not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
