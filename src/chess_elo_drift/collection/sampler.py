"""Stratified snowball sampling of games from the chess.com API.

The API offers no way to draw a random player, and the endpoints that list
accounts are dominated by the crowded middle of the rating distribution. The
sampler therefore walks the opponent graph: every archive it reads reveals more
players together with the rating they carried at that moment, which is exactly
the stratification key the study needs.

The walk is steered rather than free: at each step it visits a player drawn from
the rating band that is currently furthest from its quota, so the 600-2200
window fills evenly instead of piling up around 1000.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from chess_elo_drift import config
from chess_elo_drift.chesscom import (
    ChessComClient,
    fetch_available_months,
    fetch_monthly_archive,
)
from chess_elo_drift.chesscom.client import ChessComError
from chess_elo_drift.collection.extraction import extract_game
from chess_elo_drift.collection.store import GameStore, VisitedLog
from chess_elo_drift.config import Era, YearMonth
from chess_elo_drift.records import GameRecord

logger = logging.getLogger(__name__)

#: A cell of the sampling design: one time class at one rating band.
Cell = tuple[str, int]


def band_floor(rating: int, width: int = config.CRAWL_BAND_WIDTH) -> int | None:
    """Return the lower bound of the band holding `rating`, or None if out of range."""
    if not config.RATING_MIN <= rating <= config.RATING_MAX:
        return None
    offset = (rating - config.RATING_MIN) // width * width
    return config.RATING_MIN + min(offset, config.RATING_MAX - config.RATING_MIN - width)


def study_bands(width: int = config.CRAWL_BAND_WIDTH) -> tuple[int, ...]:
    """Every band floor covering the rating window under study."""
    return tuple(range(config.RATING_MIN, config.RATING_MAX, width))


class CoverageTarget:
    """Per-cell quotas, and the bookkeeping of how far each cell still is."""

    def __init__(self, observations_per_cell: int, width: int = config.CRAWL_BAND_WIDTH) -> None:
        self.observations_per_cell = observations_per_cell
        self.bands = study_bands(width)
        self._counts: dict[Cell, int] = defaultdict(int)

    def deficit(self, cell: Cell) -> int:
        return max(0, self.observations_per_cell - self._counts[cell])

    def band_deficit(self, band: int) -> int:
        return sum(self.deficit((time_class, band)) for time_class in config.TIME_CLASSES)

    def credit(self, cell: Cell) -> None:
        self._counts[cell] += 1

    def prime(self, records: Iterable[GameRecord]) -> None:
        """Count games already collected, so a resumed crawl tops up to the target.

        Without this a second run would read `--per-cell` as "this many more"
        rather than "this many in total", and quietly double the corpus.
        """
        for record in records:
            for side in record.sides():
                band = band_floor(side.rating)
                if band is not None:
                    self.credit((record.time_class, band))

    def wants(self, cell: Cell) -> bool:
        return self.deficit(cell) > 0

    @property
    def is_complete(self) -> bool:
        return all(
            self.deficit((time_class, band)) == 0
            for band in self.bands
            for time_class in config.TIME_CLASSES
        )

    @property
    def progress(self) -> str:
        total = len(self.bands) * len(config.TIME_CLASSES) * self.observations_per_cell
        collected = sum(min(count, self.observations_per_cell) for count in self._counts.values())
        return f"{collected}/{total}"

    def counts_by_band(self) -> dict[int, dict[str, int]]:
        return {
            band: {tc: self._counts[(tc, band)] for tc in config.TIME_CLASSES}
            for band in self.bands
        }


@dataclass
class CrawlStats:
    """What a crawl did, for the run log and the methodology section."""

    archives_fetched: int = 0
    index_requests: int = 0
    games_seen: int = 0
    games_stored: int = 0
    players_visited: int = 0
    players_active_in_era: int = 0
    accuracy_field_present: int = 0
    players_failed: int = 0
    per_band_counts: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def api_requests(self) -> int:
        return self.archives_fetched + self.index_requests

    def as_dict(self) -> dict[str, object]:
        return {
            "api_requests": self.api_requests,
            "archives_fetched": self.archives_fetched,
            "index_requests": self.index_requests,
            "games_seen": self.games_seen,
            "games_stored": self.games_stored,
            "players_visited": self.players_visited,
            "players_active_in_era": self.players_active_in_era,
            "accuracy_field_present": self.accuracy_field_present,
            "players_failed": self.players_failed,
            "per_band_counts": self.per_band_counts,
        }


class StratifiedSnowballCrawler:
    """Walks the opponent graph of one era, filling the quotas band by band."""

    def __init__(
        self,
        client: ChessComClient,
        store: GameStore,
        era: Era,
        target: CoverageTarget,
        *,
        rng: random.Random | None = None,
        max_api_requests: int = 4000,
        months_per_player: int = 2,
        visited: VisitedLog | None = None,
        max_consecutive_failures: int = 10,
    ) -> None:
        self._client = client
        self._store = store
        self._era = era
        self._target = target
        self._rng = rng or random.Random(20182025)
        self._max_requests = max_api_requests
        self._months_per_player = months_per_player
        self._max_consecutive_failures = max_consecutive_failures

        self._frontier: dict[int, deque[str]] = {band: deque() for band in target.bands}
        self._queued: set[str] = set()
        self._visited: VisitedLog | set[str] = visited if visited is not None else set()
        self._player_cell_counts: dict[tuple[str, str, int], int] = defaultdict(int)
        self.stats = CrawlStats()

    # -- public API ---------------------------------------------------------

    def crawl(self, seeds: list[str]) -> CrawlStats:
        """Sample until every cell meets its quota or the request budget runs out.

        The seed order is taken as given: the provider ranks the accounts most
        likely to have been active in this era first.

        One account failing is not the crawl failing. The public API returns the
        occasional 502 on an account the client has already retried, and there
        are always more accounts in the frontier, so a failure is logged and the
        walk moves on. A run of `max_consecutive_failures` failures is different:
        that is the API being down rather than one bad account, and continuing
        would only burn through the frontier marking good accounts as visited.
        """
        seed_queue = deque(seeds)
        consecutive_failures = 0

        while not self._target.is_complete and self.stats.api_requests < self._max_requests:
            username = self._next_player(seed_queue)
            if username is None:
                logger.warning("[%s] frontier exhausted, stopping early", self._era.name)
                break
            try:
                self._visit(username)
            except ChessComError as exc:
                self.stats.players_failed += 1
                consecutive_failures += 1
                logger.warning(
                    "[%s] skipping %s: %s (%d in a row)",
                    self._era.name, username, exc, consecutive_failures,
                )
                if consecutive_failures >= self._max_consecutive_failures:
                    logger.error(
                        "[%s] %d consecutive API failures, stopping: the API looks unavailable",
                        self._era.name, consecutive_failures,
                    )
                    break
            else:
                consecutive_failures = 0
            if self.stats.players_visited % 50 == 0:
                logger.info(
                    "[%s] %d visited (%d active), %d requests, %d games, coverage %s",
                    self._era.name,
                    self.stats.players_visited,
                    self.stats.players_active_in_era,
                    self.stats.api_requests,
                    self.stats.games_stored,
                    self._target.progress,
                )

        self.stats.per_band_counts = self._target.counts_by_band()
        return self.stats

    # -- internals ----------------------------------------------------------

    def _next_player(self, seed_queue: deque[str]) -> str | None:
        """Pick the next account to visit, preferring the neediest rating band.

        Falling back to the seed pool matters early on, when the opponent graph
        has not been explored yet, and late on, when the remaining bands are
        sparse enough that their frontier can run dry.
        """
        needy_bands = sorted(
            (band for band in self._target.bands if self._frontier[band]),
            key=self._target.band_deficit,
            reverse=True,
        )
        for band in needy_bands:
            if self._target.band_deficit(band) == 0:
                break
            username = self._pop_unvisited(band)
            if username is not None:
                return username

        while seed_queue:
            username = seed_queue.popleft()
            if username not in self._visited:
                return username

        for band in needy_bands:
            username = self._pop_unvisited(band)
            if username is not None:
                return username
        return None

    def _pop_unvisited(self, band: int) -> str | None:
        queue = self._frontier[band]
        while queue:
            username = queue.popleft()
            if username not in self._visited:
                return username
        return None

    def _visit(self, username: str) -> None:
        """Read a couple of this account's months and harvest what they contain."""
        self._visited.add(username)
        self.stats.players_visited += 1

        months = self._months_to_read(username)
        if not months:
            return
        self.stats.players_active_in_era += 1

        for month in months:
            if self.stats.api_requests >= self._max_requests:
                return
            games = fetch_monthly_archive(self._client, username, month)
            self.stats.archives_fetched += 1
            for raw in games:
                record = extract_game(raw, era_name=self._era.name, month=month)
                if record is None:
                    continue
                self.stats.games_seen += 1
                self._discover(record)
                self._consider(record)
            if games:
                return  # one productive month per account keeps the sample broad

    def _months_to_read(self, username: str) -> list[YearMonth]:
        """Pick which of this account's real archives to read, if any.

        Asking the index first costs one request and removes the guesswork:
        accounts that were dormant or did not yet exist during the era are
        dropped here instead of through a string of empty downloads.
        """
        self.stats.index_requests += 1
        available = [month for month in fetch_available_months(self._client, username) if month in self._era]
        if not available:
            return []
        return self._rng.sample(available, k=min(self._months_per_player, len(available)))

    def _discover(self, record: GameRecord) -> None:
        """Queue both players of a game as future candidates, in their own band."""
        for side in record.sides():
            band = band_floor(side.rating)
            if band is None or side.username in self._visited or side.username in self._queued:
                continue
            self._frontier[band].append(side.username)
            self._queued.add(side.username)

    def _consider(self, record: GameRecord) -> None:
        """Store the game if either side fills a cell that still wants observations."""
        creditable: set[Cell] = set()
        for side in record.sides():
            band = band_floor(side.rating)
            if band is None:
                continue
            cell = (record.time_class, band)
            if not self._target.wants(cell):
                continue
            if self._player_cell_counts[(side.username, *cell)] >= config.MAX_GAMES_PER_PLAYER_PER_CELL:
                continue
            creditable.add(cell)

        if not creditable or not self._store.add(record):
            return

        self.stats.games_stored += 1
        if record.reported_accuracy_white is not None or record.reported_accuracy_black is not None:
            self.stats.accuracy_field_present += 1

        for side in record.sides():
            band = band_floor(side.rating)
            if band is None:
                continue
            cell = (record.time_class, band)
            if cell in creditable:
                self._target.credit(cell)
                self._player_cell_counts[(side.username, *cell)] += 1
