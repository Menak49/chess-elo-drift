"""Stratified snowball sampling of games from a chess site's API.

Neither site offers a way to draw a random player, and the endpoints that list
accounts are dominated by the crowded middle of the rating distribution. The
sampler therefore walks the opponent graph: every archive it reads reveals more
players together with the rating they carried at that moment, which is exactly
the stratification key the study needs.

The walk is steered rather than free: at each step it visits a player drawn from
the rating band that is currently furthest from its quota, so the rating window
fills evenly instead of piling up around the middle. One crawl covers one site
in one year; the site itself is reached through a `GameSource`.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable

from chess_elo_drift import config
from chess_elo_drift.collection.sources import GameSource
from chess_elo_drift.collection.store import GameStore, SnapshotStore, VisitedLog
from chess_elo_drift.config import Era, YearMonth
from chess_elo_drift.http import ApiError
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
    """Per-cell quotas, and the bookkeeping of how far each cell still is.

    `time_classes` names the pools that existed in the period being crawled; a
    pool that did not exist yet (Lichess rapid before 2018) has no quota, so the
    crawl does not spend its budget looking for it.

    Given the era's `months`, each cell's quota is split evenly across them, so
    every sampled month carries the same weight and seasonality cancels out of
    the year-to-year comparison. Without the split a snowball can fill a whole
    year from the one month its first productive account happened to be in.
    """

    def __init__(
        self,
        observations_per_cell: int,
        width: int = config.CRAWL_BAND_WIDTH,
        *,
        time_classes: tuple[str, ...] = config.TIME_CLASSES,
        months: Iterable[YearMonth] | None = None,
    ) -> None:
        self.observations_per_cell = observations_per_cell
        self.bands = study_bands(width)
        self.time_classes = time_classes
        self.months: tuple[str | None, ...] = (
            tuple(str(month) for month in months) if months else (None,)
        )
        self.per_month = -(-observations_per_cell // len(self.months))
        self._counts: dict[tuple[str, int, str | None], int] = defaultdict(int)

    def _month_key(self, month: str | None) -> str | None:
        return month if self.months != (None,) else None

    def _month_deficit(self, cell: Cell, month: str | None) -> int:
        if cell[0] not in self.time_classes or month not in self.months:
            return 0
        return max(0, self.per_month - self._counts[(*cell, month)])

    def deficit(self, cell: Cell, month: str | None = None) -> int:
        """What `cell` still wants in `month`, or across all months when None."""
        if month is not None and self.months != (None,):
            return self._month_deficit(cell, month)
        return sum(self._month_deficit(cell, each) for each in self.months)

    def band_deficit(self, band: int) -> int:
        return sum(self.deficit((time_class, band)) for time_class in self.time_classes)

    def month_deficit(self, month: str) -> int:
        """What the whole design still wants from one month."""
        return sum(
            self.deficit((time_class, band), month)
            for band in self.bands
            for time_class in self.time_classes
        )

    def credit(self, cell: Cell, month: str | None = None) -> None:
        self._counts[(*cell, self._month_key(month))] += 1

    def prime(self, records: Iterable[GameRecord]) -> None:
        """Count games already collected, so a resumed crawl tops up to the target.

        Without this a second run would read `--per-cell` as "this many more"
        rather than "this many in total", and quietly double the corpus.
        """
        for record in records:
            for side in record.sides():
                band = band_floor(side.rating)
                if band is not None:
                    self.credit((record.time_class, band), record.month)

    def wants(self, cell: Cell, month: str | None = None) -> bool:
        return self.deficit(cell, month) > 0

    @property
    def is_complete(self) -> bool:
        return all(
            self.deficit((time_class, band)) == 0
            for band in self.bands
            for time_class in self.time_classes
        )

    @property
    def progress(self) -> str:
        total = len(self.bands) * len(self.time_classes) * len(self.months) * self.per_month
        collected = sum(
            min(count, self.per_month)
            for (time_class, _, month), count in self._counts.items()
            if time_class in self.time_classes and month in self.months
        )
        return f"{collected}/{total}"

    def counts_by_band(self) -> dict[int, dict[str, int]]:
        return {
            band: {
                tc: sum(self._counts[(tc, band, month)] for month in self.months)
                for tc in self.time_classes
            }
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
    snapshots_recorded: int = 0
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
            "snapshots_recorded": self.snapshots_recorded,
            "players_failed": self.players_failed,
            "per_band_counts": self.per_band_counts,
        }


class StratifiedSnowballCrawler:
    """Walks the opponent graph of one site in one year, filling the quotas band by band."""

    def __init__(
        self,
        source: GameSource,
        store: GameStore,
        era: Era,
        target: CoverageTarget,
        *,
        rng: random.Random | None = None,
        max_api_requests: int = 4000,
        months_per_player: int = 2,
        visited: VisitedLog | None = None,
        snapshots: SnapshotStore | None = None,
        max_consecutive_failures: int = 10,
        stall_requests: int = 300,
    ) -> None:
        self._source = source
        self._label = f"{source.platform} {era.name}"
        self._snapshots = snapshots
        self._store = store
        self._era = era
        self._target = target
        self._rng = rng or random.Random(20182025)
        self._max_requests = max_api_requests
        self._months_per_player = months_per_player
        self._max_consecutive_failures = max_consecutive_failures
        self._stall_requests = stall_requests
        self._last_progress_at = 0

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

        A crawl also stops once `stall_requests` requests in a row have filled
        nothing. Some cells cannot be filled at all -- Lichess had a rating floor
        of 800 until 2019, and chess.com rapid above 2200 barely existed while it
        meant fifteen-minute games -- and without this rule every such year would
        spend its whole budget looking for them.
        """
        seed_queue = deque(seeds)
        self._resume_frontier()
        consecutive_failures = 0
        self._last_progress_at = self.stats.api_requests

        while not self._target.is_complete and self.stats.api_requests < self._max_requests:
            if self.stats.api_requests - self._last_progress_at >= self._stall_requests:
                logger.warning(
                    "[%s] no quota filled in %d requests, stopping: coverage %s",
                    self._label, self._stall_requests, self._target.progress,
                )
                break
            username = self._next_player(seed_queue)
            if username is None:
                logger.warning("[%s] frontier exhausted, stopping early", self._label)
                break
            try:
                self._visit(username)
            except ApiError as exc:
                self.stats.players_failed += 1
                consecutive_failures += 1
                logger.warning(
                    "[%s] skipping %s: %s (%d in a row)",
                    self._label, username, exc, consecutive_failures,
                )
                if consecutive_failures >= self._max_consecutive_failures:
                    logger.error(
                        "[%s] %d consecutive API failures, stopping: the API looks unavailable",
                        self._label, consecutive_failures,
                    )
                    break
            else:
                consecutive_failures = 0
            if self.stats.players_visited % 50 == 0:
                # Checkpointed, not only saved at the end: a crawl stopped from
                # outside would otherwise re-walk every account it had visited.
                if isinstance(self._visited, VisitedLog):
                    self._visited.save()
                logger.info(
                    "[%s] %d visited (%d active), %d requests, %d games, coverage %s",
                    self._label,
                    self.stats.players_visited,
                    self.stats.players_active_in_era,
                    self.stats.api_requests,
                    self.stats.games_stored,
                    self._target.progress,
                )

        self.stats.per_band_counts = self._target.counts_by_band()
        return self.stats

    # -- internals ----------------------------------------------------------

    def _resume_frontier(self) -> None:
        """Queue the players of the games already stored, as a first crawl would have.

        A resumed crawl starts with an empty frontier and a visited log that
        already holds every seed it walked, so without this a year topped up
        after its first run (to fill months the first run never read) finds
        nobody to visit and stops at once. The opponents in its stored games
        were queued but mostly never read, and are the natural next accounts.
        """
        for record in self._store.read_all():
            self._discover(record)

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
            harvest = self._source.month_games(username, month)
            self.stats.archives_fetched += 1
            if harvest.snapshot is not None and self._snapshots is not None:
                self._snapshots.add(harvest.snapshot)
                self.stats.snapshots_recorded += 1
            for record in harvest.games:
                self.stats.games_seen += 1
                self._discover(record)
                self._consider(record)
            if harvest.games:
                return  # one productive month per account keeps the sample broad

    def _months_to_read(self, username: str) -> list[YearMonth]:
        """Pick which of this account's real archives to read, if any.

        Asking the index first costs one request and removes the guesswork:
        accounts that were dormant or did not yet exist during the era are
        dropped here instead of through a string of empty downloads.
        """
        if not getattr(self._source, "index_is_free", False):
            self.stats.index_requests += 1
        available = self._source.active_months(username, self._era)
        if not available:
            return []
        months = self._rng.sample(available, k=len(available))
        if self._target.months != (None,):
            # Neediest month first, and none whose quota is already met. The
            # sort is stable, so ties keep the random order drawn above.
            months = [month for month in months if self._target.month_deficit(str(month)) > 0]
            months.sort(key=lambda month: self._target.month_deficit(str(month)), reverse=True)
        return months[: self._months_per_player]

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
            if not self._target.wants(cell, record.month):
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
                self._target.credit(cell, record.month)
                self._last_progress_at = self.stats.api_requests
                self._player_cell_counts[(side.username, *cell)] += 1
