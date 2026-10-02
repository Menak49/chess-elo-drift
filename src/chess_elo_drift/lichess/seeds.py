"""Bootstrapping the Lichess crawl with a pool of starting accounts.

Seeds only bootstrap the walk; they are not the sample. The crawler visits an
account from the opponent graph whenever one is available and falls back to the
seed pool only when a rating band has no candidates left, so the great majority
of sampled players are discovered through games rather than through this module.
What the walk does need is a few accounts that really played in *each* year,
the earliest included, and Lichess offers no cheap list of "players active in
2014". Two things it does list cheaply are people who have kept a mark on the
site for a long time:

* the leaders of the most popular teams, fifteen teams and their leaders per
  request, and
* the current leaderboards, whose long-standing entries are old accounts.

Neither list says when an account was created or last seen, so the ids are
looked up in bulk (300 per request), which also returns how many rated games
each has played. That is enough to filter every year with the same rule:

* An account qualifies for an era if it was created before the era began and
  was seen on the site at or after its start. Both bounds are constructed the
  same way for every year, so the bootstrap does not favour one of them. The
  filter constrains the *seeds only*; accounts created during an era are still
  sampled in full, since the crawler reaches them through the games its seeds
  played.
* Bots, accounts flagged for terms-of-service violations and closed accounts
  never qualify, and neither do accounts with too few lifetime blitz and rapid
  games to have plausibly played in any given year.

Ratings on a leaderboard sit far above the study's window, and the popular
teams' leaders are an uneven lot, so the seeds are dealt out round-robin across
rating bands (by the account's current rating, a rough proxy for what it had
then) rather than in one long shuffle. The pool costs about fifty light JSON
requests, none of them a stream, and is cached on disk and shared by all
thirteen years; each era then takes its own deterministic shuffle of it.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

from chess_elo_drift import config
from chess_elo_drift.config import Era
from chess_elo_drift.lichess.client import LichessClient
from chess_elo_drift.lichess.games import (
    BULK_USERS_LIMIT,
    fetch_leaderboard,
    fetch_popular_team_leaders,
    fetch_users,
)

logger = logging.getLogger(__name__)

#: How far down the popularity ranking to read team leaders, in pages of 15 teams.
#: Lichess answers 400 for a page beyond the 40th, so this is also the most there is.
DEFAULT_TEAM_PAGES = 40

DEFAULT_LEADERBOARDS: tuple[str, ...] = ("bullet", "blitz", "rapid", "classical")
LEADERBOARD_SIZE = 200

#: Lifetime blitz plus rapid games below which an account is too thin to have
#: been playing in any particular year.
MIN_LIFETIME_GAMES = 100

POOL_FILE = "seed_pool.json"
POOL_VERSION = 1


def load_lichess_seeds(
    client: LichessClient,
    era: Era,
    cache_dir: Path,
    *,
    team_pages: int = DEFAULT_TEAM_PAGES,
    leaderboards: tuple[str, ...] = DEFAULT_LEADERBOARDS,
) -> list[str]:
    """Return the era's seed accounts, spread across rating bands.

    The returned order is meaningful: the crawler consumes it front to back.
    The pool is fetched once and cached; each era then filters it by its own
    start date, so thirteen years cost one download.
    """
    pool = _load_pool(client, cache_dir, team_pages, leaderboards)
    start_ms = _era_start_ms(era)
    eligible = [
        account for account in pool
        if account["created"] < start_ms <= account["seen"]
    ]
    ordered = _spread_across_bands(eligible, random.Random(era.name))
    logger.info("[lichess %s] %d seeds (of %d in the pool)", era.name, len(ordered), len(pool))
    return ordered


def _load_pool(
    client: LichessClient, cache_dir: Path, team_pages: int, leaderboards: tuple[str, ...]
) -> list[dict[str, Any]]:
    cache_path = cache_dir / POOL_FILE
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("version") == POOL_VERSION:
            return cached["accounts"]

    candidates: dict[str, None] = {}
    for perf in leaderboards:
        candidates.update(dict.fromkeys(fetch_leaderboard(client, perf, LEADERBOARD_SIZE)))
    for page in range(1, team_pages + 1):
        candidates.update(dict.fromkeys(fetch_popular_team_leaders(client, page)))
    logger.info("lichess seed candidates: %d accounts", len(candidates))

    ids = list(candidates)
    accounts = []
    for start in range(0, len(ids), BULK_USERS_LIMIT):
        for profile in fetch_users(client, ids[start:start + BULK_USERS_LIMIT]):
            account = _pool_entry(profile)
            if account is not None:
                accounts.append(account)
    accounts.sort(key=lambda account: account["id"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"version": POOL_VERSION, "accounts": accounts}), encoding="utf-8"
    )
    return accounts


def _pool_entry(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Reduce a bulk profile to what seeding needs, or None if it cannot be a seed."""
    if profile.get("disabled") or profile.get("tosViolation") or profile.get("title") == "BOT":
        return None
    if not profile.get("createdAt") or not profile.get("seenAt"):
        return None

    perfs = profile.get("perfs") or {}
    games = sum(perfs.get(time_class, {}).get("games", 0) for time_class in config.TIME_CLASSES)
    if games < MIN_LIFETIME_GAMES:
        return None
    return {
        "id": profile["id"],
        "created": int(profile["createdAt"]),
        "seen": int(profile["seenAt"]),
        "rating": _current_rating(perfs),
    }


def _current_rating(perfs: dict[str, Any]) -> int:
    """The rating in the cadence the account has played most, blitz or rapid."""
    time_class = max(config.TIME_CLASSES, key=lambda name: perfs.get(name, {}).get("games", 0))
    return int(perfs.get(time_class, {}).get("rating", config.RATING_MIN))


def _spread_across_bands(accounts: list[dict[str, Any]], rng: random.Random) -> list[str]:
    """Shuffle within each rating band, then deal the bands out round-robin."""
    bands: dict[int, list[str]] = defaultdict(list)
    for account in sorted(accounts, key=lambda account: account["id"]):
        bands[account["rating"] // config.CRAWL_BAND_WIDTH].append(account["id"])

    columns = []
    for band in sorted(bands):
        rng.shuffle(bands[band])
        columns.append(bands[band])
    return [name for row in zip_longest(*columns) for name in row if name is not None]


def _era_start_ms(era: Era) -> int:
    first = min(era.months)
    return int(datetime(first.year, first.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
