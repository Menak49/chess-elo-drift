"""Bootstrapping the crawl with a pool of starting accounts.

Seeds only bootstrap the walk; they are not the sample. The crawler visits an
account from the opponent graph whenever one is available and falls back to the
seed pool only when a rating band has no candidates left, so the great majority
of sampled players are discovered through games rather than through this module.

Two rules matter for the comparison to stay honest:

* The construction is *identical for every year*. Seeds are club rosters filtered
  to accounts that already existed when the era began, because an account cannot
  be sampled in a period that predates it. Applying the same rule to every year
  keeps the bootstrap from favouring one of them.
* The filter constrains the *seeds only*. Accounts created during an era are
  still sampled in full, since the crawler reaches them through the games its
  seeds played -- which is precisely the influx of new players the study is
  about.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from chess_elo_drift.chesscom import ChessComClient, fetch_club_members_with_dates, fetch_country_players
from chess_elo_drift.config import Era

logger = logging.getLogger(__name__)

#: Large national and interest clubs whose rosters span the whole rating range
#: and reach far enough back to seed the earliest years.
DEFAULT_SEED_CLUBS: tuple[str, ...] = (
    "team-usa", "team-england", "team-germany", "team-spain", "team-brazil",
    "team-netherlands", "team-poland", "team-france", "team-australia",
    "team-sweden", "chess-com-developer-community",
)

#: Fallback breadth, used only once the club pool is exhausted. Country rosters
#: expose no join date, so they cannot be filtered by era and are mostly duds
#: for the older period.
DEFAULT_SEED_COUNTRIES: tuple[str, ...] = ("US", "IN", "GB", "FR", "DE", "BR", "ES", "PL", "NL", "CA")


def load_seeds_for_era(
    client: ChessComClient,
    era: Era,
    cache_dir: Path,
    *,
    clubs: tuple[str, ...] = DEFAULT_SEED_CLUBS,
    countries: tuple[str, ...] = DEFAULT_SEED_COUNTRIES,
) -> list[str]:
    """Return the era's seed accounts, club-derived ones first.

    The returned order is meaningful: the crawler consumes it front to back.
    The rosters are fetched once and cached; each era then filters the same
    pool by its own start date, so thirteen years cost one download.
    """
    rng = random.Random(era.name)
    pool = _load_pool(client, cache_dir, clubs, countries)
    cutoff = _era_start_epoch(era)

    club_seeds = {name for name, joined in pool["clubs"] if joined and joined < cutoff}
    country_seeds = set(pool["countries"]) - club_seeds
    ordered = _shuffled(club_seeds, rng) + _shuffled(country_seeds, rng)
    logger.info("[chesscom %s] %d seeds (%d from clubs)", era.name, len(ordered), len(club_seeds))
    return ordered


def _load_pool(
    client: ChessComClient, cache_dir: Path, clubs: tuple[str, ...], countries: tuple[str, ...]
) -> dict[str, list]:
    cache_path = cache_dir / "seed_pool.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    members: dict[str, int] = {}
    for club in clubs:
        roster = fetch_club_members_with_dates(client, club)
        for name, joined in roster:
            members[name] = min(members.get(name, joined), joined)
        logger.info("club %s: %d members", club, len(roster))

    country_players: set[str] = set()
    for country in countries:
        country_players |= set(fetch_country_players(client, country))

    pool = {"clubs": sorted(members.items()), "countries": sorted(country_players)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(pool), encoding="utf-8")
    return pool


def _era_start_epoch(era: Era) -> float:
    first = min(era.months)
    return datetime(first.year, first.month, 1, tzinfo=timezone.utc).timestamp()


def _shuffled(names: set[str], rng: random.Random) -> list[str]:
    ordered = sorted(names)
    rng.shuffle(ordered)
    return ordered
