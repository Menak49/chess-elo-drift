"""More chess.com accounts for the rapid-against-blitz pairing, from game archives.

The crawl's usernames are few, so the same-player pairing on chess.com is thin.
Every rated standard game in a player's monthly archive names an opponent who
was active that month, so reading archives yields active 2026 players.

The way they are picked matters more than how many there are. An opponent is
found *because of a rating*: a player met in rapid games is one whose rapid
rating sat near the hub's. Take many opponents from one hub and the sample is
selected on that one rating, and a conversion fitted on it regresses towards the
other pool (a player picked for a high rapid rating has a lower blitz rating than
the ladder average, and the reverse). Three rules limit this:

* Many hubs, few opponents each: at most `MAX_PER_CADENCE` opponents per cadence
  per hub, so no single hub or rating neighbourhood dominates.
* Balanced cadences: up to `MAX_PER_CADENCE` from the hub's blitz games and the
  same from its rapid games. A cadence with fewer opponents is not topped up from
  the other, which would put the selection back on one rating.
* One month per hub, chosen deterministically, so the cost is about one request
  per hub. An empty month is skipped for the next.

Each opponent is stored with the hub (`via`) and the cadence it was met in
(`via_cadence`), so the analysis can cap accounts per hub or per cadence and
can check that a formula does not depend on how the account was found.
"""

from __future__ import annotations

import json
import logging
import random
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from chess_elo_drift import config
from chess_elo_drift.chesscom import fetch_monthly_archive
from chess_elo_drift.config import YearMonth
from chess_elo_drift.conversion.jsonl import IdList, KeyedJsonl
from chess_elo_drift.http import ApiError

logger = logging.getLogger("survey")

OPPONENTS_FILE = "chesscom_opponents.jsonl"
HUBS_DONE_FILE = "chesscom_hubs_done.txt"
ARCHIVE_SOURCE = "chesscom_archive_opponent"

#: Opponents kept per cadence per hub.
MAX_PER_CADENCE = 6

#: Fixed so that a re-run orders the hubs and picks the same opponents.
SHUFFLE_SEED = 20262028

MAX_CONSECUTIVE_FAILURES = 10


def opponents_by_cadence(games: Iterable[dict[str, Any]], hub: str) -> dict[str, list[str]]:
    """The hub's opponents in rated standard games, by cadence, lowercase and sorted."""
    found: dict[str, set[str]] = {cadence: set() for cadence in config.TIME_CLASSES}
    for game in games:
        if not game.get("rated") or game.get("rules", "chess") != "chess":
            continue
        cadence = game.get("time_class")
        if cadence not in found:
            continue
        white = str((game.get("white") or {}).get("username", "")).lower()
        black = str((game.get("black") or {}).get("username", "")).lower()
        if hub == white and black:
            found[cadence].add(black)
        elif hub == black and white:
            found[cadence].add(white)
    found = {cadence: names - {hub} for cadence, names in found.items()}
    return {cadence: sorted(names) for cadence, names in found.items()}


def hub_months(hub: str, months: list[YearMonth]) -> list[YearMonth]:
    """The months to try for `hub`: a deterministic first choice, then the others."""
    start = zlib.crc32(hub.encode("utf-8")) % len(months)
    return months[start:] + months[:start]


def read_hubs(games_file: Path, visited_file: Path) -> list[str]:
    """Every distinct username of the crawl's 2026 games and visited list, shuffled."""
    hubs: set[str] = set()
    if games_file.exists():
        with games_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for side in ("white_username", "black_username"):
                    if game.get(side):
                        hubs.add(str(game[side]).lower())
                del game
    if visited_file.exists():
        try:
            hubs.update(str(name).lower() for name in json.loads(visited_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            pass
    ordered = sorted(hubs)
    random.Random(SHUFFLE_SEED).shuffle(ordered)
    return ordered


def collect_opponents(
    client: Any,
    hubs: Iterable[str],
    opponents: KeyedJsonl,
    done: IdList,
    fetched: Callable[[str], bool],
    months: Iterable[YearMonth],
    wanted: int,
) -> int:
    """Read one month of each hub until `wanted` new opponents are waiting.

    An opponent already fetched, or already stored, is not new and does not count
    against the hub's quota. Returns the number of opponents added.
    """
    months = list(months)
    added = failures = read = 0
    waiting = sum(1 for row in opponents.read_all() if "via_cadence" in row and not fetched(row["username"]))

    for hub in hubs:
        if waiting + added >= wanted:
            break
        if hub in done:
            continue
        try:
            by_cadence: dict[str, list[str]] = {}
            for month in hub_months(hub, months):
                by_cadence = opponents_by_cadence(fetch_monthly_archive(client, hub, month), hub)
                if any(by_cadence.values()):
                    break
        except ApiError as exc:
            failures += 1
            logger.warning("chess.com archive of %s failed: %s", hub, exc)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error("too many consecutive failures, stopping the archive stage")
                break
            continue
        failures = 0

        rng = random.Random(f"{SHUFFLE_SEED}:{hub}")
        for cadence in config.TIME_CLASSES:
            new = [name for name in by_cadence.get(cadence, []) if name not in opponents and not fetched(name)]
            for name in rng.sample(new, min(MAX_PER_CADENCE, len(new))):
                if opponents.add({"username": name, "via": hub, "via_cadence": cadence}):
                    added += 1
        done.add_many([hub])
        read += 1
        if read % 50 == 0:
            logger.info("chess.com hubs: %d read, %d opponents added", read, added)
    return added


def pending_opponents(opponents: KeyedJsonl, fetched: Callable[[str], bool]) -> list[dict[str, str]]:
    """Opponents picked by the per-hub rules and not yet fetched, in a fixed shuffled order.

    Rows stored by the earlier, uncapped harvest carry no `via_cadence` and are
    left alone.
    """
    rows = [
        {"username": row["username"], "via": row["via"], "via_cadence": row["via_cadence"]}
        for row in opponents.read_all()
        if "via_cadence" in row and not fetched(row["username"])
    ]
    rows.sort(key=lambda row: row["username"])
    random.Random(SHUFFLE_SEED).shuffle(rows)
    return rows
