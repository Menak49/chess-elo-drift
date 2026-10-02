"""Collecting the paired ratings the 2026 conversion formulas are fitted on.

Rating systems are only comparable if the same ability is measured in both. The
survey gathers three kinds of pair, each answering a different question and each
with its own bias:

1. chess.com rapid against chess.com blitz, same player (`chesscom_stats.jsonl`).
   Sampled from accounts seen in the 2026 crawl, so it inherits the crawl's
   snowball bias toward players who meet many opponents.
2. Lichess rapid against Lichess blitz, same player (`lichess_users.jsonl`).
   Harvested from tournaments, leaderboards and team lists (see `harvest`), which
   over-represent regulars and tournament players.
3. chess.com against Lichess, same person (`linked_accounts.jsonl`). The only
   direct cross-site evidence, and the weakest sample: the pairs are people who
   chose to link their accounts on a Lichess profile, so they are self-declared,
   self-selected (more engaged, more likely to play both sites seriously) and not
   verified from the chess.com side.

Two limits apply to all three. A rating read from the API is the player's latest,
not one taken at a common instant, so the two halves of a pair can be weeks
apart. And a pool the player has stopped using keeps its last rating with an
ever larger uncertainty. The collector therefore stores what is needed to judge
this (chess.com `last_date`, `rd` and `games`; Lichess `rd`, `games`,
`provisional` and `seen_at`) and filters nothing: who counts as established is
decided once, in the analysis.

Every file is append-only JSONL and resumable: a re-run skips the usernames it
already holds. The caps count the work done by this run.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from chess_elo_drift import config
from chess_elo_drift.chesscom.client import ChessComClient
from chess_elo_drift.config import YearMonth
from chess_elo_drift.conversion import archives
from chess_elo_drift.conversion.harvest import LichessHarvester
from chess_elo_drift.conversion.jsonl import IdList, KeyedJsonl
from chess_elo_drift.conversion.links import chesscom_link
from chess_elo_drift.conversion.pools import chesscom_pools, lichess_pools
from chess_elo_drift.http import ApiError, NotFoundError
from chess_elo_drift.lichess.client import LichessClient

logger = logging.getLogger("survey")

CHESSCOM_STATS_FILE = "chesscom_stats.jsonl"
LICHESS_USERS_FILE = "lichess_users.jsonl"
LINKED_ACCOUNTS_FILE = "linked_accounts.jsonl"
LICHESS_MISSING_FILE = "lichess_missing.txt"

#: The bulk endpoint silently ignores every id past the 300th.
LICHESS_BATCH_SIZE = 300

#: Lichess asks for one request at a time and a calm pace.
LICHESS_MIN_INTERVAL = 1.0

#: `POST /api/users` allows 8000 accounts per 10 minutes and 120000 per day.
#: One full batch every 25 seconds stays under the first limit (7200 per 10
#: minutes) and keeps a day's run well inside the second.
LICHESS_BULK_INTERVAL = 25.0

#: A bulk lookup that still fails after the client's own retries is tried again
#: this many times in all, each after a pause: a 429 here means "wait", not "stop".
BULK_ATTEMPTS = 3

#: Fixed so that a re-run sees the same candidate order.
SHUFFLE_SEED = 20262026

#: A run that fails this many requests in a row has lost the network; stop
#: instead of burning the budget on retries.
MAX_CONSECUTIVE_FAILURES = 10

LOG_EVERY = 300

#: Earliest `seenAt` (ms) that counts as active in the conversion year.
_YEAR_START_MS = int(datetime(config.CONVERSION_YEAR, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- chess.com ---------------------------------------------------------------


def chesscom_row(
    client: ChessComClient,
    username: str,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch one player's stats. Costs one request; a 404 is a row, not an error.

    `source` says how the username was found, and `extra` adds fields such as the
    hub and cadence an archive opponent came from. Both are stored only when given,
    so rows written before they existed stay valid.
    """
    fetched_at = now_utc()
    try:
        stats = client.get(f"player/{username}/stats")
    except NotFoundError:
        row = {"username": username, "fetched_at": fetched_at, "found": False,
               "blitz": None, "rapid": None}
    else:
        pools = chesscom_pools(stats)
        row = {"username": username, "fetched_at": fetched_at, "found": True,
               "blitz": pools["blitz"], "rapid": pools["rapid"]}
    if source is not None:
        row["source"] = source
    if extra:
        row.update(extra)
    return row


def chesscom_candidates(games_file: Path, snapshots_file: Path, year: int) -> list[str]:
    """Usernames seen in `year` by the chess.com crawl, in a fixed shuffled order."""
    names: set[str] = set()
    for row in _read_jsonl(games_file):
        for side in ("white_username", "black_username"):
            if row.get(side):
                names.add(row[side].lower())
    for row in _read_jsonl(snapshots_file):
        if str(row.get("month", "")).startswith(str(year)) and row.get("username"):
            names.add(row["username"].lower())
    ordered = sorted(names)
    random.Random(SHUFFLE_SEED).shuffle(ordered)
    return ordered


def survey_chesscom(
    client: ChessComClient,
    store: KeyedJsonl,
    candidates: Iterable[str | tuple],
    max_requests: int,
) -> int:
    """Fetch stats for unseen candidates until `max_requests` are spent. Returns rows added.

    A candidate is a username, a `(username, source)` pair or a
    `(username, source, extra_fields)` triple.
    """
    added = failures = 0
    for candidate in candidates:
        username, source, extra = _split_candidate(candidate)
        if added >= max_requests:
            break
        if username in store:
            continue
        try:
            row = chesscom_row(client, username, source, extra)
        except ApiError as exc:
            failures += 1
            logger.warning("chess.com %s failed: %s", username, exc)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error("too many consecutive failures, stopping the chess.com stage")
                break
            continue
        failures = 0
        store.add(row)
        added += 1
        if added % LOG_EVERY == 0:
            logger.info("chess.com stats: %d fetched this run (%d stored)", added, len(store))
    return added


def _split_candidate(candidate: str | tuple) -> tuple[str, str | None, dict | None]:
    if isinstance(candidate, str):
        return candidate, None, None
    username, source, *rest = candidate
    return username, source, (rest[0] if rest else None)


# -- Lichess -----------------------------------------------------------------


def lichess_row(user: dict[str, Any], source: str) -> dict[str, Any]:
    """One `lichess_users.jsonl` row from a user object of the bulk endpoint."""
    pools = lichess_pools(user)
    return {
        "username": str(user["id"]).lower(),
        "fetched_at": now_utc(),
        "created_at": int(user.get("createdAt") or 0),
        "seen_at": user.get("seenAt"),
        "title": user.get("title"),
        "disabled": bool(user.get("disabled", False)),
        "tos_violation": bool(user.get("tosViolation", False)),
        "blitz": pools["blitz"],
        "rapid": pools["rapid"],
        "chesscom_link": chesscom_link(user.get("profile")),
        "source": source,
    }


def survey_lichess(
    client: Any,
    store: KeyedJsonl,
    missing: IdList,
    harvester: LichessHarvester,
    max_users: int,
    *,
    bulk_interval: float = LICHESS_BULK_INTERVAL,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Look up harvested usernames in batches of 300. Returns rows added.

    Ids the endpoint does not return are closed accounts; they are remembered so
    that a re-run does not ask for them again. Calls are spaced `bulk_interval`
    seconds apart, counted from the end of the previous one.
    """
    added = requested = 0
    last_call = None
    exhausted = False
    candidates = harvester.candidates()

    while requested < max_users and not exhausted:
        batch: list[tuple[str, str]] = []
        room = min(LICHESS_BATCH_SIZE, max_users - requested)
        while len(batch) < room:
            try:
                username, source = next(candidates)
            except StopIteration:
                exhausted = True
                break
            if username not in store and username not in missing:
                batch.append((username, source))
        if not batch:
            break

        sources = dict(batch)
        if last_call is not None:
            sleep(max(0.0, bulk_interval - (clock() - last_call)))
        try:
            users = _bulk_lookup(client, list(sources), sleep)
        except ApiError as exc:
            logger.error("bulk lookup failed, stopping the Lichess stage: %s", exc)
            break
        last_call = clock()
        before = requested
        requested += len(batch)

        returned: set[str] = set()
        for user in users:
            key = str(user.get("id", "")).lower()
            if key in sources and store.add(lichess_row(user, sources[key])):
                returned.add(key)
                added += 1
        missing.add_many([name for name in sources if name not in returned and name not in store])

        if requested // LOG_EVERY != before // LOG_EVERY:
            logger.info("lichess: %d looked up this run, %d stored, %d closed",
                        requested, len(store), len(missing))
    return added


def _bulk_lookup(client: Any, ids: list[str], sleep: Callable[[float], None]) -> list[dict]:
    for attempt in range(BULK_ATTEMPTS):
        try:
            return client.post_text("api/users", ",".join(ids))
        except ApiError as exc:
            if attempt == BULK_ATTEMPTS - 1:
                raise
            logger.warning("bulk lookup failed (%s); pausing before trying again", exc)
            sleep(config.LICHESS_RATE_LIMIT_PAUSE_SECONDS)
    raise ApiError("unreachable")  # pragma: no cover


# -- linked accounts ---------------------------------------------------------


def survey_linked(
    chesscom: ChessComClient,
    lichess_store: KeyedJsonl,
    chesscom_store: KeyedJsonl,
    linked: KeyedJsonl,
) -> int:
    """Fetch the chess.com side of every Lichess account that declares one.

    Only Lichess accounts seen in the conversion year are followed; an account
    dormant since 2025 cannot give a 2026 pair. A chess.com name already fetched
    by the chess.com stage is reused rather than requested again. Returns rows added.
    """
    known = {row["username"]: row for row in chesscom_store.read_all()}
    added = failures = 0

    for user in lichess_store.read_all():
        name = user.get("chesscom_link")
        if not name or user["username"] in linked or user.get("disabled"):
            continue
        if (user.get("seen_at") or 0) < _YEAR_START_MS:
            continue

        stats = known.get(name)
        if stats is None:
            try:
                stats = chesscom_row(chesscom, name)
            except ApiError as exc:
                failures += 1
                logger.warning("linked chess.com %s failed: %s", name, exc)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("too many consecutive failures, stopping the linked stage")
                    break
                continue
            failures = 0
            known[name] = stats

        linked.add({
            "lichess_username": user["username"],
            "chesscom_username": name,
            "fetched_at": now_utc(),
            "chesscom_found": stats["found"],
            "lichess": {"blitz": user["blitz"], "rapid": user["rapid"], "seen_at": user["seen_at"]},
            "chesscom": {"blitz": stats["blitz"], "rapid": stats["rapid"]},
        })
        added += 1
        if added % LOG_EVERY == 0:
            logger.info("linked accounts: %d written this run", added)
    return added


# -- entry point -------------------------------------------------------------


def collect_survey(
    platforms: tuple[str, ...] = config.PLATFORMS,
    max_chesscom_requests: int = 6000,
    max_lichess_users: int = 60000,
    min_interval: float = 0.0,
    *,
    out_dir: Path | None = None,
    chesscom_client: ChessComClient | None = None,
    lichess_client: Any | None = None,
    bulk_interval: float = LICHESS_BULK_INTERVAL,
    read_archives: bool = True,
) -> dict[str, Any]:
    """Run the survey and return a summary of what the files now hold.

    `max_chesscom_requests` caps the chess.com stats requests of this run and
    `max_lichess_users` the Lichess accounts looked up in this run. The linked
    accounts stage is not capped: it is bounded by the links the Lichess stage
    found. It runs only when both platforms are selected.
    """
    out_dir = Path(out_dir) if out_dir is not None else config.CONVERSION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    chesscom_store = KeyedJsonl(out_dir / CHESSCOM_STATS_FILE)
    lichess_store = KeyedJsonl(out_dir / LICHESS_USERS_FILE)
    linked_store = KeyedJsonl(out_dir / LINKED_ACCOUNTS_FILE, key="lichess_username")
    missing = IdList(out_dir / LICHESS_MISSING_FILE)

    own_chesscom = own_lichess = False
    if chesscom_client is None and "chesscom" in platforms:
        chesscom_client = ChessComClient(min_interval_seconds=min_interval)
        own_chesscom = True
    if lichess_client is None and "lichess" in platforms:
        lichess_client = LichessClient(
            min_interval_seconds=max(min_interval, LICHESS_MIN_INTERVAL)
        )
        own_lichess = True

    summary: dict[str, Any] = {}
    try:
        if "lichess" in platforms:
            harvester = LichessHarvester(
                lichess_client,
                games_path=config.games_path("lichess", config.CONVERSION_YEAR),
                snapshots_path=config.snapshots_path("lichess"),
                seed_pool_path=config.raw_dir("lichess") / "seed_pool.json",
            )
            summary["lichess_added"] = survey_lichess(
                lichess_client, lichess_store, missing, harvester, max_lichess_users,
                bulk_interval=bulk_interval,
            )

        if "chesscom" in platforms:
            candidates = chesscom_candidates(
                config.games_path("chesscom", config.CONVERSION_YEAR),
                config.snapshots_path("chesscom"),
                config.CONVERSION_YEAR,
            )
            summary["chesscom_candidates"] = len(candidates)
            waiting = [(name, "crawl") for name in candidates]
            if read_archives:
                waiting += _archive_candidates(
                    chesscom_client, out_dir, chesscom_store, candidates, max_chesscom_requests
                )
            summary["chesscom_candidates_with_archives"] = len(waiting)
            summary["chesscom_added"] = survey_chesscom(
                chesscom_client, chesscom_store, waiting, max_chesscom_requests
            )

        if "chesscom" in platforms and "lichess" in platforms:
            summary["linked_added"] = survey_linked(
                chesscom_client, lichess_store, chesscom_store, linked_store
            )
    finally:
        if own_chesscom:
            chesscom_client.close()
        if own_lichess:
            lichess_client.close()

    summary.update(_totals(out_dir))
    if chesscom_client is not None:
        summary["chesscom_requests"] = chesscom_client.request_count
    if lichess_client is not None:
        summary["lichess_requests"] = lichess_client.request_count
    return summary


def _archive_candidates(
    client: ChessComClient,
    out_dir: Path,
    chesscom_store: KeyedJsonl,
    crawl_candidates: list[str],
    wanted: int,
) -> list[tuple[str, str, dict[str, str]]]:
    """Opponents from one 2026 month of each hub's archive, ready to fetch.

    The hubs are the crawl's accounts (see `archives`). Hubs are read only until
    `wanted` new accounts are waiting in all, so a run with a small cap reads few.
    """
    opponents = KeyedJsonl(out_dir / archives.OPPONENTS_FILE)
    done = IdList(out_dir / archives.HUBS_DONE_FILE)
    fetched = chesscom_store.__contains__
    already = sum(1 for name in crawl_candidates if not fetched(name))
    hubs = archives.read_hubs(
        config.games_path("chesscom", config.CONVERSION_YEAR),
        config.visited_path("chesscom", config.CONVERSION_YEAR),
    )
    months = [YearMonth(config.CONVERSION_YEAR, month) for month in config.SAMPLED_MONTHS]
    archives.collect_opponents(
        client, hubs, opponents, done, fetched, months, max(0, wanted - already)
    )
    return [
        (row["username"], archives.ARCHIVE_SOURCE,
         {"via": row["via"], "via_cadence": row["via_cadence"]})
        for row in archives.pending_opponents(opponents, fetched)
    ]


def _totals(out_dir: Path) -> dict[str, Any]:
    """Row counts and the link rate of the files as they stand."""
    chesscom_rows = sum(1 for _ in KeyedJsonl(out_dir / CHESSCOM_STATS_FILE).read_all())
    users = list(KeyedJsonl(out_dir / LICHESS_USERS_FILE).read_all())
    linked = list(KeyedJsonl(out_dir / LINKED_ACCOUNTS_FILE, key="lichess_username").read_all())
    live = [user for user in users if not user["disabled"]]
    with_link = [user for user in live if user["chesscom_link"]]
    return {
        "chesscom_stats_rows": chesscom_rows,
        "lichess_users_rows": len(users),
        "lichess_with_chesscom_link": len(with_link),
        "link_rate": round(len(with_link) / len(live), 4) if live else 0.0,
        "linked_accounts_rows": len(linked),
        "linked_chesscom_found": sum(1 for row in linked if row["chesscom_found"]),
    }


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
