"""The raw material of the 2026 conversion: one person, several ratings.

A conversion between two rating pools needs the *same person* measured in both.
Three sources provide that, and this module turns each into one tidy frame with
one row per person and one column per pool (plus its rating deviation where the
site publishes one):

* **current-rating surveys** -- chess.com's and Lichess's own profile of each
  surveyed account, giving today's blitz and rapid ratings side by side;
* **same-month snapshots** -- a player's end-of-month rating in both cadences,
  from a month of 2026 in which they played enough of each;
* **linked accounts** -- Lichess profiles that declare a chess.com account,
  the only direct evidence of one person's rating on both sites.

Only *established* ratings are kept, because a rating the site itself is unsure
of measures the rating system's prior, not the player:

* at least `MIN_POOL_GAMES` games in the pool: below that both systems are
  still moving a newcomer's number by large steps;
* a rating deviation of at most `MAX_ESTABLISHED_RD`. That is Lichess's own
  threshold for calling a rating provisional; both sites run Glicko, whose RD is
  in rating points, so the same bar is applied to chess.com;
* recent play: chess.com's last game in the pool within `RECENT_DAYS` of the
  survey, and a Lichess account seen during the conversion year. An old rating
  describes the pool as it was when the player left it, not the 2026 pool;
* closed Lichess accounts and those marked for terms-of-service violations
  (mostly engine use) are dropped: their ratings do not describe human play.

chess.com accounts reach the survey as opponents of a *hub* account whose games
were read. Opponents are matched to the hub in one cadence, so a hub that plays
mostly rapid at 2200 hands over people chosen for their rapid rating -- and a
sample selected on one rating drags any symmetric conversion toward it (the
other rating regresses to the mean). No hub may therefore contribute more than
`MAX_PER_HUB` people to the survey fit.

*Which cadence* a person was met in matters as much. Players found through a
hub's rapid games are rapid regulars, and their rapid rating runs far ahead of
their blitz (about -290 points on average, against -90 for those met in blitz
games). A conversion is only neutral between the two pools if both kinds of
player weigh the same, so where the survey records the cadence of discovery,
the fit uses those people alone, in equal numbers from each cadence.

Files that are missing or half-written (the survey may still be running) are
read as far as they go; a missing file is an empty source, never an error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from chess_elo_drift import config

logger = logging.getLogger(__name__)

MIN_POOL_GAMES = 20
MAX_ESTABLISHED_RD = 110
RECENT_DAYS = 183
#: Games of *each* cadence a player must have finished in a snapshot month for
#: both month-end ratings to be current. Snapshots carry no RD, so this is the
#: only establishment check available for them.
SNAPSHOT_MIN_GAMES = 5

#: People one hub account may contribute to the chess.com survey fit.
MAX_PER_HUB = 12

CHESSCOM_STATS_FILE = "chesscom_stats.jsonl"
CHESSCOM_OPPONENTS_FILE = "chesscom_opponents.jsonl"
LICHESS_USERS_FILE = "lichess_users.jsonl"
LINKED_ACCOUNTS_FILE = "linked_accounts.jsonl"


@dataclass(frozen=True)
class Pool:
    """One rating pool: a site and a cadence."""

    platform: str
    time_class: str

    @property
    def key(self) -> str:
        return f"{self.platform}_{self.time_class}"

    @property
    def label(self) -> str:
        return f"{config.PLATFORM_LABELS.get(self.platform, self.platform)} {self.time_class}"


POOLS: tuple[Pool, ...] = (
    Pool("chesscom", "rapid"),
    Pool("chesscom", "blitz"),
    Pool("lichess", "rapid"),
    Pool("lichess", "blitz"),
)


@dataclass
class SourceLog:
    """What a source contained and what survived the filters, for the findings."""

    name: str
    available: bool
    records: int = 0
    notes: list[str] = field(default_factory=list)
    kept: dict[str, int] = field(default_factory=dict)


# -- reading ----------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every parseable line of a JSON-lines file; nothing if it does not exist.

    A survey that is still being written can end in a partial line, which is
    skipped rather than allowed to sink the whole report.
    """
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def timestamp(value: Any) -> pd.Timestamp | None:
    """Epoch seconds, epoch milliseconds or ISO text, as a UTC timestamp."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        if isinstance(value, (int, float)):
            unit = "ms" if value > 1e11 else "s"
            return pd.Timestamp(value, unit=unit, tz="UTC")
        parsed = pd.Timestamp(str(value))
        return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    except (ValueError, TypeError, OverflowError):
        return None


def survey_reference(records: Iterable[dict[str, Any]]) -> pd.Timestamp:
    """The moment "recent" is measured from: the latest fetch, or the end of the study."""
    stamps = [stamp for stamp in (timestamp(r.get("fetched_at")) for r in records) if stamp is not None]
    if stamps:
        return max(stamps)
    return pd.Timestamp(f"{config.CONVERSION_YEAR}-09-30", tz="UTC")


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


# -- establishment rules ----------------------------------------------------


def established_chesscom(pool: Any, fetched: pd.Timestamp) -> tuple[float, float] | None:
    """(rating, rd) of a chess.com pool that passes every rule, else None."""
    if not isinstance(pool, dict):
        return None
    rating, rd, games = _number(pool.get("rating")), _number(pool.get("rd")), _number(pool.get("games"))
    if not np.isfinite(rating) or not (games >= MIN_POOL_GAMES):
        return None
    if np.isfinite(rd) and rd > MAX_ESTABLISHED_RD:
        return None
    last = timestamp(pool.get("last_date"))
    if last is not None and (fetched - last).days > RECENT_DAYS:
        return None
    return rating, rd


def established_lichess(pool: Any) -> tuple[float, float] | None:
    """(rating, rd) of a Lichess pool that passes every rule, else None."""
    if not isinstance(pool, dict) or pool.get("provisional"):
        return None
    rating, rd, games = _number(pool.get("rating")), _number(pool.get("rd")), _number(pool.get("games"))
    if not np.isfinite(rating) or not (games >= MIN_POOL_GAMES):
        return None
    if np.isfinite(rd) and rd > MAX_ESTABLISHED_RD:
        return None
    return rating, rd


def lichess_account_usable(record: dict[str, Any]) -> bool:
    """Open, in good standing, and seen during the conversion year."""
    if record.get("disabled") or record.get("tos_violation"):
        return False
    seen = timestamp(record.get("seen_at"))
    return seen is None or seen.year >= config.CONVERSION_YEAR


# -- frames -----------------------------------------------------------------


def _empty_people() -> pd.DataFrame:
    columns = ["person"] + [c for pool in POOLS for c in (pool.key, f"{pool.key}_rd")]
    return pd.DataFrame(columns=columns)


def _people(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows) if rows else _empty_people()
    for column in _empty_people().columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[list(_empty_people().columns)].drop_duplicates("person", keep="last").reset_index(drop=True)


def chesscom_survey(records: list[dict[str, Any]], *, available: bool) -> tuple[pd.DataFrame, SourceLog]:
    log = SourceLog("chess.com current ratings", available, len(records))
    reference = survey_reference(records)
    rows = []
    for record in records:
        if not record.get("found", True) or not record.get("username"):
            continue
        fetched = timestamp(record.get("fetched_at")) or reference
        row: dict[str, Any] = {"person": f"chesscom:{str(record['username']).lower()}"}
        for time_class in config.TIME_CLASSES:
            kept = established_chesscom(record.get(time_class), fetched)
            if kept is not None:
                row[f"chesscom_{time_class}"], row[f"chesscom_{time_class}_rd"] = kept
        rows.append(row)
    people = _people(rows)
    log.kept = _pool_counts(people, ("chesscom_rapid", "chesscom_blitz"))
    log.notes.append(f"{len(rows)} accounts found")
    return people, log


def cap_per_hub(
    people: pd.DataFrame, hubs: dict[str, str], max_per_hub: int = MAX_PER_HUB
) -> tuple[pd.DataFrame, int]:
    """Keep at most `max_per_hub` people per hub account, chosen reproducibly.

    A person with no known hub (found some other way) is their own hub. The
    choice within a hub is by a hash of the name, so it depends on neither file
    order nor the ratings themselves.
    """
    if people.empty:
        return people, 0
    names = people["person"].str.split(":", n=1).str[1]
    hub = names.map(lambda name: hubs.get(name) or f"self:{name}")
    order = pd.util.hash_pandas_object(names, index=False)
    ranked = (
        people.assign(_hub=hub.to_numpy(), _order=order.to_numpy())
        .sort_values(["_hub", "_order"], kind="stable")
    )
    kept = ranked[ranked.groupby("_hub").cumcount() < max_per_hub]
    kept = kept.drop(columns=["_hub", "_order"]).sort_index()
    return kept, len(people) - len(kept)


def balance_by_cadence(
    people: pd.DataFrame, cadences: dict[str, str]
) -> tuple[pd.DataFrame, str | None]:
    """Equal numbers of people discovered through blitz and through rapid games.

    Returns the frame unchanged (and no note) when no discovery cadence is
    recorded. Otherwise only people with a recorded cadence are kept, and the
    larger group is cut to the size of the smaller one, chosen by name hash.
    """
    if people.empty or not cadences:
        return people, None
    names = people["person"].str.split(":", n=1).str[1]
    cadence = names.map(cadences)
    tagged = people.assign(_cadence=cadence.to_numpy(), _order=pd.util.hash_pandas_object(names, index=False).to_numpy())
    tagged = tagged[tagged["_cadence"].isin(config.TIME_CLASSES)]
    sizes = tagged["_cadence"].value_counts()
    if len(sizes) < len(config.TIME_CLASSES):
        return people, None
    per_cadence = int(sizes.min())
    kept = (
        tagged.sort_values(["_cadence", "_order"], kind="stable")
        .groupby("_cadence").head(per_cadence)
        .drop(columns=["_cadence", "_order"]).sort_index()
    )
    note = (
        f"fit on {len(kept)} people with a recorded discovery cadence, "
        f"{per_cadence} met in each of blitz and rapid games ({len(people) - len(kept)} others set aside)"
    )
    return kept, note


def cadences_of(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Username -> the cadence of the hub game that revealed it, where recorded."""
    found: dict[str, str] = {}
    for record in records:
        name, cadence = record.get("username"), record.get("via_cadence")
        if name and cadence:
            found.setdefault(str(name).lower(), str(cadence))
    return found


def hubs_of(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Username -> the hub whose games revealed it, first sighting wins."""
    hubs: dict[str, str] = {}
    for record in records:
        name, via = record.get("username"), record.get("via")
        if name and via:
            hubs.setdefault(str(name).lower(), str(via).lower())
    return hubs


def lichess_survey(records: list[dict[str, Any]], *, available: bool) -> tuple[pd.DataFrame, SourceLog]:
    log = SourceLog("Lichess current ratings", available, len(records))
    rows, closed = [], 0
    for record in records:
        if not record.get("username"):
            continue
        if not lichess_account_usable(record):
            closed += 1
            continue
        row: dict[str, Any] = {"person": f"lichess:{str(record['username']).lower()}"}
        for time_class in config.TIME_CLASSES:
            kept = established_lichess(record.get(time_class))
            if kept is not None:
                row[f"lichess_{time_class}"], row[f"lichess_{time_class}_rd"] = kept
        rows.append(row)
    people = _people(rows)
    log.kept = _pool_counts(people, ("lichess_rapid", "lichess_blitz"))
    log.notes.append(f"{closed} closed, flagged or not seen in {config.CONVERSION_YEAR}")
    return people, log


def snapshot_pairs(records: list[dict[str, Any]], platform: str, *, available: bool) -> tuple[pd.DataFrame, SourceLog]:
    """Same-month blitz and rapid ratings from the conversion year, latest month per player."""
    log = SourceLog(f"{config.PLATFORM_LABELS[platform]} same-month snapshots", available, len(records))
    year_prefix = f"{config.CONVERSION_YEAR}-"
    rows = []
    for record in sorted(records, key=lambda r: str(r.get("month", ""))):
        if not str(record.get("month", "")).startswith(year_prefix) or not record.get("username"):
            continue
        if _number(record.get("blitz_games")) < SNAPSHOT_MIN_GAMES or _number(record.get("rapid_games")) < SNAPSHOT_MIN_GAMES:
            continue
        blitz, rapid = _number(record.get("blitz_rating")), _number(record.get("rapid_rating"))
        if np.isfinite(blitz) and np.isfinite(rapid):
            rows.append(
                {
                    "person": f"{platform}:{str(record['username']).lower()}",
                    f"{platform}_blitz": blitz,
                    f"{platform}_rapid": rapid,
                }
            )
    people = _people(rows)
    log.kept = _pool_counts(people, (f"{platform}_rapid", f"{platform}_blitz"))
    return people, log


def linked_people(
    linked: list[dict[str, Any]],
    lichess_records: list[dict[str, Any]],
    chesscom_people: pd.DataFrame,
    lichess_people: pd.DataFrame,
    *,
    available: bool,
) -> tuple[pd.DataFrame, SourceLog]:
    """One row per person who declared both accounts, with all four pools.

    Links come from `linked_accounts.jsonl` and, as a supplement, from Lichess
    survey profiles that carry a chess.com link. A pool's rating is taken from
    the survey files when the account was surveyed there, because those carry
    the RD, game count and recency the establishment rules need; otherwise from
    the link record itself, checked as far as the record allows.
    """
    log = SourceLog("linked accounts", available, len(linked))
    chesscom_index = chesscom_people.set_index("person") if not chesscom_people.empty else None
    lichess_index = lichess_people.set_index("person") if not lichess_people.empty else None
    lichess_by_name = {str(r.get("username", "")).lower(): r for r in lichess_records}
    reference = survey_reference(linked)

    links = _collect_links(linked, lichess_records)
    rows, unchecked, dropped = [], 0, 0
    for lichess_name, chesscom_name, record in links:
        profile = lichess_by_name.get(lichess_name)
        if (profile is not None and not lichess_account_usable(profile)) or record.get("chesscom_found") is False:
            dropped += 1
            continue
        row: dict[str, Any] = {"person": f"linked:{lichess_name}"}
        fetched = timestamp(record.get("fetched_at")) or reference
        for platform, name, index in (
            ("lichess", lichess_name, lichess_index),
            ("chesscom", chesscom_name, chesscom_index),
        ):
            side = record.get(platform) or {}
            for time_class in config.TIME_CLASSES:
                key = f"{platform}_{time_class}"
                value, verified = _linked_rating(platform, name, time_class, side, index, fetched, record)
                if value is not None:
                    row[key], row[f"{key}_rd"] = value
                    unchecked += not verified
        rows.append(row)

    people = _people(rows)
    log.kept = _pool_counts(people, tuple(pool.key for pool in POOLS))
    log.notes.append(f"{len(links)} links, {dropped} dropped (account closed, flagged or not found)")
    if unchecked:
        log.notes.append(
            f"{unchecked} ratings came from the link record alone and could not be checked "
            "for games played or rating deviation"
        )
    return people, log


def _collect_links(
    linked: list[dict[str, Any]], lichess_records: list[dict[str, Any]]
) -> list[tuple[str, str, dict[str, Any]]]:
    links: dict[str, tuple[str, str, dict[str, Any]]] = {}
    for record in lichess_records:
        chesscom_name = _chesscom_name(record.get("chesscom_link"))
        if chesscom_name and record.get("username"):
            name = str(record["username"]).lower()
            links[name] = (name, chesscom_name, {"fetched_at": record.get("fetched_at")})
    # The dedicated link file wins: it was fetched for this purpose.
    for record in linked:
        lichess_name = str(record.get("lichess_username") or "").lower()
        chesscom_name = _chesscom_name(record.get("chesscom_username"))
        if lichess_name and chesscom_name:
            links[lichess_name] = (lichess_name, chesscom_name, record)
    return list(links.values())


def _chesscom_name(link: Any) -> str | None:
    """A chess.com username from a bare name or a profile URL."""
    if not link or not isinstance(link, str):
        return None
    name = link.strip().rstrip("/").split("/")[-1].split("?")[0]
    return name.lower() or None


def _linked_rating(
    platform: str,
    name: str,
    time_class: str,
    side: dict[str, Any],
    surveyed: pd.DataFrame | None,
    fetched: pd.Timestamp,
    record: dict[str, Any],
) -> tuple[tuple[float, float] | None, bool]:
    """A linked pool's (rating, rd), and whether it could be fully checked."""
    key = f"{platform}_{time_class}"
    person = f"{platform}:{name}"
    if surveyed is not None and person in surveyed.index:
        rating = surveyed.at[person, key]
        if np.isfinite(rating):
            return (float(rating), float(surveyed.at[person, f"{key}_rd"])), True
        return None, True

    pool = side.get(time_class) if isinstance(side, dict) else None
    if isinstance(pool, dict):
        kept = established_chesscom(pool, fetched) if platform == "chesscom" else established_lichess(pool)
        return kept, True
    rating = _number(pool)
    if not np.isfinite(rating):
        return None, True
    if platform == "lichess":
        seen = timestamp(side.get("seen_at")) if isinstance(side, dict) else None
        if seen is not None and seen.year < config.CONVERSION_YEAR:
            return None, True
    return (rating, float("nan")), False


def _pool_counts(people: pd.DataFrame, keys: tuple[str, ...]) -> dict[str, int]:
    counts = {key: int(people[key].notna().sum()) for key in keys}
    counts["all of these"] = int(people[list(keys)].notna().all(axis=1).sum()) if len(people) else 0
    return counts


# -- entry point ------------------------------------------------------------


@dataclass
class ConversionInputs:
    """Everything the conversion study reads, as per-person frames."""

    chesscom_survey: pd.DataFrame
    lichess_survey: pd.DataFrame
    chesscom_snapshots: pd.DataFrame
    lichess_snapshots: pd.DataFrame
    linked: pd.DataFrame
    logs: list[SourceLog]


def load_conversion_inputs(data_raw: Path | None = None) -> ConversionInputs:
    """Read every conversion source under `data_raw` (default: the project's)."""
    data_raw = data_raw or config.DATA_RAW
    conversion_dir = data_raw / "conversion"

    def source(path: Path) -> tuple[list[dict[str, Any]], bool]:
        return read_jsonl(path), path.exists()

    chesscom_records, chesscom_available = source(conversion_dir / CHESSCOM_STATS_FILE)
    lichess_records, lichess_available = source(conversion_dir / LICHESS_USERS_FILE)
    linked_records, linked_available = source(conversion_dir / LINKED_ACCOUNTS_FILE)

    chesscom_people, chesscom_log = chesscom_survey(chesscom_records, available=chesscom_available)
    lichess_people, lichess_log = lichess_survey(lichess_records, available=lichess_available)
    # Linked accounts are enriched from every surveyed account; only the survey's
    # own rapid/blitz fit is limited per hub.
    linked, linked_log = linked_people(
        linked_records, lichess_records, chesscom_people, lichess_people, available=linked_available
    )
    hub_records = list(chesscom_records) + read_jsonl(conversion_dir / CHESSCOM_OPPONENTS_FILE)
    chesscom_people, capped = cap_per_hub(chesscom_people, hubs_of(hub_records))
    if capped:
        chesscom_log.notes.append(f"{capped} dropped to keep at most {MAX_PER_HUB} per hub account")
    established_both = chesscom_people.dropna(subset=["chesscom_rapid", "chesscom_blitz"])
    balanced, note = balance_by_cadence(established_both, cadences_of(hub_records))
    if note:
        chesscom_people = pd.concat(
            [balanced, chesscom_people[~chesscom_people["person"].isin(established_both["person"])]]
        )
        chesscom_log.notes.append(note)
    chesscom_log.kept = _pool_counts(chesscom_people, ("chesscom_rapid", "chesscom_blitz"))

    snapshots: dict[str, tuple[pd.DataFrame, SourceLog]] = {}
    for platform in config.PLATFORMS:
        path = data_raw / platform / "rating_snapshots.jsonl"
        records, available = source(path)
        snapshots[platform] = snapshot_pairs(records, platform, available=available)

    return ConversionInputs(
        chesscom_survey=chesscom_people,
        lichess_survey=lichess_people,
        chesscom_snapshots=snapshots["chesscom"][0],
        lichess_snapshots=snapshots["lichess"][0],
        linked=linked,
        logs=[chesscom_log, lichess_log, snapshots["chesscom"][1], snapshots["lichess"][1], linked_log],
    )
