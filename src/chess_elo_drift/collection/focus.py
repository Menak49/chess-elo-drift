"""The single-clock corpus: only 3+0 blitz and 15+10 rapid games.

Part one of the study compares years on one clock per cadence, so nothing has to
be corrected for the mix of time controls. The sites cannot filter an export by
clock, so a `FocusedSource` reads what the wrapped source returns and keeps the
games on those two clocks. Within the kept games a cadence has exactly one clock,
so the crawler's blitz and rapid quotas become the 3+0 and 15+10 quotas without
the crawler knowing.
"""

from __future__ import annotations

from chess_elo_drift import config
from chess_elo_drift.collection.sources import GameSource, MonthHarvest
from chess_elo_drift.config import Era, YearMonth
from chess_elo_drift.records import GameRecord


def is_focus_game(record: GameRecord) -> bool:
    return record.time_control in config.FOCUS_TIME_CONTROLS.get(record.time_class, frozenset())


class FocusedSource:
    """Wraps a `GameSource` and drops every game outside the focus clocks."""

    def __init__(self, source: GameSource) -> None:
        self._source = source
        self.platform = source.platform
        self.index_is_free = getattr(source, "index_is_free", False)

    def active_months(self, username: str, era: Era) -> list[YearMonth]:
        return self._source.active_months(username, era)

    def month_games(self, username: str, month: YearMonth) -> MonthHarvest:
        harvest = self._source.month_games(username, month)
        return MonthHarvest([game for game in harvest.games if is_focus_game(game)], None)


def import_existing_games(platform: str, year: int, store) -> int:
    """Copy the main corpus's 3+0 and 15+10 games of one site-year into the focus store.

    The main crawl met some of these clocks by chance; reusing them (and, at the
    evaluation step, their engine scores) saves requests without changing a
    single game. Returns how many were new to the focus store.
    """
    from chess_elo_drift.collection.store import GameStore

    main = GameStore(config.games_path(platform, year))
    return sum(1 for record in main.read_all() if is_focus_game(record) and store.add(record))


def import_existing_evaluations(focus_path, main_path, focus_ids: set[str]) -> int:
    """Copy the main corpus's engine rows for the focus games into the focus CSV."""
    import csv

    if not main_path.exists():
        return 0
    from chess_elo_drift.engine.results import COLUMNS, EvaluationWriter

    writer = EvaluationWriter(focus_path)
    done = writer.completed_game_ids()
    with main_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["game_id"] in focus_ids and row["game_id"] not in done
        ]
    writer.append({column: row.get(column, "") for column in COLUMNS} for row in rows)
    return len(rows)
