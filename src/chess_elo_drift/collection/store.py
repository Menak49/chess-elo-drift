"""Append-only JSONL corpus of collected games.

Collection is slow and interruptible, so the store is written incrementally and
is safe to resume: re-running a crawl tops up the same file instead of starting
over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from chess_elo_drift.records import GameRecord, RatingSnapshot


class GameStore:
    """A JSONL file holding one game per line, unique by `game_id`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = {record.game_id for record in self.read_all()}

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, game_id: str) -> bool:
        return game_id in self._seen

    def add(self, record: GameRecord) -> bool:
        """Append a game unless it is already stored. Returns whether it was new."""
        if record.game_id in self._seen:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._seen.add(record.game_id)
        return True

    def read_all(self) -> Iterator[GameRecord]:
        """Yield every stored game, skipping any line left truncated by a crash."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield GameRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue


class VisitedLog:
    """Accounts whose archives have already been read, persisted across runs.

    Keeping this out of the game store matters: most visited accounts yield
    nothing usable, and without a record of them a resumed crawl would spend its
    whole budget walking the same barren ground again.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._names: set[str] = set()
        if self.path.exists():
            self._names = set(json.loads(self.path.read_text(encoding="utf-8")))

    def __contains__(self, username: str) -> bool:
        return username in self._names

    def __len__(self) -> int:
        return len(self._names)

    def add(self, username: str) -> None:
        self._names.add(username)

    def save(self) -> None:
        self.path.write_text(json.dumps(sorted(self._names)), encoding="utf-8")


class SnapshotStore:
    """Append-only JSONL of `RatingSnapshot`s, unique by (username, month).

    The same account can be read in the same month by two crawls -- as a seed
    and again from the opponent graph -- and must only count once.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[tuple[str, str]] = {
            (snapshot.username, snapshot.month) for snapshot in self.read_all()
        }

    def __len__(self) -> int:
        return len(self._seen)

    def add(self, snapshot: RatingSnapshot) -> bool:
        key = (snapshot.username, snapshot.month)
        if key in self._seen:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + "\n")
        self._seen.add(key)
        return True

    def read_all(self) -> Iterator[RatingSnapshot]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield RatingSnapshot.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
