"""Append-only JSONL corpus of collected games.

Collection is slow and interruptible, so the store is written incrementally and
is safe to resume: re-running a crawl tops up the same file instead of starting
over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from chess_elo_drift.records import GameRecord


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
