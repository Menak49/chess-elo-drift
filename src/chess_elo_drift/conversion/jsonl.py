"""A small append-only JSONL file, keyed by one field.

The survey runs for an hour and is expected to be interrupted, so every row is
flushed as soon as it is written and a re-run skips the keys already present.
A line left truncated by a crash is ignored when the file is read back.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class KeyedJsonl:
    def __init__(self, path: Path, key: str = "username") -> None:
        self.path = path
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: set[str] = {row[key] for row in self.read_all() if key in row}
        self._end_truncated_line()

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def add(self, row: dict) -> bool:
        """Append `row` unless its key is already stored. Returns whether it was new."""
        key = row[self.key]
        if key in self._keys:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._keys.add(key)
        return True

    def _end_truncated_line(self) -> None:
        """Close a line a crash left unfinished so the next row starts on its own."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb") as handle:
            handle.seek(-1, 2)
            unfinished = handle.read(1) != b"\n"
        if unfinished:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

    def read_all(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


class IdList:
    """A plain-text set of identifiers, one per line, appended to as it grows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids: set[str] = set()
        if path.exists():
            self._ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def __contains__(self, item: str) -> bool:
        return item in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def add_many(self, items: list[str]) -> None:
        new = [item for item in items if item not in self._ids]
        if not new:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("".join(f"{item}\n" for item in new))
        self._ids.update(new)
