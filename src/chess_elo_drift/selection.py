"""Choosing which collected games to spend engine time on.

Engine time is the scarce resource, so when the corpus is larger than the budget
the subsample must not be drawn uniformly: that would spend most of the budget
on the crowded rating bands and leave the tails too thin to say anything about.
Games are therefore taken round-robin across the cells of the design, which
keeps every band populated as the budget shrinks.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable, Sequence

from chess_elo_drift.collection.sampler import band_floor
from chess_elo_drift.records import GameRecord


def balanced_subsample(
    records: Iterable[GameRecord], max_games: int, *, seed: int = 20182025
) -> list[GameRecord]:
    """Return at most `max_games` records, spread evenly across the design."""
    pool = list(records)
    if max_games <= 0 or len(pool) <= max_games:
        return pool

    rng = random.Random(seed)
    by_cell: dict[tuple[str, int, str, int], list[GameRecord]] = defaultdict(list)
    for record in pool:
        by_cell[_cell_of(record)].append(record)
    for bucket in by_cell.values():
        rng.shuffle(bucket)

    cells = sorted(by_cell)
    chosen: list[GameRecord] = []
    depth = 0
    while len(chosen) < max_games:
        added = False
        for cell in cells:
            bucket = by_cell[cell]
            if depth < len(bucket):
                chosen.append(bucket[depth])
                added = True
                if len(chosen) == max_games:
                    break
        if not added:
            break
        depth += 1
    return chosen


def _cell_of(record: GameRecord) -> tuple[str, int, str, int]:
    """The design cell a game belongs to, keyed on its lower-rated side.

    A game is one observation per player and the two ratings are close by
    construction, so the lower one is a stable, deterministic label.
    """
    ratings = [record.white_rating, record.black_rating]
    band = band_floor(min(ratings)) or band_floor(max(ratings)) or -1
    return (record.platform, record.year, record.time_class, band)


def summarise_cells(records: Sequence[GameRecord]) -> dict[tuple[str, int, str, int], int]:
    """Count games per design cell, for logging what a selection looks like."""
    counts: dict[tuple[str, int, str, int], int] = defaultdict(int)
    for record in records:
        counts[_cell_of(record)] += 1
    return dict(counts)
