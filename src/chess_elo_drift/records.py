"""The record types that flow through the pipeline.

One `GameRecord` holds a single game exactly once. A game is an observation of
*two* players at once, so the per-player view used by the analysis is derived on
demand through `sides()` rather than stored twice.

A `RatingSnapshot` is what one player's month said about their ratings in both
cadences at once. It is the raw material of the rapid-to-blitz conversion, and
it is collected from archives the crawler reads anyway.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterator

#: chess.com reports an outcome code per side; these are the drawing ones.
_DRAW_CODES = frozenset(
    {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"}
)

#: `{[%clk 0:02:13.7]}` after every single move; stripped before any counting
#: because the tenths of a second otherwise look exactly like move numbers.
_COMMENT_PATTERN = re.compile(r"\{[^}]*\}")
_MOVE_NUMBER_PATTERN = re.compile(r"\b\d+\.(?:\.\.)?(?=\s)")


def count_plies(pgn: str) -> int:
    """Count half-moves in a chess.com PGN without a full parse.

    chess.com numbers both halves of a move (`12. Nf3` / `12... Nc6`), so
    counting move-number tokens is exact and about two orders of magnitude
    cheaper than parsing -- which matters when a crawl walks through tens of
    thousands of games it will then discard.
    """
    _, separator, movetext = pgn.partition("\n\n")
    if not separator:
        movetext = pgn
    return len(_MOVE_NUMBER_PATTERN.findall(_COMMENT_PATTERN.sub(" ", movetext)))


def estimated_duration_seconds(time_control: str) -> int | None:
    """Expected length of one player's clock: base + 40 x increment.

    Both sites classify cadences from this quantity, so it is the natural way
    to compare time controls across them. Returns None for daily or malformed
    controls.
    """
    base, _, increment = time_control.partition("+")
    try:
        return int(base) + 40 * int(increment or 0)
    except ValueError:
        return None


def classify_result(code: str) -> str:
    """Map a chess.com per-side outcome code to `win`, `draw` or `loss`."""
    if code == "win":
        return "win"
    if code in _DRAW_CODES:
        return "draw"
    return "loss"


@dataclass(frozen=True)
class GameRecord:
    """One rated game, stored once, with both sides described."""

    game_id: str
    url: str
    platform: str
    year: int
    month: str
    time_class: str
    time_control: str
    ply_count: int
    white_username: str
    white_rating: int
    white_result: str
    black_username: str
    black_rating: int
    black_result: str
    pgn: str
    reported_accuracy_white: float | None = None
    reported_accuracy_black: float | None = None
    #: Lichess flags ratings still inside their provisional period; chess.com
    #: exposes nothing equivalent per game, so this stays None there.
    white_provisional: bool | None = None
    black_provisional: bool | None = None

    def sides(self) -> Iterator["SideView"]:
        """Yield the two per-player views of this game."""
        yield SideView(self, "white")
        yield SideView(self, "black")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GameRecord":
        fields = {name: payload[name] for name in cls.__dataclass_fields__ if name in payload}
        return cls(**fields)


@dataclass(frozen=True)
class SideView:
    """One player's perspective on a game: their rating, their result."""

    game: GameRecord
    colour: str

    @property
    def username(self) -> str:
        return getattr(self.game, f"{self.colour}_username")

    @property
    def rating(self) -> int:
        return getattr(self.game, f"{self.colour}_rating")

    @property
    def result(self) -> str:
        return getattr(self.game, f"{self.colour}_result")

    @property
    def opponent_rating(self) -> int:
        other = "black" if self.colour == "white" else "white"
        return getattr(self.game, f"{other}_rating")

    @property
    def reported_accuracy(self) -> float | None:
        return getattr(self.game, f"reported_accuracy_{self.colour}")

    @property
    def provisional(self) -> bool | None:
        return getattr(self.game, f"{self.colour}_provisional")


@dataclass(frozen=True)
class RatingSnapshot:
    """One player's ratings in both cadences, as a single month of games showed them.

    `*_rating` is the rating the player carried *after* their last rated game of
    that cadence in the month (or on it, where the site does not report the
    change), and `*_games` how many rated games of that cadence they finished.
    A cadence the player did not touch that month has rating None and 0 games.
    """

    platform: str
    username: str
    month: str
    blitz_rating: int | None
    blitz_games: int
    rapid_rating: int | None
    rapid_games: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RatingSnapshot":
        fields = {name: payload[name] for name in cls.__dataclass_fields__ if name in payload}
        return cls(**fields)
