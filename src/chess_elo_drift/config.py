"""Central configuration for the study.

Every tunable of the experiment lives here so that a run is fully described by
this module plus the command line arguments. Values are immutable: a run should
never be able to mutate the protocol it is executing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"


@dataclass(frozen=True, order=True)
class YearMonth:
    """A month of play, the granularity of the chess.com archive endpoint."""

    year: int
    month: int

    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}"

    @property
    def path_fragment(self) -> str:
        """The `YYYY/MM` fragment used by the archive URL."""
        return f"{self.year}/{self.month:02d}"


@dataclass(frozen=True)
class Era:
    """A period being compared, made of the months we sample from it."""

    name: str
    months: tuple[YearMonth, ...]

    def __contains__(self, month: YearMonth) -> bool:
        return month in self.months


# The two periods under comparison. The same calendar months are used on both
# sides so that any seasonality in who plays (school terms, holidays, the
# January influx of new accounts) cancels out of the comparison.
_SAMPLED_MONTHS = (3, 6, 9)

LEGACY_ERA = Era(
    name="2018-2019",
    months=tuple(YearMonth(y, m) for y in (2018, 2019) for m in _SAMPLED_MONTHS),
)
MODERN_ERA = Era(
    name="2024-2025",
    months=tuple(YearMonth(y, m) for y in (2024, 2025) for m in _SAMPLED_MONTHS),
)
ERAS: tuple[Era, ...] = (LEGACY_ERA, MODERN_ERA)


# --- Game eligibility -------------------------------------------------------

#: Only these two time classes are studied, and never pooled together: a blitz
#: rating and a rapid rating are different scales on chess.com.
TIME_CLASSES: tuple[str, ...] = ("blitz", "rapid")

#: Rating window requested for the study, applied to each side of a game
#: independently using the rating that side carried *at the time of that game*.
RATING_MIN = 600
RATING_MAX = 2200

#: Stratification used to steer the crawler so that the whole window is covered
#: rather than only the crowded middle of the distribution.
CRAWL_BAND_WIDTH = 200

#: Coarser bands used for reporting, where cell counts need to stay usable.
REPORT_BAND_WIDTH = 400

#: Games shorter than this are aborts, pre-moved resignations or disconnections;
#: they carry no usable signal about playing strength.
MIN_PLIES = 20

#: Guards against a single prolific account dominating a cell of the design.
MAX_GAMES_PER_PLAYER_PER_CELL = 3


# --- Engine analysis --------------------------------------------------------

#: Fixed search depth. Identical for every game of every era: the measurement
#: instrument must not drift between the two things being compared.
ENGINE_DEPTH = 12

#: Opening plies skipped when scoring. Book moves are memorised, not calculated,
#: and their availability has changed over the years; scoring them would measure
#: opening preparation rather than playing strength.
OPENING_PLIES_SKIPPED = 8

#: Safety cap on how many plies of a single game are scored, counted from the
#: end of the opening. Set high enough that ordinary games are scored in full;
#: it exists so that a pathological 300-move game cannot monopolise a worker.
MAX_PLIES_SCORED_PER_GAME = 200

#: A side scored on fewer moves than this says too little to enter the analysis.
MIN_MOVES_SCORED_PER_SIDE = 8

#: Where the UCI engine binary lives, and how many are run at once.
ENGINE_PATH = PROJECT_ROOT / "tools" / "stockfish" / "stockfish.exe"
DEFAULT_ENGINE_WORKERS = 14


# --- HTTP -------------------------------------------------------------------

API_ROOT = "https://api.chess.com/pub"

#: chess.com asks for a contactable User-Agent and throttles parallel callers.
USER_AGENT = "chess-elo-drift-research/1.0 (academic study; contact via repository)"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0
