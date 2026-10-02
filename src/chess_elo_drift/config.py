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

    @classmethod
    def parse(cls, text: str) -> "YearMonth":
        year, month = text.split("-")
        return cls(int(year), int(month))


@dataclass(frozen=True)
class Era:
    """A period being compared, made of the months we sample from it.

    Every era of this study is a single calendar year; the name is the year.
    """

    name: str
    months: tuple[YearMonth, ...]

    def __contains__(self, month: YearMonth) -> bool:
        return month in self.months

    @property
    def year(self) -> int:
        return self.months[0].year


# --- Platforms and periods --------------------------------------------------

PLATFORMS: tuple[str, ...] = ("chesscom", "lichess")

PLATFORM_LABELS = {"chesscom": "chess.com", "lichess": "Lichess"}

#: Every year from the first to the last is compared with every other.
FIRST_YEAR = 2014
LAST_YEAR = 2026
YEARS: tuple[int, ...] = tuple(range(FIRST_YEAR, LAST_YEAR + 1))

#: The same calendar months are sampled in every year so that any seasonality in
#: who plays (school terms, holidays, the January influx of new accounts) cancels
#: out of the year-to-year comparison.
SAMPLED_MONTHS: tuple[int, ...] = (3, 6, 9)

ERAS: tuple[Era, ...] = tuple(
    Era(name=str(year), months=tuple(YearMonth(year, month) for month in SAMPLED_MONTHS))
    for year in YEARS
)

#: The year whose four pools are put on a common scale by the conversion study.
CONVERSION_YEAR = LAST_YEAR


def era_for_year(year: int) -> Era:
    for era in ERAS:
        if era.year == year:
            return era
    raise KeyError(year)


# --- Rating pools -----------------------------------------------------------

#: Only these two time classes are studied, and never pooled together: a blitz
#: rating and a rapid rating are different scales on both sites.
TIME_CLASSES: tuple[str, ...] = ("blitz", "rapid")

#: First month in which a (platform, time class) rating pool existed.
#:
#: Lichess only split Rapid out of Classical on 1 December 2017, seeding each
#: player's new rapid rating with a copy of their classical one. Its API labels an
#: old game by today's speed rules, so a 2016 `10+0` game comes back as "rapid"
#: while the rating printed on it belongs to the Classical pool of the time. Those
#: games are not rapid ratings and are excluded, not relabelled. chess.com stores
#: the time class a game was rated in when it was played, so its label and its
#: rating always agree, even though the class boundaries themselves moved.
POOL_START: dict[tuple[str, str], YearMonth] = {
    ("lichess", "rapid"): YearMonth(2018, 1),
}


def pool_exists(platform: str, time_class: str, month: YearMonth) -> bool:
    """Whether `time_class` was a rating pool of its own on `platform` in `month`."""
    start = POOL_START.get((platform, time_class))
    return start is None or month >= start


#: Time controls whose class never changed on chess.com over the whole period,
#: used as a robustness check. chess.com rated 10|0 as blitz until 10 September
#: 2020 and as rapid since, so it is in neither set.
STABLE_TIME_CONTROLS: dict[str, frozenset[str]] = {
    "blitz": frozenset({"180", "180+2", "300", "300+2", "300+3", "300+5"}),
    "rapid": frozenset({"900+10", "900", "1800", "1800+20", "1500+10"}),
}


#: Rating window requested for the study, applied to each side of a game
#: independently using the rating that side carried *at the time of that game*.
#: It reaches higher than it would for chess.com alone because Lichess ratings
#: sit a few hundred points above chess.com's for the same player.
RATING_MIN = 600
RATING_MAX = 2400

#: Stratification used to steer the crawler so that the whole window is covered
#: rather than only the crowded middle of the distribution.
CRAWL_BAND_WIDTH = 200

#: Coarser bands used for reporting, where cell counts need to stay usable.
REPORT_BAND_WIDTH = 300

#: Games shorter than this are aborts, pre-moved resignations or disconnections;
#: they carry no usable signal about playing strength.
MIN_PLIES = 20

#: Guards against a single prolific account dominating a cell of the design.
MAX_GAMES_PER_PLAYER_PER_CELL = 3


# --- Engine analysis --------------------------------------------------------

#: Fixed search depth. Identical for every game of every year and platform: the
#: measurement instrument must not drift between the things being compared.
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
LICHESS_ROOT = "https://lichess.org"

#: Both sites ask for a contactable User-Agent and throttle parallel callers.
USER_AGENT = "chess-elo-drift-research/2.0 (academic study; contact via repository)"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0

#: Lichess asks that a client which receives a 429 waits a full minute.
LICHESS_RATE_LIMIT_PAUSE_SECONDS = 61.0


# --- Where each stage writes ------------------------------------------------


def raw_dir(platform: str) -> Path:
    """Raw collected data for one site: games, visited accounts, snapshots."""
    return DATA_RAW / platform


def games_path(platform: str, year: int) -> Path:
    return raw_dir(platform) / f"games_{year}.jsonl"


def visited_path(platform: str, year: int) -> Path:
    return raw_dir(platform) / f"visited_{year}.json"


def snapshots_path(platform: str) -> Path:
    return raw_dir(platform) / "rating_snapshots.jsonl"


def crawl_stats_path(platform: str, year: int) -> Path:
    return raw_dir(platform) / f"crawl_stats_{year}.json"


EVALUATIONS_PATH = DATA_PROCESSED / "evaluations.csv"

#: Current-rating surveys and cross-site account links for the conversion study.
CONVERSION_DIR = DATA_RAW / "conversion"
