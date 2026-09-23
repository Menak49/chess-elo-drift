import pytest

from chess_elo_drift import config
from chess_elo_drift.chesscom.client import ChessComError
from chess_elo_drift.collection.extraction import extract_game
from chess_elo_drift.collection.sampler import CoverageTarget, band_floor, study_bands
from chess_elo_drift.config import YearMonth
from chess_elo_drift.records import GameRecord
from chess_elo_drift.selection import balanced_subsample, summarise_cells

LONG_PGN = '[White "a"]\n[Black "b"]\n\n' + " ".join(
    f"{n}. e4 {n}... e5" for n in range(1, 21)
)


def raw_game(**overrides):
    payload = {
        "uuid": "abc",
        "url": "https://example.invalid/abc",
        "rules": "chess",
        "rated": True,
        "time_class": "blitz",
        "time_control": "300",
        "pgn": LONG_PGN,
        "white": {"username": "Alice", "rating": 1500, "result": "win"},
        "black": {"username": "Bob", "rating": 1490, "result": "resigned"},
        "accuracies": {"white": 77.7, "black": 61.2},
    }
    payload.update(overrides)
    return payload


def extract(**overrides):
    return extract_game(raw_game(**overrides), era_name="2018-2019", month=YearMonth(2018, 3))


# -- rating bands ------------------------------------------------------------


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(599, None), (600, 600), (799, 600), (800, 800), (1234, 1200), (2199, 2000), (2201, None)],
)
def test_band_floor_places_a_rating_in_its_band(rating, expected):
    assert band_floor(rating) == expected


def test_the_top_of_the_window_stays_in_the_last_band():
    """2200 is the inclusive edge of the study, not the start of a ninth band."""
    assert band_floor(config.RATING_MAX) == study_bands()[-1]


def test_bands_tile_the_whole_window():
    bands = study_bands()
    assert bands[0] == config.RATING_MIN
    assert bands[-1] + config.CRAWL_BAND_WIDTH == config.RATING_MAX


# -- quotas ------------------------------------------------------------------


def test_a_target_is_incomplete_until_every_cell_is_filled():
    target = CoverageTarget(2)
    for band in study_bands():
        for time_class in config.TIME_CLASSES:
            target.credit((time_class, band))
            target.credit((time_class, band))
    assert target.is_complete


def test_overfilling_one_cell_does_not_complete_the_design():
    target = CoverageTarget(2)
    for _ in range(50):
        target.credit(("blitz", 600))
    assert not target.is_complete
    assert target.deficit(("blitz", 600)) == 0
    assert target.deficit(("rapid", 600)) == 2


def test_band_deficit_sums_over_time_classes():
    target = CoverageTarget(5)
    target.credit(("blitz", 1000))
    assert target.band_deficit(1000) == 9


# -- extraction --------------------------------------------------------------


def test_a_normal_rated_game_is_extracted():
    record = extract()
    assert record is not None
    assert record.white_username == "alice"  # normalised for joining
    assert record.black_result == "loss"
    assert record.reported_accuracy_white == 77.7
    assert record.era == "2018-2019" and record.month == "2018-03"


@pytest.mark.parametrize(
    "override",
    [
        {"rules": "chess960"},
        {"rated": False},
        {"time_class": "bullet"},
        {"time_class": "daily"},
        {"pgn": '[White "a"]\n\n1. e4 1... e5 0-1'},
        {"white": {"username": "Alice", "result": "win"}},
    ],
)
def test_games_outside_the_protocol_are_rejected(override):
    assert extract(**override) is None


def test_a_game_with_no_analysis_still_extracts():
    record = extract(accuracies=None)
    assert record is not None
    assert record.reported_accuracy_white is None


# -- subsampling -------------------------------------------------------------


def _record(index: int, era: str, time_class: str, rating: int) -> GameRecord:
    return GameRecord(
        game_id=f"{era}-{time_class}-{rating}-{index}",
        url="https://example.invalid",
        era=era,
        month="2018-03",
        time_class=time_class,
        time_control="300",
        ply_count=40,
        white_username="a",
        white_rating=rating,
        white_result="win",
        black_username="b",
        black_rating=rating,
        black_result="loss",
        pgn=LONG_PGN,
    )


def test_subsampling_returns_everything_when_under_budget():
    records = [_record(i, "2018-2019", "blitz", 1000) for i in range(5)]
    assert balanced_subsample(records, 50) == records


def test_subsampling_spreads_the_budget_across_cells():
    """A uniform draw would spend the budget on the one crowded cell."""
    crowded = [_record(i, "2018-2019", "blitz", 1000) for i in range(200)]
    sparse = [_record(i, "2018-2019", "blitz", 2000) for i in range(10)]

    chosen = balanced_subsample(crowded + sparse, 20)

    counts = summarise_cells(chosen)
    assert counts[("2018-2019", "blitz", 2000)] == 10
    assert counts[("2018-2019", "blitz", 1000)] == 10


def test_subsampling_is_deterministic():
    records = [_record(i, "2024-2025", "rapid", 1400) for i in range(100)]
    assert balanced_subsample(records, 10) == balanced_subsample(records, 10)


# -- surviving a flaky API ---------------------------------------------------


class FlakyClient:
    """A stand-in for the API that fails on the accounts it is told to fail on."""

    def __init__(self, *, failing: frozenset[str] = frozenset(), fail_everything: bool = False):
        self._failing = failing
        self._fail_everything = fail_everything
        self.calls: list[str] = []

    def get(self, path: str) -> dict:
        self.calls.append(path)
        username = path.split("/")[1]
        if self._fail_everything or username in self._failing:
            raise ChessComError(f"HTTP 502 for {path}")
        if path.endswith("/archives"):
            return {"archives": [f"https://api.chess.com/pub/player/{username}/games/2018/03"]}
        return {"games": [raw_game(uuid=f"{username}-game")]}


def crawler_over(client, tmp_path, seeds, **kwargs):
    from chess_elo_drift.collection.sampler import StratifiedSnowballCrawler
    from chess_elo_drift.collection.store import GameStore

    return StratifiedSnowballCrawler(
        client,
        GameStore(tmp_path / "games.jsonl"),
        config.LEGACY_ERA,
        CoverageTarget(observations_per_cell=1),
        max_api_requests=200,
        **kwargs,
    )


def test_one_failing_account_does_not_end_the_crawl(tmp_path):
    """A 502 the client already retried is one bad account, not a dead API."""
    client = FlakyClient(failing=frozenset({"broken"}))
    crawler = crawler_over(client, tmp_path, ["broken", "alice"])

    stats = crawler.crawl(["broken", "alice"])

    assert stats.players_failed == 1
    assert stats.games_stored > 0  # the crawl carried on and collected


def test_a_run_of_failures_stops_the_crawl_rather_than_burning_the_frontier(tmp_path):
    """When every call fails the API is down, and walking on only marks good
    accounts as visited so a later resumed run would skip them."""
    client = FlakyClient(fail_everything=True)
    crawler = crawler_over(client, tmp_path, [], max_consecutive_failures=3)

    stats = crawler.crawl([f"player{i}" for i in range(50)])

    assert stats.players_failed == 3
    assert stats.games_stored == 0
