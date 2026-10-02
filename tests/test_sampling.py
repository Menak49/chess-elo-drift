import pytest

from chess_elo_drift import config
from chess_elo_drift.chesscom import ChessComClient
from chess_elo_drift.chesscom.client import ChessComError
from chess_elo_drift.collection.extraction import extract_game
from chess_elo_drift.collection.sampler import CoverageTarget, band_floor, study_bands
from chess_elo_drift.collection.sources import ChessComSource, chesscom_snapshot
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
    return extract_game(raw_game(**overrides), month=YearMonth(2018, 3))


# -- rating bands ------------------------------------------------------------


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(599, None), (600, 600), (799, 600), (800, 800), (1234, 1200), (2399, 2200), (2401, None)],
)
def test_band_floor_places_a_rating_in_its_band(rating, expected):
    assert band_floor(rating) == expected


def test_the_top_of_the_window_stays_in_the_last_band():
    """The top of the window is its inclusive edge, not the start of a new band."""
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


def test_a_full_month_does_not_fill_the_other_months():
    """A year filled from one month alone lets seasonality into the comparison."""
    target = CoverageTarget(6, months=config.era_for_year(2018).months)
    for _ in range(6):
        target.credit(("blitz", 1000), "2018-03")
    assert not target.wants(("blitz", 1000), "2018-03")
    assert target.wants(("blitz", 1000), "2018-06")
    assert target.deficit(("blitz", 1000)) == 4
    assert target.month_deficit("2018-03") < target.month_deficit("2018-09")


def test_priming_counts_each_game_in_its_own_month():
    target = CoverageTarget(3, months=config.era_for_year(2018).months)
    target.prime([extract()])  # a March 2018 game at 1500 and 1490
    assert not target.wants(("blitz", 1400), "2018-03")
    assert target.wants(("blitz", 1400), "2018-06")


# -- extraction --------------------------------------------------------------


def test_a_normal_rated_game_is_extracted():
    record = extract()
    assert record is not None
    assert record.white_username == "alice"  # normalised for joining
    assert record.black_result == "loss"
    assert record.reported_accuracy_white == 77.7
    assert record.platform == "chesscom"
    assert record.year == 2018 and record.month == "2018-03"


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


def test_ten_minute_games_keep_the_class_they_were_rated_in():
    """chess.com rated 10|0 as blitz until 2020; the rating printed on such a game
    belongs to the blitz pool, so the recorded class must stay blitz."""
    record = extract(time_control="600", time_class="blitz")
    assert record is not None and record.time_class == "blitz"


def test_a_pool_that_did_not_exist_yet_has_no_quota():
    target = CoverageTarget(3, time_classes=("blitz",))
    assert target.deficit(("rapid", 1000)) == 0
    for band in study_bands():
        for _ in range(3):
            target.credit(("blitz", band))
    assert target.is_complete


def test_a_snapshot_keeps_the_last_rating_of_each_cadence():
    def game(end_time, time_class, colour, rating, **extra):
        opponent = "black" if colour == "white" else "white"
        return raw_game(
            end_time=end_time,
            time_class=time_class,
            **{
                colour: {"username": "Alice", "rating": rating, "result": "win"},
                opponent: {"username": "Carol", "rating": 1400, "result": "resigned"},
            },
            **extra,
        )

    games = [
        game(10, "blitz", "white", 1500),
        game(30, "blitz", "black", 1516),
        game(20, "rapid", "black", 1611),
        game(40, "bullet", "white", 1300),
        game(50, "rapid", "white", 1, rated=False),
    ]
    snapshot = chesscom_snapshot(games, "Alice", YearMonth(2026, 3))
    assert snapshot is not None
    assert (snapshot.blitz_rating, snapshot.blitz_games) == (1516, 2)
    assert (snapshot.rapid_rating, snapshot.rapid_games) == (1611, 1)
    assert snapshot.month == "2026-03" and snapshot.username == "alice"


def test_a_month_with_no_blitz_or_rapid_has_no_snapshot():
    assert chesscom_snapshot([raw_game(time_class="bullet")], "alice", YearMonth(2026, 3)) is None


def test_a_game_with_no_analysis_still_extracts():
    record = extract(accuracies=None)
    assert record is not None
    assert record.reported_accuracy_white is None


# -- subsampling -------------------------------------------------------------


def _record(index: int, year: int, time_class: str, rating: int) -> GameRecord:
    return GameRecord(
        game_id=f"{year}-{time_class}-{rating}-{index}",
        url="https://example.invalid",
        platform="chesscom",
        year=year,
        month=f"{year}-03",
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
    records = [_record(i, 2018, "blitz", 1000) for i in range(5)]
    assert balanced_subsample(records, 50) == records


def test_subsampling_spreads_the_budget_across_cells():
    """A uniform draw would spend the budget on the one crowded cell."""
    crowded = [_record(i, 2018, "blitz", 1000) for i in range(200)]
    sparse = [_record(i, 2018, "blitz", 2000) for i in range(10)]

    chosen = balanced_subsample(crowded + sparse, 20)

    counts = summarise_cells(chosen)
    assert counts[("chesscom", 2018, "blitz", 2000)] == 10
    assert counts[("chesscom", 2018, "blitz", 1000)] == 10


def test_subsampling_is_deterministic():
    records = [_record(i, 2024, "rapid", 1400) for i in range(100)]
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
        ChessComSource(client),
        GameStore(tmp_path / "games.jsonl"),
        config.era_for_year(2018),
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


def test_a_crawl_that_fills_nothing_stops_before_its_budget(tmp_path):
    """Unreachable cells (a band below a site's rating floor) must not drain the
    whole request budget of every year they appear in."""
    client = FlakyClient()
    crawler = crawler_over(client, tmp_path, [], stall_requests=6)
    crawler._target.credit(("blitz", 1400))  # the one game on offer fills nothing new
    for _ in range(3):
        crawler._target.credit(("rapid", 1400))
    alice = [f"alice{i}" for i in range(40)]

    stats = crawler.crawl(alice)

    assert stats.api_requests < 20


def test_a_game_from_a_set_up_position_is_rejected():
    """Thematic tournaments start mid-gambit: the players did not choose those
    moves, and the engine cannot replay them from the normal start."""
    danish = "r1bqkbnr/pppp1ppp/2n5/8/3pP3/2P5/PP3PPP/RNBQKBNR w KQkq - 0 1"
    assert extract(initial_setup=danish) is None


class MonthRecordingSource:
    """A site where everyone played every sampled month, recording which month is read."""

    platform = "chesscom"
    index_is_free = True

    def __init__(self):
        self.months_read: list[YearMonth] = []

    def active_months(self, username, era):
        return list(era.months)

    def month_games(self, username, month):
        from chess_elo_drift.collection.sources import MonthHarvest

        self.months_read.append(month)
        record = extract_game(raw_game(uuid=f"{username}-{month}"), month=month)
        return MonthHarvest([record], None)


def test_the_crawl_reads_the_month_its_quotas_need_most(tmp_path):
    from chess_elo_drift.collection.sampler import StratifiedSnowballCrawler
    from chess_elo_drift.collection.store import GameStore

    era = config.era_for_year(2018)
    target = CoverageTarget(3, months=era.months)
    for band in study_bands():
        for time_class in config.TIME_CLASSES:
            target.credit((time_class, band), "2018-03")
            target.credit((time_class, band), "2018-09")
    source = MonthRecordingSource()
    crawler = StratifiedSnowballCrawler(
        source, GameStore(tmp_path / "games.jsonl"), era, target, max_api_requests=1
    )

    crawler.crawl(["alice"])

    assert source.months_read == [YearMonth(2018, 6)]


def test_a_resumed_crawl_walks_on_from_the_games_it_already_holds(tmp_path):
    """Every seed visited and a month still empty: the stored games' opponents
    are who the first run would have read next, not a reason to stop."""
    from chess_elo_drift.collection.sampler import StratifiedSnowballCrawler
    from chess_elo_drift.collection.store import GameStore

    era = config.era_for_year(2018)
    store = GameStore(tmp_path / "games.jsonl")
    store.add(extract_game(raw_game(uuid="old"), month=YearMonth(2018, 3)))
    source = MonthRecordingSource()
    crawler = StratifiedSnowballCrawler(
        source, store, era, CoverageTarget(3, months=era.months),
        max_api_requests=2, visited={"alice", "Alice"},
    )

    crawler.crawl(["alice"])

    assert source.months_read  # Bob, met in the stored game, was read
