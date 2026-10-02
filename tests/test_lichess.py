import io
import json
from datetime import datetime, timezone

import chess.pgn
import pytest

from chess_elo_drift import config
from chess_elo_drift.config import YearMonth, era_for_year
from chess_elo_drift.http import NotFoundError
from chess_elo_drift.lichess.extraction import extract_game
from chess_elo_drift.lichess.games import (
    BULK_USERS_LIMIT,
    fetch_month_games,
    fetch_rating_history,
    fetch_users,
    month_bounds_ms,
)
from chess_elo_drift.lichess.seeds import load_lichess_seeds
from chess_elo_drift.lichess.source import LichessSource, lichess_snapshot

MARCH_2018 = YearMonth(2018, 3)

#: Twenty-two plies of a Ruy Lopez, in the movetext shape `pgnInJson` produces.
MOVES = (
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 c4 c6"
)

LICHESS_PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/AbCd1234"]
[Date "2018.03.14"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[UTCDate "2018.03.14"]
[UTCTime "12:00:00"]
[WhiteElo "1500"]
[BlackElo "1480"]
[WhiteRatingDiff "+6"]
[BlackRatingDiff "-6"]
[Variant "Standard"]
[TimeControl "300+3"]
[ECO "C95"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 c6 1-0"""


def raw_game(**overrides):
    payload = {
        "id": "AbCd1234",
        "rated": True,
        "variant": "standard",
        "speed": "blitz",
        "perf": "blitz",
        "createdAt": 1521028800000,
        "lastMoveAt": 1521029400000,
        "status": "resign",
        "players": {
            "white": {
                "user": {"name": "Alice", "id": "alice"},
                "rating": 1500,
                "ratingDiff": 6,
            },
            "black": {
                "user": {"name": "Bob", "id": "bob"},
                "rating": 1480,
                "ratingDiff": -6,
                "provisional": True,
            },
        },
        "winner": "white",
        "moves": MOVES,
        "pgn": LICHESS_PGN,
        "clock": {"initial": 300, "increment": 3, "totalTime": 420},
    }
    payload.update(overrides)
    return payload


def extract(month=MARCH_2018, **overrides):
    return extract_game(raw_game(**overrides), month=month)


def with_player(colour, **changes):
    """A `players` override that changes one side and keeps the other."""
    players = raw_game()["players"]
    players[colour] = {**players[colour], **changes}
    return {"players": players}


class FakeLichess:
    """Answers the three client calls from canned payloads and records them."""

    def __init__(self, *, history=None, games=None, users=None):
        self.history = history
        self.games = games or []
        self.users = users or []
        self.calls: list[tuple[str, str, object]] = []

    def get(self, path, *, params=None):
        self.calls.append(("get", path, params))
        if self.history is None:
            raise NotFoundError(path)
        return self.history

    def get_ndjson(self, path, *, params=None):
        self.calls.append(("ndjson", path, params))
        if self.games is None:
            raise NotFoundError(path)
        return self.games

    def post_text(self, path, body, *, params=None):
        self.calls.append(("post", path, body))
        return self.users


# -- extraction --------------------------------------------------------------


def test_a_normal_game_becomes_a_record_with_both_sides_described():
    record = extract()
    assert record.game_id == "lichess:AbCd1234"
    assert record.url == "https://lichess.org/AbCd1234"
    assert record.platform == "lichess"
    assert (record.year, record.month) == (2018, "2018-03")
    assert record.time_class == "blitz"
    assert record.time_control == "300+3"
    assert record.ply_count == 22
    assert (record.white_username, record.white_rating, record.white_result) == ("alice", 1500, "win")
    assert (record.black_username, record.black_rating, record.black_result) == ("bob", 1480, "loss")
    assert record.reported_accuracy_white is None and record.reported_accuracy_black is None


def test_usernames_are_lowercased_from_the_account_id():
    payload = with_player("white", user={"name": "AliCE", "id": "AliCE"})
    assert extract(**payload).white_username == "alice"


def test_provisional_flags_default_to_false_when_lichess_omits_them():
    record = extract()
    assert record.white_provisional is False
    assert record.black_provisional is True


@pytest.mark.parametrize(
    ("winner", "white", "black"),
    [("white", "win", "loss"), ("black", "loss", "win"), (None, "draw", "draw")],
)
def test_results_follow_the_winner_and_a_missing_winner_is_a_draw(winner, white, black):
    overrides = {"winner": winner} if winner else {}
    payload = raw_game(**overrides)
    if not winner:
        payload.pop("winner")
    record = extract_game(payload, month=MARCH_2018)
    assert (record.white_result, record.black_result) == (white, black)


@pytest.mark.parametrize(
    "overrides",
    [
        {"variant": "chess960"},
        {"rated": False},
        {"initialFen": "8/8/8/8/8/8/8/K6k w - - 0 1"},
        {"speed": "bullet", "perf": "bullet"},
        {"speed": "classical", "perf": "classical"},
        {"speed": "correspondence", "perf": "correspondence"},
        {"perf": "rapid"},
        {"moves": "e4 e5 Nf3 Nc6"},
        {"pgn": None},
        {"moves": ""},
    ],
    ids=lambda overrides: ",".join(overrides),
)
def test_games_outside_the_study_are_rejected(overrides):
    assert extract(**overrides) is None


@pytest.mark.parametrize(
    "status", ["aborted", "noStart", "created", "started", "unknownFinish", "cheat"]
)
def test_games_that_never_finished_properly_are_rejected(status):
    assert extract(status=status) is None


@pytest.mark.parametrize("status", ["mate", "resign", "outoftime", "stalemate", "draw", "timeout"])
def test_every_ordinary_ending_is_accepted(status):
    assert extract(status=status) is not None


def test_a_game_of_exactly_the_minimum_length_is_kept():
    moves = " ".join(MOVES.split()[: config.MIN_PLIES])
    assert extract(moves=moves).ply_count == config.MIN_PLIES
    assert extract(moves=" ".join(MOVES.split()[: config.MIN_PLIES - 1])) is None


def test_games_against_the_engine_are_rejected():
    payload = raw_game(players={
        "white": {"user": {"name": "Alice", "id": "alice"}, "rating": 1500},
        "black": {"aiLevel": 3},
    })
    assert extract_game(payload, month=MARCH_2018) is None


def test_games_against_a_bot_account_are_rejected():
    payload = with_player("black", user={"name": "SomeBot", "id": "somebot", "title": "BOT"})
    assert extract(**payload) is None


def test_a_titled_human_is_not_mistaken_for_a_bot():
    payload = with_player("black", user={"name": "GM", "id": "gm", "title": "GM"})
    assert extract(**payload).black_username == "gm"


def test_an_anonymous_or_unrated_side_is_rejected():
    anonymous = raw_game()
    anonymous["players"]["black"] = {"rating": 1480}
    assert extract_game(anonymous, month=MARCH_2018) is None
    unrated = with_player("white", rating=None)
    assert extract(**unrated) is None


def test_a_game_without_a_clock_is_rejected():
    payload = raw_game()
    del payload["clock"]
    assert extract_game(payload, month=MARCH_2018) is None


# -- the Rapid pool ----------------------------------------------------------


def test_a_pre_2018_rapid_game_is_dropped_because_its_rating_is_a_classical_one():
    """Lichess labels a 2016 `10+0` game "rapid" today; the rating on it is Classical."""
    payload = {"speed": "rapid", "perf": "rapid", "clock": {"initial": 600, "increment": 0}}
    assert extract(month=YearMonth(2016, 3), **payload) is None
    assert extract(month=YearMonth(2017, 9), **payload) is None


def test_rapid_counts_from_the_month_the_pool_opened():
    payload = {"speed": "rapid", "perf": "rapid", "clock": {"initial": 600, "increment": 0}}
    assert extract(month=YearMonth(2018, 1), **payload).time_class == "rapid"


def test_pre_2018_blitz_is_kept():
    assert extract(month=YearMonth(2014, 3)).time_class == "blitz"


# -- the pgn -----------------------------------------------------------------


def test_the_embedded_pgn_parses_and_agrees_with_the_move_list():
    record = extract()
    game = chess.pgn.read_game(io.StringIO(record.pgn))
    assert game is not None
    assert not game.errors
    assert len(list(game.mainline_moves())) == record.ply_count
    assert game.headers["White"] == "alice"
    assert game.headers["Result"] == "1-0"


# -- month bounds ------------------------------------------------------------


def test_month_bounds_are_utc_month_starts_in_milliseconds():
    assert month_bounds_ms(YearMonth(2018, 3)) == (1519862400000, 1522540800000)


def test_month_bounds_roll_over_the_year_end():
    since, until = month_bounds_ms(YearMonth(2017, 12))
    assert until == month_bounds_ms(YearMonth(2018, 1))[0]
    assert until - since == 31 * 24 * 3600 * 1000


def test_the_games_request_asks_for_one_capped_window_of_rated_blitz_and_rapid():
    client = FakeLichess(games=[])
    fetch_month_games(client, "alice", YearMonth(2018, 3), max_games=40)
    kind, path, params = client.calls[0]
    assert (kind, path) == ("ndjson", "/api/games/user/alice")
    assert (params["since"], params["until"]) == month_bounds_ms(YearMonth(2018, 3))
    assert params["max"] == 40
    assert params["rated"] == "true"
    assert params["perfType"] == "blitz,rapid"
    assert params["pgnInJson"] == "true"
    assert params["moves"] == "true"


def test_a_closed_account_has_no_games():
    assert fetch_month_games(FakeLichess(games=None), "gone", MARCH_2018, max_games=5) == []


def test_bulk_user_lookups_refuse_more_than_lichess_accepts():
    with pytest.raises(ValueError):
        fetch_users(FakeLichess(), ["a"] * (BULK_USERS_LIMIT + 1))


def test_bulk_user_lookups_join_the_ids_into_one_request():
    client = FakeLichess(users=[{"id": "a"}, {"id": "b"}])
    assert fetch_users(client, ["a", "b"]) == [{"id": "a"}, {"id": "b"}]
    assert client.calls == [("post", "/api/users", "a,b")]


# -- rating history and active months ----------------------------------------


def history(**series):
    return [{"name": name, "points": points} for name, points in series.items()]


def test_rating_history_months_are_zero_based_on_the_wire():
    client = FakeLichess(history=history(Blitz=[[2018, 2, 14, 1500], [2018, 11, 1, 1550]]))
    points = fetch_rating_history(client, "alice")["Blitz"]
    assert [point.month for point in points] == [YearMonth(2018, 3), YearMonth(2018, 12)]
    assert points[0].rating == 1500 and points[0].day == 14


def test_an_unknown_account_has_no_history():
    assert fetch_rating_history(FakeLichess(history=None), "ghost") == {}


def history_source(client):
    return LichessSource(client, use_rating_history=True)


def test_history_pruning_keeps_the_era_months_with_a_blitz_point():
    client = FakeLichess(history=history(Blitz=[
        [2018, 2, 14, 1500],   # March: sampled
        [2018, 3, 2, 1510],    # April: not a sampled month
        [2018, 8, 30, 1520],   # September: sampled
        [2019, 5, 1, 1530],    # another year
    ]))
    months = history_source(client).active_months("alice", era_for_year(2018))
    assert months == [YearMonth(2018, 3), YearMonth(2018, 9)]
    assert client.calls == [("get", "/api/user/alice/rating-history", None)]


def test_the_zero_based_month_is_not_read_as_one_based():
    """A point in `month0 == 3` is April; read naively it would look like March."""
    client = FakeLichess(history=history(Blitz=[[2018, 3, 5, 1500]]))
    assert history_source(client).active_months("alice", era_for_year(2018)) == []


def test_history_rapid_points_count_only_once_the_rapid_pool_exists():
    client = FakeLichess(history=history(Rapid=[[2017, 2, 5, 1500], [2017, 5, 5, 1500]]))
    assert history_source(client).active_months("alice", era_for_year(2017)) == []

    client = FakeLichess(history=history(Rapid=[[2018, 2, 5, 1500]]))
    assert history_source(client).active_months("alice", era_for_year(2018)) == [MARCH_2018]


def test_history_of_other_cadences_does_not_make_a_month_active():
    client = FakeLichess(history=history(Bullet=[[2018, 2, 5, 1500]], Classical=[[2018, 5, 5, 1500]]))
    assert history_source(client).active_months("alice", era_for_year(2018)) == []


def test_an_empty_history_means_unknown_not_idle():
    """Unauthenticated, Lichess answers `[]` (or series with no points) for most accounts."""
    era = era_for_year(2018)
    bare = [{"name": "Blitz", "points": []}, {"name": "Rapid", "points": []}]
    for payload in ([], bare):
        source = history_source(FakeLichess(history=payload))
        assert source.active_months("alice", era) == list(era.months)


def test_a_closed_account_is_unknown_to_the_history_too():
    era = era_for_year(2018)
    assert history_source(FakeLichess(history=None)).active_months("gone", era) == list(era.months)


def test_without_history_pruning_an_unknown_account_costs_no_request():
    client = FakeLichess()
    era = era_for_year(2018)
    assert LichessSource(client).active_months("alice", era) == list(era.months)
    assert client.calls == []


def test_a_game_puts_its_month_first_for_both_players():
    """The opponents the crawler queues were found in a game, so their month is known."""
    client = FakeLichess(games=[raw_game()])
    source = LichessSource(client)
    june = YearMonth(2018, 6)
    source.month_games("alice", june)

    era = era_for_year(2018)
    others = [month for month in era.months if month != june]
    assert source.active_months("bob", era) == [june, *others]
    assert source.active_months("Alice", era) == [june, *others]
    assert source.active_months("carol", era) == list(era.months)
    assert client.calls[-1][0] == "ndjson"  # no extra request for the lookups


def test_a_proven_month_does_not_hide_the_others():
    """Offering only the month an opponent was met in kept every crawl in the
    month its first productive account was read in: one month per year."""
    client = FakeLichess(games=[raw_game()])
    source = LichessSource(client)
    source.month_games("alice", MARCH_2018)
    era = era_for_year(2018)
    assert set(source.active_months("bob", era)) == set(era.months)


def test_months_proven_in_another_year_do_not_leak_into_this_one():
    client = FakeLichess(games=[raw_game()])
    source = LichessSource(client)
    source.month_games("alice", MARCH_2018)
    era_2019 = era_for_year(2019)
    assert source.active_months("bob", era_2019) == list(era_2019.months)


def test_a_game_too_short_to_use_proves_nothing():
    client = FakeLichess(games=[raw_game(moves="e4 e5")])
    source = LichessSource(client)
    source.month_games("alice", MARCH_2018)
    era = era_for_year(2018)
    assert source.active_months("bob", era) == list(era.months)


# -- reading a month and the snapshot ----------------------------------------


def test_month_games_returns_the_usable_games_and_a_snapshot_from_all_of_them():
    short = raw_game(id="short001", moves="e4 e5 Nf3", lastMoveAt=1521031000000)
    client = FakeLichess(games=[short, raw_game()])
    harvest = LichessSource(client).month_games("alice", MARCH_2018)
    assert [game.game_id for game in harvest.games] == ["lichess:AbCd1234"]
    assert harvest.snapshot.blitz_games == 2
    assert client.calls[0][2]["max"] == 100


def test_the_game_cap_is_a_constructor_argument():
    client = FakeLichess(games=[])
    LichessSource(client, max_games_per_month=25).month_games("alice", MARCH_2018)
    assert client.calls[0][2]["max"] == 25


def test_the_snapshot_ends_on_the_rating_after_the_last_game_of_each_cadence():
    earlier = raw_game(id="g1", lastMoveAt=1000)
    earlier["players"]["white"].update(rating=1490, ratingDiff=10)
    later = raw_game(id="g2", lastMoveAt=2000)
    later["players"]["white"].update(rating=1500, ratingDiff=-8)
    rapid = raw_game(id="g3", speed="rapid", perf="rapid", lastMoveAt=500)
    rapid["players"]["black"].update(rating=1700, ratingDiff=5)

    # Lichess lists newest first; the snapshot must not depend on the order.
    snapshot = lichess_snapshot([later, earlier, rapid], "Alice", MARCH_2018)
    assert (snapshot.platform, snapshot.username, snapshot.month) == ("lichess", "alice", "2018-03")
    assert (snapshot.blitz_rating, snapshot.blitz_games) == (1492, 2)
    assert (snapshot.rapid_rating, snapshot.rapid_games) == (1506, 1)

    bob = lichess_snapshot([later, earlier, rapid], "bob", MARCH_2018)
    assert (bob.blitz_rating, bob.blitz_games) == (1474, 2)
    assert (bob.rapid_rating, bob.rapid_games) == (1705, 1)


def test_the_snapshot_falls_back_to_the_rating_when_there_is_no_rating_diff():
    game = raw_game()
    del game["players"]["white"]["ratingDiff"]
    assert lichess_snapshot([game], "alice", MARCH_2018).blitz_rating == 1500


def test_the_snapshot_counts_short_games_and_games_against_bots():
    short = raw_game(moves="e4 e5")
    bot = raw_game(**with_player("black", user={"name": "B", "id": "b", "title": "BOT"}))
    snapshot = lichess_snapshot([short, bot], "alice", MARCH_2018)
    assert snapshot.blitz_games == 2


def test_the_snapshot_ignores_unrated_variant_and_other_cadence_games():
    games = [
        raw_game(rated=False),
        raw_game(variant="crazyhouse"),
        raw_game(speed="bullet", perf="bullet"),
        raw_game(status="aborted"),
    ]
    assert lichess_snapshot(games, "alice", MARCH_2018) is None


def test_the_snapshot_leaves_out_pre_2018_rapid_whose_rating_is_classical():
    game = raw_game(speed="rapid", perf="rapid")
    snapshot = lichess_snapshot([game], "alice", YearMonth(2016, 3))
    assert snapshot is None


def test_no_games_means_no_snapshot():
    assert lichess_snapshot([], "alice", MARCH_2018) is None


# -- seeds -------------------------------------------------------------------

DAY_MS = 24 * 3600 * 1000


def profile(name, *, created, seen, blitz=(1500, 500), rapid=(0, 0), **extra):
    """A bulk `/api/users` entry; `created` and `seen` are ISO dates."""
    def ms(day):
        return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
    return {
        "id": name,
        "username": name,
        "createdAt": ms(created),
        "seenAt": ms(seen),
        "perfs": {
            "blitz": {"rating": blitz[0], "games": blitz[1]},
            "rapid": {"rating": rapid[0], "games": rapid[1]},
        },
        **extra,
    }


class FakeSeedApi:
    """Serves team pages, leaderboards and bulk profiles, and counts the requests."""

    def __init__(self, profiles, *, leaders=(), board=()):
        self.profiles = {entry["id"]: entry for entry in profiles}
        self.leaders = list(leaders)
        self.board = list(board)
        self.requests = 0

    def get(self, path, *, params=None):
        self.requests += 1
        if path == "/api/team/all":
            page = [{"id": "team", "leaders": [{"id": name} for name in self.leaders]}]
            return {"currentPageResults": page if params["page"] == 1 else []}
        assert path.startswith("/api/player/top/")
        return {"users": [{"id": name} for name in self.board]}

    def post_text(self, path, body, *, params=None):
        self.requests += 1
        assert path == "/api/users"
        return [self.profiles[name] for name in body.split(",") if name in self.profiles]


def seed_names(api, era_year, tmp_path, **kwargs):
    return load_lichess_seeds(api, era_for_year(era_year), tmp_path, team_pages=2, **kwargs)


def test_a_seed_must_exist_before_the_era_and_have_been_seen_since_it_began(tmp_path):
    api = FakeSeedApi(
        [
            profile("veteran", created="2012-01-01", seen="2026-09-01"),
            profile("newcomer", created="2015-06-01", seen="2026-09-01"),
            profile("lapsed", created="2012-01-01", seen="2013-06-01"),
        ],
        leaders=["veteran", "newcomer", "lapsed"],
    )
    assert seed_names(api, 2014, tmp_path) == ["veteran"]
    assert sorted(seed_names(api, 2016, tmp_path)) == ["newcomer", "veteran"]


def test_an_account_created_during_the_era_is_not_a_seed_for_it(tmp_path):
    api = FakeSeedApi([profile("march", created="2018-03-15", seen="2026-01-01")], leaders=["march"])
    assert seed_names(api, 2018, tmp_path) == []
    assert seed_names(api, 2019, tmp_path) == ["march"]


def test_bots_flagged_closed_and_thin_accounts_never_seed(tmp_path):
    base = dict(created="2012-01-01", seen="2026-09-01")
    api = FakeSeedApi(
        [
            profile("good", **base),
            profile("robot", title="BOT", **base),
            profile("cheater", tosViolation=True, **base),
            profile("closed", disabled=True, **base),
            profile("thin", blitz=(1500, 3), **base),
            {"id": "gone", "username": "gone", "disabled": True},
        ],
        leaders=["good", "robot", "cheater", "closed", "thin", "gone"],
    )
    assert seed_names(api, 2020, tmp_path) == ["good"]


def test_rapid_games_count_towards_having_been_active(tmp_path):
    api = FakeSeedApi(
        [profile("rapidfan", created="2012-01-01", seen="2026-09-01", blitz=(0, 0), rapid=(1600, 400))],
        leaders=["rapidfan"],
    )
    assert seed_names(api, 2020, tmp_path) == ["rapidfan"]


def test_the_pool_is_fetched_once_and_shared_by_every_era(tmp_path):
    api = FakeSeedApi(
        [profile("veteran", created="2012-01-01", seen="2026-09-01")],
        leaders=["veteran"], board=["veteran"],
    )
    seed_names(api, 2014, tmp_path)
    first_cost = api.requests
    assert first_cost <= 4 + 2 + 1  # leaderboards, team pages, one bulk lookup
    assert (tmp_path / "seed_pool.json").exists()

    for year in (2015, 2020, 2026):
        seed_names(api, year, tmp_path)
    assert api.requests == first_cost


def test_the_same_account_on_two_lists_is_looked_up_once(tmp_path):
    api = FakeSeedApi(
        [profile("both", created="2012-01-01", seen="2026-09-01")], leaders=["both"], board=["both"]
    )
    assert seed_names(api, 2020, tmp_path) == ["both"]


def test_the_seed_order_is_deterministic_per_era(tmp_path):
    profiles = [
        profile(f"player{i:02d}", created="2012-01-01", seen="2026-09-01", blitz=(900 + 40 * i, 500))
        for i in range(30)
    ]
    api = FakeSeedApi(profiles, leaders=[entry["id"] for entry in profiles])
    first = seed_names(api, 2020, tmp_path)
    assert seed_names(api, 2020, tmp_path) == first
    assert sorted(first) == sorted(entry["id"] for entry in profiles)
    assert seed_names(api, 2021, tmp_path) != first


def test_seeds_are_dealt_out_across_rating_bands_before_repeating_one(tmp_path):
    """The first seeds must already span the ratings rather than crowd one band."""
    low = [profile(f"low{i}", created="2012-01-01", seen="2026-09-01", blitz=(1000, 500)) for i in range(20)]
    high = [profile(f"high{i}", created="2012-01-01", seen="2026-09-01", blitz=(2200, 500)) for i in range(20)]
    api = FakeSeedApi(low + high, leaders=[entry["id"] for entry in low + high])
    first_four = seed_names(api, 2020, tmp_path)[:4]
    assert sum(name.startswith("low") for name in first_four) == 2
    assert sum(name.startswith("high") for name in first_four) == 2


def test_an_unreadable_pool_from_an_older_layout_is_rebuilt(tmp_path):
    (tmp_path / "seed_pool.json").write_text(json.dumps({"version": 0, "accounts": []}))
    api = FakeSeedApi([profile("veteran", created="2012-01-01", seen="2026-09-01")], leaders=["veteran"])
    assert seed_names(api, 2020, tmp_path) == ["veteran"]
