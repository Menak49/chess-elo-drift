import json

import pytest

from chess_elo_drift import config
from chess_elo_drift.conversion import archives as archive_module
from chess_elo_drift.conversion import survey
from chess_elo_drift.conversion.harvest import LichessHarvester
from chess_elo_drift.conversion.links import chesscom_link, parse_chesscom_names
from chess_elo_drift.conversion.pools import chesscom_pool, lichess_pool
from chess_elo_drift.http import ApiError, NotFoundError

SEEN_2026_MS = 1_780_000_000_000  # mid 2026
SEEN_2025_MS = 1_740_000_000_000


# -- link parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("https://www.chess.com/member/Alice_99", ["alice_99"]),
        ("http://chess.com/member/alice", ["alice"]),
        ("chess.com/member/Alice", ["alice"]),
        ("www.chess.com/member/alice/", ["alice"]),
        ("https://www.chess.com/member/alice?tab=stats", ["alice"]),
        ("https://www.chess.com/member/alice#about", ["alice"]),
        ("HTTPS://WWW.CHESS.COM/MEMBER/Alice", ["alice"]),
        ("https://www.chess.com/es/member/alice", ["alice"]),
        ("https://www.chess.com/members/view/alice", ["alice"]),
        ("https://www.chess.com/stats/live/blitz/alice", ["alice"]),
        ("https://www.chess.com/games/archive/alice", ["alice"]),
        ("https://www.chess.com/blog/AliceB", ["aliceb"]),
        ("https://www.chess.com/member/mykola-bortnyk\r\nhttps://www.twitch.tv/x", ["mykola-bortnyk"]),
        ("my page: https://www.chess.com/member/alice-b, follow me", ["alice-b"]),
        ("github.com/me\nchess.com/member/alice\nmas.to/@me", ["alice"]),
        ("https://www.chess.com/member/alice https://chess.com/member/ALICE", ["alice"]),
        ("https://www.chess.com/member/alice https://www.chess.com/member/bob", ["alice", "bob"]),
    ],
)
def test_parse_chesscom_names(text, expected):
    assert parse_chesscom_names(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "I play on chess.com too",
        "chess.com: alice",
        "https://www.chess.com/players/magnus-carlsen",
        "https://www.chess.com/",
        "https://www.chess.com/member/",
        "https://www.chess.com/stats/live/blitz",
        "https://fakechess.com/member/alice",
        "https://lichess.org/@/alice",
        "https://www.chess.com/member/ab",
    ],
)
def test_parse_chesscom_names_ignores_junk(text):
    assert parse_chesscom_names(text) == []


def test_chesscom_link_reads_links_then_bio():
    assert chesscom_link({"links": "chess.com/member/Alice"}) == "alice"
    assert chesscom_link({"bio": "hi! https://www.chess.com/member/bob"}) == "bob"
    assert chesscom_link({"links": "chess.com/member/x1y", "bio": "chess.com/member/X1Y"}) == "x1y"


def test_chesscom_link_is_none_without_or_with_conflicting_links():
    assert chesscom_link(None) is None
    assert chesscom_link({}) is None
    assert chesscom_link({"links": "github.com/me"}) is None
    assert chesscom_link({"links": "chess.com/member/alice\nchess.com/member/bob"}) is None


# -- pools -------------------------------------------------------------------


def test_chesscom_pool_takes_the_documented_fields():
    stats = {
        "chess_blitz": {
            "last": {"rating": 1500, "date": 1_780_000_000, "rd": 60},
            "best": {"rating": 1650, "date": 1, "game": "x"},
            "record": {"win": 10, "loss": 8, "draw": 2},
        }
    }
    assert chesscom_pool(stats, "blitz") == {
        "rating": 1500, "rd": 60, "last_date": 1_780_000_000, "games": 20, "best": 1650
    }
    assert chesscom_pool(stats, "rapid") is None


def test_chesscom_pool_tolerates_missing_best():
    stats = {"chess_rapid": {"last": {"rating": 900, "date": 5, "rd": 200},
                             "record": {"win": 1, "loss": 0, "draw": 0}}}
    assert chesscom_pool(stats, "rapid")["best"] is None


def test_lichess_pool_reads_provisional_flag():
    user = {"perfs": {"blitz": {"games": 12, "rating": 1600, "rd": 90},
                      "rapid": {"games": 3, "rating": 1500, "rd": 300, "prov": True}}}
    assert lichess_pool(user, "blitz") == {
        "rating": 1600, "rd": 90, "games": 12, "provisional": False
    }
    assert lichess_pool(user, "rapid")["provisional"] is True
    assert lichess_pool({"perfs": {}}, "blitz") is None
    assert lichess_pool({}, "blitz") is None


# -- fakes -------------------------------------------------------------------


def chesscom_stats(blitz=1200, rapid=1300):
    return {
        "chess_blitz": {"last": {"rating": blitz, "date": 1_780_000_000, "rd": 50},
                        "best": {"rating": blitz + 50},
                        "record": {"win": 30, "loss": 30, "draw": 5}},
        "chess_rapid": {"last": {"rating": rapid, "date": 1_780_000_000, "rd": 70},
                        "record": {"win": 10, "loss": 10, "draw": 0}},
    }


class FakeChessCom:
    def __init__(self, known=None):
        self.known = {} if known is None else known
        self.calls = []
        self.request_count = 0

    def get(self, path, *, params=None):
        assert path.startswith("player/") and path.endswith("/stats")
        name = path.split("/")[1]
        self.calls.append(name)
        self.request_count += 1
        if name not in self.known:
            raise NotFoundError(path)
        return self.known[name]

    def close(self):
        pass


def lichess_user(name, *, link=None, seen=SEEN_2026_MS, blitz=1700, rapid=1800, **extra):
    profile = {"links": f"https://www.chess.com/member/{link}"} if link else {}
    user = {
        "id": name.lower(), "username": name, "createdAt": 1_600_000_000_000, "seenAt": seen,
        "profile": profile,
        "perfs": {"blitz": {"games": 50, "rating": blitz, "rd": 60},
                  "rapid": {"games": 5, "rating": rapid, "rd": 200, "prov": True}},
    }
    user.update(extra)
    return user


class FakeLichess:
    """Answers the JSON endpoints the harvester and the bulk lookup use."""

    def __init__(self, users, arena_players=(), team_leaders=()):
        self.users = {user["id"]: user for user in users}
        self.arena_players = list(arena_players)
        self.team_leaders = list(team_leaders)
        self.request_count = 0
        self.lookups = []
        self.paths = []

    def get(self, path, *, params=None):
        self.paths.append(path)
        if path == "api/tournament":
            return {"finished": [{"id": "arena1", "nbPlayers": max(len(self.arena_players), 30), "status": 30,
                                  "variant": {"key": "standard"}, "perf": {"key": "blitz"}}],
                    "started": []}
        if path == "api/tournament/arena1":
            start = (params["page"] - 1) * 10
            names = self.arena_players[start:start + 10]
            return {"standing": {"page": params["page"], "players": [{"name": n} for n in names]}}
        if path.startswith("api/player/top"):
            return {"users": []}
        if path == "api/team/all":
            return self._teams()
        if path == "api/team/search":
            return self._teams() if params["text"] == "chess.com" else {"currentPageResults": []}
        raise NotFoundError(path)

    def _teams(self):
        return {"currentPageResults": [
            {"id": "t1", "leaders": [{"id": name, "name": name} for name in self.team_leaders]}
        ], "nextPage": None}

    def post_text(self, path, body, *, params=None):
        assert path == "api/users"
        ids = body.split(",")
        assert len(ids) <= survey.LICHESS_BATCH_SIZE
        self.request_count += 1
        self.lookups.append(ids)
        return [self.users[i] for i in ids if i in self.users]

    def close(self):
        pass


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_RAW", tmp_path / "raw")
    return tmp_path


def write_chesscom_crawl(names_games=(), names_snapshots=()):
    games = config.games_path("chesscom", 2026)
    games.parent.mkdir(parents=True, exist_ok=True)
    with games.open("w", encoding="utf-8") as handle:
        for white, black in names_games:
            handle.write(json.dumps({"white_username": white, "black_username": black}) + "\n")
    with config.snapshots_path("chesscom").open("w", encoding="utf-8") as handle:
        for name, month in names_snapshots:
            handle.write(json.dumps({"username": name, "month": month}) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# -- chess.com stage ---------------------------------------------------------


def test_chesscom_candidates_use_2026_games_and_snapshots(data_root):
    write_chesscom_crawl(
        names_games=[("Alice", "Bob")],
        names_snapshots=[("carol", "2026-06"), ("dave", "2025-06"), ("BOB", "2026-03")],
    )
    names = survey.chesscom_candidates(
        config.games_path("chesscom", 2026), config.snapshots_path("chesscom"), 2026
    )
    assert sorted(names) == ["alice", "bob", "carol"]
    again = survey.chesscom_candidates(
        config.games_path("chesscom", 2026), config.snapshots_path("chesscom"), 2026
    )
    assert names == again


def test_chesscom_row_schema_and_not_found():
    client = FakeChessCom({"alice": chesscom_stats()})
    row = survey.chesscom_row(client, "alice")
    assert set(row) == {"username", "fetched_at", "found", "blitz", "rapid"}
    assert row["found"] is True
    assert row["blitz"] == {"rating": 1200, "rd": 50, "last_date": 1_780_000_000,
                            "games": 65, "best": 1250}
    assert row["rapid"]["best"] is None
    gone = survey.chesscom_row(client, "ghost")
    assert gone["found"] is False and gone["blitz"] is None and gone["rapid"] is None


def test_survey_chesscom_honours_cap_and_resumes(tmp_path):
    names = [f"user{i}" for i in range(6)]
    client = FakeChessCom({name: chesscom_stats() for name in names})
    store = survey.KeyedJsonl(tmp_path / "stats.jsonl")

    assert survey.survey_chesscom(client, store, names, max_requests=4) == 4
    assert len(client.calls) == 4

    store = survey.KeyedJsonl(tmp_path / "stats.jsonl")
    assert survey.survey_chesscom(client, store, names, max_requests=4) == 2
    assert client.calls == names  # nothing fetched twice
    assert [row["username"] for row in read_rows(tmp_path / "stats.jsonl")] == names


def test_survey_chesscom_skips_failures_and_stops_when_all_fail(tmp_path):
    class Broken(FakeChessCom):
        def get(self, path, *, params=None):
            self.request_count += 1
            raise ApiError("down")

    client = Broken()
    store = survey.KeyedJsonl(tmp_path / "stats.jsonl")
    names = [f"user{i}" for i in range(50)]
    assert survey.survey_chesscom(client, store, names, max_requests=50) == 0
    assert client.request_count == survey.MAX_CONSECUTIVE_FAILURES
    assert len(store) == 0


# -- Lichess stage -----------------------------------------------------------


def test_lichess_row_schema():
    user = lichess_user("Alice", link="AliceC", title="NM", tosViolation=True)
    row = survey.lichess_row(user, "arena:abc")
    assert set(row) == {"username", "fetched_at", "created_at", "seen_at", "title", "disabled",
                        "tos_violation", "blitz", "rapid", "chesscom_link", "source"}
    assert row["username"] == "alice"
    assert row["chesscom_link"] == "alicec"
    assert row["title"] == "NM" and row["tos_violation"] is True and row["disabled"] is False
    assert row["blitz"] == {"rating": 1700, "rd": 60, "games": 50, "provisional": False}
    assert row["rapid"]["provisional"] is True
    assert row["source"] == "arena:abc"


def test_lichess_row_of_a_disabled_account_has_no_pools():
    row = survey.lichess_row({"id": "gone", "username": "Gone", "disabled": True}, "team:x")
    assert row["disabled"] is True and row["blitz"] is None and row["rapid"] is None
    assert row["chesscom_link"] is None and row["seen_at"] is None


def test_harvester_reads_files_then_endpoints_and_labels_sources(tmp_path):
    games = tmp_path / "games_2026.jsonl"
    games.write_text(
        json.dumps({"white_username": "Ann", "black_username": "Ben"}) + "\n" + "{truncated",
        encoding="utf-8",
    )
    snapshots = tmp_path / "snapshots.jsonl"
    snapshots.write_text(
        json.dumps({"username": "Old", "month": "2025-06"}) + "\n"
        + json.dumps({"username": "Sam", "month": "2026-06"}) + "\n",
        encoding="utf-8",
    )
    pool = tmp_path / "seed_pool.json"
    pool.write_text(json.dumps({"accounts": [
        {"id": "pia", "seen": 1_780_000_000_000}, {"id": "dormant", "seen": 1_600_000_000_000},
    ]}), encoding="utf-8")

    client = FakeLichess([], arena_players=["Ben", "Cy", "Dee"], team_leaders=["Lea", "ann"])
    harvester = LichessHarvester(
        client, games_path=games, snapshots_path=snapshots, seed_pool_path=pool
    )
    assert list(harvester.candidates()) == [
        ("ann", "games_2026"), ("ben", "games_2026"), ("sam", "snapshots_2026"),
        ("pia", "seed_pool"), ("lea", "team_leader:biggest"),
        ("cy", "arena:arena1"), ("dee", "arena:arena1"),
    ]


def test_harvester_reads_every_standings_page_of_an_arena():
    players = [f"p{i:02d}" for i in range(35)]
    client = FakeLichess([], arena_players=players)
    names = [name for name, _ in LichessHarvester(client).candidates()]
    assert names == players
    assert client.paths.count("api/tournament/arena1") == 4  # 10 players per page


def test_harvester_survives_a_failing_source():
    class Broken(FakeLichess):
        def get(self, path, *, params=None):
            if path.startswith("api/player/top") or path == "api/team/all":
                raise ApiError("HTTP 500")
            return super().get(path, params=params)

    client = Broken([], arena_players=["Cy"])
    assert list(LichessHarvester(client).candidates()) == [("cy", "arena:arena1")]


def test_survey_lichess_batches_caps_and_resumes(tmp_path):
    names = [f"p{i:03d}" for i in range(700)]
    client = FakeLichess([lichess_user(name) for name in names], arena_players=names)
    store = survey.KeyedJsonl(tmp_path / "users.jsonl")
    missing = survey.IdList(tmp_path / "missing.txt")

    harvester = LichessHarvester(client)
    assert survey.survey_lichess(
        client, store, missing, harvester, max_users=450, bulk_interval=0
    ) == 450
    assert [len(batch) for batch in client.lookups] == [300, 150]

    store = survey.KeyedJsonl(tmp_path / "users.jsonl")
    missing = survey.IdList(tmp_path / "missing.txt")
    assert survey.survey_lichess(
        client, store, missing, LichessHarvester(client), 10_000, bulk_interval=0
    ) == 250
    looked_up = [name for batch in client.lookups for name in batch]
    assert sorted(looked_up) == names  # no username asked for twice
    assert len(read_rows(tmp_path / "users.jsonl")) == 700


def test_survey_lichess_remembers_closed_accounts(tmp_path):
    client = FakeLichess([lichess_user("open")], arena_players=["open", "closed"])
    store = survey.KeyedJsonl(tmp_path / "users.jsonl")
    missing = survey.IdList(tmp_path / "missing.txt")
    assert survey.survey_lichess(
        client, store, missing, LichessHarvester(client), 100, bulk_interval=0
    ) == 1
    assert "closed" in survey.IdList(tmp_path / "missing.txt")

    calls_before = client.request_count
    survey.survey_lichess(
        client, survey.KeyedJsonl(tmp_path / "users.jsonl"),
        survey.IdList(tmp_path / "missing.txt"), LichessHarvester(client), 100, bulk_interval=0,
    )
    assert client.request_count == calls_before  # nothing left to look up


def test_survey_lichess_spaces_bulk_calls_and_retries_a_429(tmp_path):
    names = [f"p{i:03d}" for i in range(650)]

    class Flaky(FakeLichess):
        failures_left = 2

        def post_text(self, path, body, *, params=None):
            if self.failures_left:
                self.failures_left -= 1
                raise ApiError("HTTP 429")
            return super().post_text(path, body, params=params)

    client = Flaky([lichess_user(name) for name in names], arena_players=names)
    sleeps = []
    now = [100.0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    added = survey.survey_lichess(
        client, survey.KeyedJsonl(tmp_path / "u.jsonl"), survey.IdList(tmp_path / "m.txt"),
        LichessHarvester(client), 10_000, bulk_interval=25.0, sleep=sleep, clock=lambda: now[0],
    )
    assert added == 650
    pauses = [seconds for seconds in sleeps if seconds > 60]
    assert pauses == [config.LICHESS_RATE_LIMIT_PAUSE_SECONDS] * 2  # waited, did not stop
    spacing = [seconds for seconds in sleeps if seconds <= 25]
    assert len(spacing) == 2 and all(seconds == 25.0 for seconds in spacing)


def test_survey_lichess_gives_up_after_repeated_failures(tmp_path):
    class Down(FakeLichess):
        def post_text(self, path, body, *, params=None):
            raise ApiError("HTTP 429")

    client = Down([lichess_user("a")], arena_players=["a"])
    store = survey.KeyedJsonl(tmp_path / "u.jsonl")
    added = survey.survey_lichess(
        client, store, survey.IdList(tmp_path / "m.txt"), LichessHarvester(client), 100,
        bulk_interval=0, sleep=lambda seconds: None,
    )
    assert added == 0 and len(store) == 0


# -- linked accounts ---------------------------------------------------------


def test_survey_linked_rows_reuse_and_filters(tmp_path):
    users = survey.KeyedJsonl(tmp_path / "users.jsonl")
    for user, source in [
        (lichess_user("Liz", link="lizc"), "arena:a"),
        (lichess_user("Old", link="oldc", seen=SEEN_2025_MS), "arena:a"),
        (lichess_user("Ghost", link="nobody"), "arena:a"),
        (lichess_user("Twin", link="known"), "arena:a"),
        (lichess_user("Plain"), "arena:a"),
    ]:
        users.add(survey.lichess_row(user, source))

    chesscom_store = survey.KeyedJsonl(tmp_path / "stats.jsonl")
    chesscom_store.add(survey.chesscom_row(FakeChessCom({"known": chesscom_stats(1000, 1100)}), "known"))

    client = FakeChessCom({"lizc": chesscom_stats(1400, 1500), "oldc": chesscom_stats()})
    linked = survey.KeyedJsonl(tmp_path / "linked.jsonl", key="lichess_username")
    assert survey.survey_linked(client, users, chesscom_store, linked) == 3

    assert sorted(client.calls) == ["lizc", "nobody"]  # cached and dormant ones not requested
    rows = {row["lichess_username"]: row for row in read_rows(tmp_path / "linked.jsonl")}
    assert set(rows) == {"liz", "ghost", "twin"}

    liz = rows["liz"]
    assert set(liz) == {"lichess_username", "chesscom_username", "fetched_at",
                        "chesscom_found", "lichess", "chesscom"}
    assert liz["chesscom_username"] == "lizc" and liz["chesscom_found"] is True
    assert set(liz["lichess"]) == {"blitz", "rapid", "seen_at"}
    assert liz["lichess"]["seen_at"] == SEEN_2026_MS
    assert liz["chesscom"]["blitz"]["rating"] == 1400
    assert rows["ghost"]["chesscom_found"] is False
    assert rows["ghost"]["chesscom"] == {"blitz": None, "rapid": None}
    assert rows["twin"]["chesscom"]["blitz"]["rating"] == 1000

    again = survey.KeyedJsonl(tmp_path / "linked.jsonl", key="lichess_username")
    assert survey.survey_linked(client, users, chesscom_store, again) == 0


# -- end to end --------------------------------------------------------------


def test_collect_survey_end_to_end_and_resume(data_root):
    write_chesscom_crawl(names_games=[("Liz", "Bob")], names_snapshots=[("Carol", "2026-05")])
    chesscom = FakeChessCom({name: chesscom_stats() for name in ("liz", "bob", "carol", "lizc")})
    lichess = FakeLichess(
        [lichess_user("Liz", link="LizC"), lichess_user("Zed")], arena_players=["Liz", "Zed"]
    )
    out = data_root / "conversion"

    summary = survey.collect_survey(
        platforms=("chesscom", "lichess"), max_chesscom_requests=2, max_lichess_users=10,
        out_dir=out, chesscom_client=chesscom, lichess_client=lichess, bulk_interval=0, read_archives=False,
    )
    assert summary["chesscom_candidates"] == 3
    assert summary["chesscom_added"] == 2
    assert summary["lichess_added"] == 2
    assert summary["linked_added"] == 1
    assert summary["chesscom_stats_rows"] == 2
    assert summary["lichess_users_rows"] == 2
    assert summary["lichess_with_chesscom_link"] == 1
    assert summary["link_rate"] == 0.5
    assert summary["linked_accounts_rows"] == 1 and summary["linked_chesscom_found"] == 1

    second = survey.collect_survey(
        platforms=("chesscom", "lichess"), max_chesscom_requests=5, max_lichess_users=10,
        out_dir=out, chesscom_client=chesscom, lichess_client=lichess, bulk_interval=0, read_archives=False,
    )
    assert second["chesscom_added"] == 1  # only the third candidate was left
    assert second["lichess_added"] == 0 and second["linked_added"] == 0
    assert len(read_rows(out / survey.CHESSCOM_STATS_FILE)) == 3


def test_collect_survey_single_platform_skips_linking(data_root):
    lichess = FakeLichess([lichess_user("Liz", link="lizc")], arena_players=["Liz"])
    summary = survey.collect_survey(
        platforms=("lichess",), max_lichess_users=10, out_dir=data_root / "c",
        lichess_client=lichess, bulk_interval=0, read_archives=False,
    )
    assert summary["lichess_added"] == 1
    assert "linked_added" not in summary and "chesscom_added" not in summary
    assert summary["linked_accounts_rows"] == 0


def test_a_truncated_line_does_not_break_resuming(tmp_path):
    path = tmp_path / "stats.jsonl"
    path.write_text(json.dumps({"username": "a"}) + "\n" + '{"username": "b", "fou', encoding="utf-8")
    store = survey.KeyedJsonl(path)
    assert "a" in store and "b" not in store
    store.add({"username": "b"})
    assert [row["username"] for row in survey.KeyedJsonl(path).read_all()] == ["a", "b"]


# -- chess.com archive opponents ---------------------------------------------


def archive_game(white, black, *, rated=True, time_class="blitz", rules="chess"):
    return {"rated": rated, "time_class": time_class, "rules": rules,
            "white": {"username": white}, "black": {"username": black}}




class FakeChessComWithArchives(FakeChessCom):
    """Serves stats and archives. `archives` maps a hub, or (hub, "MM"), to its games."""

    def __init__(self, known, archives):
        super().__init__(known)
        self.archives = archives
        self.archive_calls = []

    def get(self, path, *, params=None):
        parts = path.split("/")
        if len(parts) == 5 and parts[2] == "games":
            self.archive_calls.append(path)
            self.request_count += 1
            hub, month = parts[1], parts[4]
            return {"games": self.archives.get((hub, month), self.archives.get(hub, []))}
        return super().get(path, params=params)


MONTHS = [config.YearMonth(2026, month) for month in (3, 6, 9)]


def test_opponents_by_cadence_keeps_only_rated_blitz_and_rapid_chess():
    games = [
        archive_game("Hub", "Bob"),
        archive_game("Cy", "hub", time_class="rapid"),
        archive_game("Hub", "Eve", rated=False),
        archive_game("Hub", "Gus", time_class="bullet"),
        archive_game("Hub", "Ivy", time_class="daily"),
        archive_game("Hub", "Kim", rules="chess960"),
        archive_game("Ann", "Bob"),  # the hub is not in this one
        archive_game("Hub", "Bob"),
    ]
    assert archive_module.opponents_by_cadence(games, "hub") == {"blitz": ["bob"], "rapid": ["cy"]}


def test_hub_months_is_deterministic_and_covers_every_month():
    order = archive_module.hub_months("rajsen1968", MONTHS)
    assert order == archive_module.hub_months("rajsen1968", MONTHS)
    assert sorted(order) == MONTHS
    assert len({archive_module.hub_months(f"hub{i}", MONTHS)[0] for i in range(30)}) == 3


def test_read_hubs_takes_both_sides_and_visited_accounts(tmp_path):
    games = tmp_path / "games.jsonl"
    games.write_text(
        json.dumps({"white_username": "Ann", "black_username": "bob"}) + "\n{truncated",
        encoding="utf-8",
    )
    visited = tmp_path / "visited.json"
    visited.write_text(json.dumps(["Cat", "ann"]), encoding="utf-8")
    hubs = archive_module.read_hubs(games, visited)
    assert sorted(hubs) == ["ann", "bob", "cat"]
    assert hubs == archive_module.read_hubs(games, visited)
    assert archive_module.read_hubs(tmp_path / "no.jsonl", tmp_path / "no.json") == []


def test_collect_opponents_caps_each_cadence_per_hub(tmp_path):
    games = (
        [archive_game("hub", f"b{i:02d}", time_class="blitz") for i in range(20)]
        + [archive_game(f"r{i:02d}", "hub", time_class="rapid") for i in range(4)]  # only 4
    )
    client = FakeChessComWithArchives({}, {"hub": games})
    opponents = survey.KeyedJsonl(tmp_path / "opp.jsonl")
    done = survey.IdList(tmp_path / "done.txt")

    added = archive_module.collect_opponents(client, ["hub"], opponents, done, lambda n: False, MONTHS, 100)
    rows = list(opponents.read_all())
    blitz = [row for row in rows if row["via_cadence"] == "blitz"]
    rapid = [row for row in rows if row["via_cadence"] == "rapid"]
    assert added == len(rows) == 10
    assert len(blitz) == archive_module.MAX_PER_CADENCE  # capped
    assert len(rapid) == 4  # not topped up from the blitz side
    assert {row["via"] for row in rows} == {"hub"}
    assert all(row["username"].startswith("b") for row in blitz)
    assert len(client.archive_calls) == 1  # one month was enough


def test_collect_opponents_skips_empty_months_and_known_accounts(tmp_path):
    first = archive_module.hub_months("hub", MONTHS)[0]
    fetched = {"seen"}
    client = FakeChessComWithArchives({}, {
        ("hub", f"{first.month:02d}"): [],
        "hub": [archive_game("hub", "seen"), archive_game("hub", "fresh")],
    })
    opponents = survey.KeyedJsonl(tmp_path / "opp.jsonl")
    done = survey.IdList(tmp_path / "done.txt")
    archive_module.collect_opponents(client, ["hub"], opponents, done, fetched.__contains__, MONTHS, 100)
    assert [row["username"] for row in opponents.read_all()] == ["fresh"]
    assert len(client.archive_calls) == 2  # the empty month, then the next one


def test_collect_opponents_reads_many_hubs_until_target_and_resumes(tmp_path):
    archives_ = {f"h{i}": [archive_game(f"h{i}", f"o{i}_{j}") for j in range(3)] for i in range(10)}
    client = FakeChessComWithArchives({}, archives_)
    opponents = survey.KeyedJsonl(tmp_path / "opp.jsonl")
    done = survey.IdList(tmp_path / "done.txt")
    hubs = list(archives_)

    added = archive_module.collect_opponents(client, hubs, opponents, done, lambda n: False, MONTHS, 7)
    assert added == 9  # three hubs of three; the target of 7 is only checked between hubs
    calls = len(client.archive_calls)
    added = archive_module.collect_opponents(client, hubs, opponents, done, lambda n: False, MONTHS, 7)
    assert added == 0 and len(client.archive_calls) == calls  # 9 already waiting
    archive_module.collect_opponents(client, hubs, opponents, done, lambda n: False, MONTHS, 12)
    assert len(client.archive_calls) == calls + 1  # continues with the next hub only


def test_pending_opponents_ignores_rows_of_the_earlier_harvest(tmp_path):
    opponents = survey.KeyedJsonl(tmp_path / "opp.jsonl")
    opponents.add({"username": "old", "via": "x"})
    opponents.add({"username": "new", "via": "hub", "via_cadence": "rapid"})
    opponents.add({"username": "done", "via": "hub", "via_cadence": "blitz"})
    pending = archive_module.pending_opponents(opponents, {"done"}.__contains__)
    assert pending == [{"username": "new", "via": "hub", "via_cadence": "rapid"}]


def test_collect_survey_records_hub_and_cadence_on_stats_rows(data_root):
    write_chesscom_crawl(names_games=[("Liz", "Bob")])
    known = {name: chesscom_stats() for name in ("liz", "bob", "opp1", "opp2")}
    client = FakeChessComWithArchives(known, {
        "liz": [archive_game("liz", "Opp1"), archive_game("Opp2", "liz", time_class="rapid")],
    })
    out = data_root / "conversion"

    summary = survey.collect_survey(
        platforms=("chesscom",), max_chesscom_requests=10, out_dir=out, chesscom_client=client
    )
    rows = {row["username"]: row for row in read_rows(out / survey.CHESSCOM_STATS_FILE)}
    assert set(rows) == {"liz", "bob", "opp1", "opp2"}
    assert rows["liz"]["source"] == "crawl" and "via" not in rows["liz"]
    assert rows["opp1"]["source"] == "chesscom_archive_opponent"
    assert (rows["opp1"]["via"], rows["opp1"]["via_cadence"]) == ("liz", "blitz")
    assert (rows["opp2"]["via"], rows["opp2"]["via_cadence"]) == ("liz", "rapid")
    assert summary["chesscom_added"] == 4
    calls = len(client.archive_calls)
    assert calls == 4  # liz: one month; bob has no games, so all three months were tried

    again = survey.collect_survey(
        platforms=("chesscom",), max_chesscom_requests=10, out_dir=out, chesscom_client=client
    )
    assert again["chesscom_added"] == 0
    assert len(client.archive_calls) == calls  # every hub is read once and remembered


def test_collect_survey_cap_bounds_stats_requests(data_root):
    write_chesscom_crawl(names_games=[("Liz", "Bob")])
    known = {name: chesscom_stats() for name in [f"o{i}" for i in range(20)] + ["liz", "bob"]}
    client = FakeChessComWithArchives(known, {"liz": [archive_game("liz", f"o{i}") for i in range(20)]})
    summary = survey.collect_survey(
        platforms=("chesscom",), max_chesscom_requests=5, out_dir=data_root / "c",
        chesscom_client=client,
    )
    assert summary["chesscom_added"] == 5
    assert len(client.calls) == 5
