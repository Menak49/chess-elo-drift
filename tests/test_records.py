from chess_elo_drift.records import GameRecord, classify_result, count_plies

PGN = (
    '[Event "Live Chess"]\n[White "alice"]\n[Black "bob"]\n\n'
    "1. e4 {[%clk 0:03:00]} 1... e5 {[%clk 0:02:59.9]} "
    "2. Nf3 {[%clk 0:02:13.7]} 2... Nc6 {[%clk 0:01:04.1]} "
    "3. Bb5 {[%clk 0:00:10.5]} 1-0\n"
)


def test_count_plies_ignores_clock_comments():
    """Tenths of a second inside {[%clk ...]} look exactly like move numbers."""
    assert count_plies(PGN) == 5


def test_count_plies_without_header_block():
    assert count_plies("1. e4 1... e5 2. d4 *") == 3


def test_count_plies_of_empty_movetext():
    assert count_plies('[White "a"]\n\n1-0') == 0


def test_classify_result_maps_every_family():
    assert classify_result("win") == "win"
    assert classify_result("agreed") == "draw"
    assert classify_result("stalemate") == "draw"
    assert classify_result("timevsinsufficient") == "draw"
    assert classify_result("resigned") == "loss"
    assert classify_result("checkmated") == "loss"
    assert classify_result("timeout") == "loss"


def _record(**overrides) -> GameRecord:
    base = dict(
        game_id="g1",
        url="https://example.invalid/1",
        platform="chesscom",
        year=2018,
        month="2018-03",
        time_class="blitz",
        time_control="300",
        ply_count=40,
        white_username="alice",
        white_rating=1500,
        white_result="win",
        black_username="bob",
        black_rating=1480,
        black_result="loss",
        pgn=PGN,
        reported_accuracy_white=88.5,
        reported_accuracy_black=None,
    )
    base.update(overrides)
    return GameRecord(**base)


def test_sides_expose_each_players_point_of_view():
    white, black = _record().sides()

    assert (white.username, white.rating, white.result) == ("alice", 1500, "win")
    assert white.opponent_rating == 1480
    assert white.reported_accuracy == 88.5

    assert (black.username, black.rating, black.result) == ("bob", 1480, "loss")
    assert black.opponent_rating == 1500
    assert black.reported_accuracy is None


def test_record_survives_a_serialisation_round_trip():
    record = _record()
    assert GameRecord.from_dict(record.to_dict()) == record


def test_from_dict_ignores_unknown_columns():
    payload = _record().to_dict() | {"future_field": 1}
    assert GameRecord.from_dict(payload).game_id == "g1"


def test_the_engine_refuses_a_game_that_starts_from_a_set_up_position():
    from chess_elo_drift.engine.evaluator import _read_mainline

    set_up = (
        '[SetUp "1"]\n'
        '[FEN "r1bqkbnr/pppp1ppp/2n5/8/3pP3/2P5/PP3PPP/RNBQKBNR w KQkq - 0 1"]\n\n'
        "1. cxd4 Nxd4 *"
    )
    assert _read_mainline(set_up) is None
    assert _read_mainline("1. e4 e5 2. Nf3 *") is not None
