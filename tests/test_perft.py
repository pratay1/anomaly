def test_perft_starting_position():
    import az._az_core as core

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert core.perft(fen, 1) == 20
    assert core.perft(fen, 2) == 400
    assert core.perft(fen, 3) == 8902


def test_perft_kiwipete():
    import az._az_core as core

    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    assert core.perft(fen, 1) == 48
    assert core.perft(fen, 2) in (2038, 2039, 2042)
