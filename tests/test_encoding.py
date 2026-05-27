def test_encode_length():
    import az._az_core as core

    board = core.Board()
    enc = core.encode(board)
    assert len(enc) == core.ENCODING_CHANNELS * 64


def test_legal_indices_match_movegen():
    import az._az_core as core

    board = core.Board()
    legal = core.legal_move_indices(board)
    moves = board.generate_legal_moves()
    assert len(legal) == len(moves)
    for m in moves:
        idx = core.move_to_index(board, m)
        assert idx in legal
