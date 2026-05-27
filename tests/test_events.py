from az.ipc.events import GameFinished, IterationComplete, MovePlayed


def test_move_played_has_game_id():
    mv = MovePlayed(fen="start", uci="e2e4", from_sq=12, to_sq=28, game_id=2)
    assert mv.game_id == 2


def test_game_finished_has_game_id():
    gf = GameFinished(result="1-0", plies=40, examples_count=40, game_id=4)
    assert gf.game_id == 4


def test_iteration_complete():
    ic = IterationComplete(iteration=1, games_finished=5, train_steps=100, brain_path="anomaly.pt")
    assert ic.games_finished == 5
