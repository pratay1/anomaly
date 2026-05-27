from __future__ import annotations

import queue
import random
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from types import SimpleNamespace

import az._az_core as core
from az.config import Config
from az.ipc.events import GameFinished, MovePlayed
from az.training.replay_buffer import ReplayBuffer
from az.training.stockfish import StockfishEngine
from az.training.stockfish_paths import stockfish_path_error

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    QObject = object  # type: ignore

    def pyqtSignal(*args, **kwargs):  # type: ignore
        return None


def _sq_uci(sq: int) -> str:
    return chr(ord("a") + (sq % 8)) + str((sq // 8) + 1)


def move_to_uci(m: core.Move) -> str:
    uci = _sq_uci(m.from_sq) + _sq_uci(m.to_sq)
    pmap = {
        core.PieceType.Knight: "n",
        core.PieceType.Bishop: "b",
        core.PieceType.Rook: "r",
        core.PieceType.Queen: "q",
    }
    if m.promotion in pmap:
        uci += pmap[m.promotion]
    return uci


def play_one_game(
    queue_inf: core.InferenceQueue,
    cfg: Config,
    stop_event: threading.Event,
    game_id: int = 0,
    rng: random.Random | None = None,
    event_sink: queue.Queue | None = None,
    game_seq: int = 0,
    stockfish: StockfishEngine | None = None,
) -> list:
    """Play one self-play game. Events go to event_sink (thread-safe), not Qt signals."""
    rng = rng or random.Random()
    mcts_cfg = cfg.to_mcts_config()
    board = core.Board()
    mcts = core.MCTS(queue_inf, mcts_cfg)
    trajectory: list[tuple] = []
    moves_uci: list[str] = []
    use_stockfish = cfg.training_opponent == "stockfish" and stockfish is not None
    mcts_color = core.Color.White if game_seq % 2 == 0 else core.Color.Black
    mcts_ply = 0

    for ply in range(cfg.max_game_length):
        if stop_event.is_set():
            break
        res = board.result()
        if res != core.GameResult.Ongoing:
            break

        side = board.side_to_move()
        if use_stockfish and side != mcts_color:
            assert stockfish is not None
            mv = stockfish.choose_move(board)
            uci = move_to_uci(mv)
            move_evt = MovePlayed(
                fen=board.fen(),
                uci=uci,
                from_sq=mv.from_sq,
                to_sq=mv.to_sq,
                game_id=game_id,
                visits=[],
            )
            if event_sink is not None:
                event_sink.put(("move_played", move_evt))
            moves_uci.append(uci)
            board.make_move(mv)
            mcts = core.MCTS(queue_inf, mcts_cfg)
            continue

        temp = 1.0 if mcts_ply < cfg.temperature_moves else 0.1
        mcts_ply += 1
        pi = mcts.run(board, temp)
        fen = board.fen()
        visit_dicts: list[dict] = []
        if cfg.emit_selfplay_visits:
            visits = mcts.root_visits(board)
            visit_dicts = [
                {"move_index": v.move_index, "N": v.N, "Q": v.Q, "P": v.P} for v in visits
            ]
            if event_sink is not None and visit_dicts:
                event_sink.put(("mcts_visits", fen, visit_dicts))
        legal = core.legal_move_indices(board)
        if not legal:
            break
        if temp < 0.5:
            idx = max(legal, key=lambda i: pi[i])
        else:
            weights = [max(pi[i], 1e-8) for i in legal]
            idx = rng.choices(legal, weights=weights)[0]

        mv = core.index_to_move(board, idx)
        uci = move_to_uci(mv)
        move_evt = MovePlayed(
            fen=fen,
            uci=uci,
            from_sq=mv.from_sq,
            to_sq=mv.to_sq,
            game_id=game_id,
            visits=visit_dicts,
        )
        if event_sink is not None:
            event_sink.put(("move_played", move_evt))
        moves_uci.append(uci)
        trajectory.append((core.encode(board), list(pi), board.side_to_move()))
        board.make_move(mv)
        mcts.advance_root(idx)

    res = board.result()
    if res == core.GameResult.Ongoing:
        res = core.GameResult.Draw

    examples = []
    for state, pi, stm in trajectory:
        v = 0.0
        if res == core.GameResult.Draw:
            v = 0.0
        elif res == core.GameResult.WhiteWin:
            v = 1.0 if stm == core.Color.White else -1.0
        else:
            v = 1.0 if stm == core.Color.Black else -1.0
        examples.append(SimpleNamespace(state=state, policy=pi, value=v))

    result = "1/2-1/2"
    if examples:
        if examples[-1].value > 0:
            result = "1-0" if res == core.GameResult.WhiteWin else "0-1"
        elif examples[-1].value < 0:
            result = "0-1" if res == core.GameResult.BlackWin else "1-0"

    finished = GameFinished(
        result=result,
        plies=len(examples),
        examples_count=len(examples),
        game_id=game_id,
        moves_uci=moves_uci,
    )
    if event_sink is not None:
        event_sink.put(("game_finished", finished))
    return examples


class SelfPlayWorker(QObject):
    move_played = pyqtSignal(object)
    game_finished = pyqtSignal(object)
    mcts_visits = pyqtSignal(str, list)

    def __init__(
        self,
        queue_inf: core.InferenceQueue,
        buffer: ReplayBuffer,
        cfg: Config,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.queue = queue_inf
        self.buffer = buffer
        self.cfg = cfg
        self.stop_event = stop_event

    def play_one_game(self, game_id: int = 0, rng: random.Random | None = None) -> list:
        sink: queue.Queue = queue.Queue()

        def drain_sink() -> list:
            events = []
            while True:
                try:
                    events.append(sink.get_nowait())
                except queue.Empty:
                    break
            return events

        examples = play_one_game(
            self.queue, self.cfg, self.stop_event, game_id, rng, event_sink=sink
        )
        for kind, *payload in drain_sink():
            if kind == "move_played":
                self.move_played.emit(payload[0])
            elif kind == "game_finished":
                self.game_finished.emit(payload[0])
            elif kind == "mcts_visits":
                self.mcts_visits.emit(payload[0], payload[1])
        return examples

    def run_loop(self) -> None:

        game_id = 0
        while not self.stop_event.is_set():
            examples = self.play_one_game(game_id=game_id)
            self.buffer.add_batch(examples)
            game_id += 1


class ParallelSelfPlayPool:
    """Parallel self-play workers; events go to outbound_events (never emit Qt from workers)."""

    def __init__(
        self,
        queue_inf: core.InferenceQueue,
        buffer: ReplayBuffer,
        cfg: Config,
        stop_event: threading.Event,
        outbound_events: queue.Queue | None = None,
    ):
        self.queue = queue_inf
        self.buffer = buffer
        self.cfg = cfg
        self.stop_event = stop_event
        self.outbound_events = outbound_events if outbound_events is not None else queue.Queue()
        self._game_seq = 0
        self._game_seq_lock = threading.Lock()
        self._stockfish: StockfishEngine | None = None

    def _next_game_seq(self) -> int:
        with self._game_seq_lock:
            seq = self._game_seq
            self._game_seq += 1
            return seq

    def _close_stockfish(self) -> None:
        if self._stockfish is not None:
            self._stockfish.close()
            self._stockfish = None

    def restart_stockfish(self) -> None:
        """Recycle the UCI engine after a crash without stopping training."""
        self._close_stockfish()

    def set_training_opponent(self, opponent: str) -> None:
        """Switch between self-play and Stockfish; releases the engine when leaving Stockfish."""
        if opponent not in ("self", "stockfish"):
            raise ValueError(f"training_opponent must be 'self' or 'stockfish', got {opponent!r}")
        if self.cfg.training_opponent == opponent:
            return
        self.cfg.training_opponent = opponent
        if opponent != "stockfish":
            self._close_stockfish()

    def _stockfish_engine(self) -> StockfishEngine:
        if self.cfg.training_opponent != "stockfish":
            raise RuntimeError("Stockfish engine requested outside Stockfish training mode")
        err = stockfish_path_error(self.cfg.stockfish_path)
        if err:
            raise FileNotFoundError(err)
        if self._stockfish is None:
            with self._game_seq_lock:
                if self._stockfish is None:
                    self._stockfish = StockfishEngine(self.cfg)
        return self._stockfish

    def _forward_event(self, item: tuple) -> None:
        self.outbound_events.put(item)

    def _drain_events(self, event_queue: queue.Queue) -> None:
        while True:
            try:
                self._forward_event(event_queue.get_nowait())
            except queue.Empty:
                break

    def run_iteration(self, num_games: int | None = None) -> list:
        """Run self-play games; Stockfish mode uses one serial game (no thread pool)."""
        stockfish_mode = self.cfg.training_opponent == "stockfish"
        n = 1 if stockfish_mode else (num_games if num_games is not None else self.cfg.num_workers)
        if self.stop_event.is_set():
            return []

        stockfish_engine: StockfishEngine | None = None
        if stockfish_mode:
            stockfish_engine = self._stockfish_engine()

        all_examples: list = []
        event_queue: queue.Queue = queue.Queue()

        def play_game(game_id: int) -> list:
            if self.stop_event.is_set():
                return []
            game_seq = self._next_game_seq()
            rng = random.Random(game_seq + game_id)
            return play_one_game(
                self.queue,
                self.cfg,
                self.stop_event,
                game_id=game_id,
                rng=rng,
                event_sink=event_queue,
                game_seq=game_seq,
                stockfish=stockfish_engine,
            )

        if stockfish_mode:
            for attempt in range(3):
                try:
                    examples = play_game(0)
                    self.buffer.add_batch(examples)
                    all_examples.extend(examples)
                    self._drain_events(event_queue)
                    return all_examples
                except Exception:
                    self.restart_stockfish()
                    stockfish_engine = self._stockfish_engine()
                    if attempt == 2:
                        self._drain_events(event_queue)
                        return all_examples
            return all_examples

        lock = threading.Lock()

        def play_game_parallel(game_id: int) -> list:
            examples = play_game(game_id)
            with lock:
                self.buffer.add_batch(examples)
            return examples

        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="SelfPlay") as pool:
            futures = [pool.submit(play_game_parallel, gid) for gid in range(n)]
            pending = set(futures)
            while pending:
                self._drain_events(event_queue)
                done, pending = wait(
                    pending, timeout=0.05, return_when=FIRST_COMPLETED
                )
                for fut in done:
                    if self.stop_event.is_set():
                        break
                    try:
                        all_examples.extend(fut.result())
                    except Exception:
                        continue
        self._drain_events(event_queue)
        return all_examples
