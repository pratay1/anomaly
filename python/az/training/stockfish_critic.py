"""Stockfish Critic Buffer.

For every position where Anomaly plays a move, we also ask Stockfish what it
would play with the same time budget.  The (board-state, sf_move_index) pairs
are stored here and used during training as a supervised signal: cross-entropy
loss between the network's policy logits and a one-hot target peaked at
Stockfish's preferred move.
"""
from __future__ import annotations

import threading

import numpy as np


class StockfishCriticBuffer:
    """Thread-safe ring buffer of (encoded_state, sf_move_index) pairs."""

    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.state_size = state_size
        self._states = np.zeros((capacity, state_size), dtype=np.float16)
        self._move_indices = np.zeros(capacity, dtype=np.int32)
        self._size = 0
        self._ptr = 0
        self._lock = threading.Lock()

    def add(self, state: list | np.ndarray, sf_move_idx: int) -> None:
        with self._lock:
            self._states[self._ptr] = np.asarray(state, dtype=np.float16)
            self._move_indices[self._ptr] = sf_move_idx
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (states float32, sf_move_indices int32)."""
        with self._lock:
            if self._size == 0:
                raise RuntimeError("Critic buffer empty")
            idx = np.random.randint(0, self._size, size=batch_size)
            return (
                self._states[idx].astype(np.float32),
                self._move_indices[idx].copy(),
            )

    def __len__(self) -> int:
        with self._lock:
            return self._size
