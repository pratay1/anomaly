from __future__ import annotations

import threading

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int, policy_size: int):
        self.capacity = capacity
        self.state_size = state_size
        self.policy_size = policy_size
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.policies = np.zeros((capacity, policy_size), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.size = 0
        self.ptr = 0
        self._lock = threading.Lock()

    def add_batch(self, examples: list) -> None:
        with self._lock:
            for ex in examples:
                s = np.asarray(ex.state, dtype=np.float32)
                p = np.asarray(ex.policy, dtype=np.float32)
                v = float(ex.value)
                self.states[self.ptr] = s
                self.policies[self.ptr] = p
                self.values[self.ptr] = v
                self.ptr = (self.ptr + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            if self.size == 0:
                raise RuntimeError("Replay buffer empty")
            idx = np.random.randint(0, self.size, size=batch_size)
            return self.states[idx].copy(), self.policies[idx].copy(), self.values[idx].copy()

    def __len__(self) -> int:
        with self._lock:
            return self.size
