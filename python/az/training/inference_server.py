from __future__ import annotations

import threading
import time

import numpy as np
import torch

import az._az_core as core
from az.config import Config
from az.network.resnet import AlphaZeroResNet


class InferenceServer(threading.Thread):
    """Drains C++ InferenceQueue and fulfills batched GPU/CPU forwards."""

    def __init__(
        self,
        queue,
        model: AlphaZeroResNet,
        cfg: Config,
        stop_event: threading.Event,
        model_lock: threading.Lock | None = None,
    ):
        super().__init__(daemon=True, name="InferenceServer")
        self.queue = queue
        self.model = model
        self.cfg = cfg
        self.stop_event = stop_event
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self._model_lock = model_lock or threading.Lock()
        self._model_version = 0

    def reload_weights(self, state_dict: dict | None = None) -> None:
        with self._model_lock:
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
            self.model.eval()
            self._model_version += 1

    def run(self) -> None:
        expected_state = core.ENCODING_CHANNELS * 64

        while not self.stop_event.is_set():
            try:
                reqs = self.queue.drain(self.cfg.max_batch, self.cfg.max_wait_us)
                if not reqs:
                    time.sleep(0.001)
                    continue

                state_sizes = [len(r.state) for r in reqs]
                if any(s != expected_state for s in state_sizes):
                    continue

                states = np.stack([np.asarray(r.state, dtype=np.float32) for r in reqs])
                b = states.shape[0]
                x = torch.from_numpy(states).view(b, core.ENCODING_CHANNELS, 8, 8).to(self.device)

                with self._model_lock:
                    with torch.no_grad():
                        logits, values = self.model(x)

                policies = []
                for i in range(b):
                    p = torch.softmax(logits[i], dim=-1).cpu().numpy()
                    policies.append(p.tolist())

                ids = [r.id for r in reqs]
                self.queue.fulfill(ids, policies, values.cpu().tolist())
            except Exception:
                time.sleep(0.05)
                continue
