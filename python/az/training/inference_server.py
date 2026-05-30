from __future__ import annotations

import logging
import threading
import time

import numpy as np
import torch

import az._az_core as core
from az.config import Config
from az.network.resnet import AlphaZeroResNet

logger = logging.getLogger(__name__)

_CUDA_RECOVERY_COOLDOWN = 5.0  # seconds between recovery attempts
_MAX_CONSECUTIVE_CUDA_FAILURES = 5  # fall back to CPU after this many


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
        self._consecutive_cuda_failures = 0
        self._last_recovery = 0.0

    def reload_weights(self, state_dict: dict | None = None) -> None:
        with self._model_lock:
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
            self.model.eval()
            self._model_version += 1

    def _try_recover_cuda(self) -> bool:
        """Attempt to recover from a CUDA context loss. Returns True on success."""
        now = time.monotonic()
        if now - self._last_recovery < _CUDA_RECOVERY_COOLDOWN:
            return False
        self._last_recovery = now
        self._consecutive_cuda_failures += 1

        if self._consecutive_cuda_failures >= _MAX_CONSECUTIVE_CUDA_FAILURES:
            logger.warning(
                "CUDA failed %d times — falling back to CPU inference",
                self._consecutive_cuda_failures,
            )
            self.device = torch.device("cpu")
            with self._model_lock:
                self.model.to(self.device)
                self.model.eval()
            self._consecutive_cuda_failures = 0
            return True

        logger.warning(
            "CUDA error in InferenceServer (attempt %d/%d) — attempting recovery",
            self._consecutive_cuda_failures,
            _MAX_CONSECUTIVE_CUDA_FAILURES,
        )
        try:
            torch.cuda.empty_cache()
            torch.cuda.init()
            with self._model_lock:
                self.model.to(self.device)
                self.model.eval()
            logger.info("CUDA recovery succeeded")
            self._consecutive_cuda_failures = 0
            return True
        except Exception:
            logger.warning("CUDA recovery failed", exc_info=True)
            return False

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
            except RuntimeError as exc:
                # CUDA errors show up as RuntimeError
                is_cuda = "cuda" in str(exc).lower() or "cudnn" in str(exc).lower()
                if is_cuda:
                    logger.warning("CUDA error in inference: %s", exc)
                    self._try_recover_cuda()
                    time.sleep(0.5)
                else:
                    logger.warning("Runtime error in inference: %s", exc)
                    time.sleep(0.05)
            except Exception as exc:
                logger.debug("Inference error: %s", exc)
                time.sleep(0.05)
                continue
