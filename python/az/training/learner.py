from __future__ import annotations

import threading

import torch
import torch.optim as optim

from az.config import Config
from az.ipc.events import TrainStep
from az.network.resnet import AlphaZeroResNet, az_loss
from az.training.replay_buffer import ReplayBuffer

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    QObject = object  # type: ignore

    def pyqtSignal(*args, **kwargs):  # type: ignore
        return None


class Learner(QObject):
    train_step = pyqtSignal(object)

    def __init__(
        self,
        model: AlphaZeroResNet,
        buffer: ReplayBuffer,
        cfg: Config,
        stop_event: threading.Event,
    ):
        super().__init__()
        self.model = model
        self.buffer = buffer
        self.cfg = cfg
        self.stop_event = stop_event
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu"
        )
        self.model.to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=cfg.learning_rate(0),
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        self.global_step = 0

    def train_once(self) -> TrainStep | None:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        import az._az_core as core

        states, policies, values = self.buffer.sample(self.cfg.batch_size)
        b = states.shape[0]
        x = torch.from_numpy(states).view(b, core.ENCODING_CHANNELS, 8, 8).to(self.device)
        tp = torch.from_numpy(policies).to(self.device)
        tv = torch.from_numpy(values).to(self.device)

        self.model.train()
        lr = self.cfg.learning_rate(self.global_step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        self.optimizer.zero_grad()
        logits, pred_v = self.model(x)
        pl, vl, total = az_loss(logits, pred_v, tp, tv)
        total.backward()
        self.optimizer.step()
        self.global_step += 1

        step = TrainStep(
            step=self.global_step,
            policy_loss=float(pl.item()),
            value_loss=float(vl.item()),
            total_loss=float(total.item()),
            lr=lr,
        )
        self.train_step.emit(step)
        return step

    def run_loop(self) -> None:
        while not self.stop_event.is_set():
            self.train_once()
