from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
import torch.optim as optim

from az.config import Config
from az.ipc.events import TrainStep
from az.network.resnet import AlphaZeroResNet, az_loss
from az.training.replay_buffer import ReplayBuffer
from az.training.stockfish_critic import StockfishCriticBuffer

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    QObject = object  # type: ignore

    def pyqtSignal(*args, **kwargs):  # type: ignore
        return None


class CentralLearner(QObject):
    train_step = pyqtSignal(object)

    def __init__(
        self,
        model: AlphaZeroResNet,
        buffer: ReplayBuffer,
        cfg: Config,
        stop_event: threading.Event,
        model_lock: threading.Lock | None = None,
        critic_buffer: StockfishCriticBuffer | None = None,
    ):
        super().__init__()
        self.model = model
        self.buffer = buffer
        self.cfg = cfg
        self.stop_event = stop_event
        self.critic_buffer = critic_buffer
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
        self._warmup_steps = 1000
        self._model_lock = model_lock or threading.Lock()

    def _warmup_lr(self, step: int, base_lr: float) -> float:
        if step < self._warmup_steps:
            return base_lr * (step + 1) / self._warmup_steps
        return base_lr

    def _compute_grads(
        self,
        states: torch.Tensor,
        policies: torch.Tensor,
        values: torch.Tensor,
        model: AlphaZeroResNet,
    ) -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        logits, pred_v = model(states)
        pl, vl, total = az_loss(logits, pred_v, policies, values)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().clone())
            else:
                grads.append(torch.zeros_like(p))
        return grads

    def _compute_critic_grads(
        self,
        states: torch.Tensor,
        sf_move_indices: torch.Tensor,
        model: AlphaZeroResNet,
    ) -> list[torch.Tensor]:
        """Cross-entropy loss: push the policy head toward Stockfish's preferred move."""
        model.zero_grad(set_to_none=True)
        logits, _ = model(states)
        loss = F.cross_entropy(logits, sf_move_indices) * self.cfg.stockfish_critic_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        grads = []
        for p in model.parameters():
            grads.append(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p))
        return grads, float(loss.item())

    def _average_grads(self, grad_lists: list[list[torch.Tensor]]) -> list[torch.Tensor]:
        if not grad_lists:
            return []
        n = len(grad_lists)
        avg = [torch.zeros_like(g) for g in grad_lists[0]]
        for gl in grad_lists:
            for i, g in enumerate(gl):
                avg[i] += g / n
        return avg

    def _apply_grads(
        self, grads: list[torch.Tensor], critic_loss: float = 0.0
    ) -> TrainStep:
        import az._az_core as core

        self.optimizer.zero_grad(set_to_none=True)
        for p, g_ in zip(self.model.parameters(), grads):
            p.grad = g_
        self.optimizer.step()
        self.global_step += 1
        raw_lr = self.cfg.learning_rate(self.global_step)
        lr = self._warmup_lr(self.global_step, raw_lr)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.model.eval()
        with torch.no_grad():
            bs = min(self.cfg.batch_size, len(self.buffer))
            states, policies, values = self.buffer.sample(bs)
            b = states.shape[0]
            x = torch.from_numpy(states).view(b, core.ENCODING_CHANNELS, 8, 8).to(self.device)
            tp = torch.from_numpy(policies).to(self.device)
            tv = torch.from_numpy(values).to(self.device)
            logits, pred_v = self.model(x)
            pl, vl, total = az_loss(logits, pred_v, tp, tv)

        step = TrainStep(
            step=self.global_step,
            policy_loss=float(pl.item()),
            value_loss=float(vl.item()),
            total_loss=float(total.item()),
            lr=lr,
            critic_loss=critic_loss,
        )
        self.train_step.emit(step)
        return step

    def train_iteration(self, num_steps: int | None = None) -> list[TrainStep]:
        """Run num_steps gradient-averaged SGD steps after a self-play barrier."""
        steps = num_steps if num_steps is not None else self.cfg.train_steps_per_iteration
        results: list[TrainStep] = []
        if len(self.buffer) < self.cfg.batch_size:
            return results

        import az._az_core as core

        n_workers = max(1, min(self.cfg.gradient_avg_workers, self.cfg.batch_size))
        sub_batch = max(1, self.cfg.batch_size // n_workers)

        with self._model_lock:
            self.model.train()
            for _ in range(steps):
                if self.stop_event.is_set():
                    break
                if len(self.buffer) < self.cfg.batch_size:
                    break

                raw_lr = self.cfg.learning_rate(self.global_step)
                lr = self._warmup_lr(self.global_step, raw_lr)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                grad_lists: list[list[torch.Tensor]] = []

                def worker_task(worker_id: int) -> list[torch.Tensor]:
                    local_model = copy.deepcopy(self.model)
                    local_model.to(self.device)
                    local_model.train()
                    bs = min(sub_batch, len(self.buffer))
                    states, policies, values = self.buffer.sample(bs)
                    b = states.shape[0]
                    x = (
                        torch.from_numpy(states)
                        .view(b, core.ENCODING_CHANNELS, 8, 8)
                        .to(self.device)
                    )
                    tp = torch.from_numpy(policies).to(self.device)
                    tv = torch.from_numpy(values).to(self.device)
                    return self._compute_grads(x, tp, tv, local_model)

                with ThreadPoolExecutor(
                    max_workers=n_workers, thread_name_prefix="GradAvg"
                ) as pool:
                    futures = [pool.submit(worker_task, i) for i in range(n_workers)]
                    for fut in as_completed(futures):
                        try:
                            grad_lists.append(fut.result())
                        except Exception:
                            pass

                if not grad_lists:
                    continue
                avg_grads = self._average_grads(grad_lists)

                # Mix in Stockfish Critic gradients when available.
                critic_loss_val = 0.0
                if (
                    self.critic_buffer is not None
                    and len(self.critic_buffer) >= self.cfg.batch_size
                ):
                    try:
                        cr_states, cr_indices = self.critic_buffer.sample(sub_batch)
                        b = cr_states.shape[0]
                        cx = (
                            torch.from_numpy(cr_states)
                            .view(b, core.ENCODING_CHANNELS, 8, 8)
                            .to(self.device)
                        )
                        ci = torch.from_numpy(cr_indices).long().to(self.device)
                        local_critic = copy.deepcopy(self.model)
                        local_critic.to(self.device)
                        local_critic.train()
                        cg, critic_loss_val = self._compute_critic_grads(cx, ci, local_critic)
                        for i, g in enumerate(cg):
                            avg_grads[i] = avg_grads[i] + g
                    except Exception:
                        pass

                results.append(self._apply_grads(avg_grads, critic_loss_val))

        return results

    def train_once(self) -> TrainStep | None:
        """Single gradient-averaged step (for tests)."""
        steps = self.train_iteration(1)
        return steps[0] if steps else None
