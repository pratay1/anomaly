from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from az.config import Config


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class AlphaZeroResNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        in_ch = cfg.encoding_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(c) for _ in range(cfg.num_res_blocks)]
        )
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(32 * 64, cfg.policy_size)
        self.value_conv = nn.Sequential(
            nn.Conv2d(c, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(64, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, 8, 8)
        h = self.stem(x)
        h = self.blocks(h)
        p = self.policy_conv(h).reshape(x.size(0), -1)
        policy_logits = self.policy_fc(p)
        v = self.value_conv(h).reshape(x.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return policy_logits, value


def az_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    legal_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if legal_masks is not None:
        policy_logits = policy_logits.masked_fill(~legal_masks.bool(), -1e9)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, target_value)
    total = policy_loss + value_loss
    return policy_loss, value_loss, total
