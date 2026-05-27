from __future__ import annotations

import time

import torch
from az.config import Config
from az.network.resnet import AlphaZeroResNet


def main():
    cfg = Config()
    model = AlphaZeroResNet(cfg).cuda() if torch.cuda.is_available() else AlphaZeroResNet(cfg)
    device = next(model.parameters()).device
    b = 64
    import az._az_core as core

    x = torch.randn(b, core.ENCODING_CHANNELS, 8, 8, device=device)
    model.eval()
    for _ in range(5):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"Batch {b}: {n} forwards in {dt:.3f}s ({n*b/dt:.0f} positions/s)")


if __name__ == "__main__":
    main()
