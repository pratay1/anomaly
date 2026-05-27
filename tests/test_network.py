import torch
from az.config import Config
from az.network.resnet import AlphaZeroResNet, az_loss


def test_resnet_forward():
    cfg = Config()
    model = AlphaZeroResNet(cfg)
    x = torch.randn(4, cfg.encoding_channels, 8, 8)
    p, v = model(x)
    assert p.shape == (4, cfg.policy_size)
    assert v.shape == (4,)


def test_loss():
    cfg = Config()
    model = AlphaZeroResNet(cfg)
    x = torch.randn(2, cfg.encoding_channels, 8, 8)
    tp = torch.softmax(torch.randn(2, cfg.policy_size), dim=-1)
    tv = torch.tensor([1.0, -1.0])
    p, v = model(x)
    pl, vl, total = az_loss(p, v, tp, tv)
    assert total.ndim == 0
