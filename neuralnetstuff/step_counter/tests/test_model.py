"""The CNN's shape, receptive field and training loop, before any real data (PLAN.md testing strategy)."""

import numpy as np
import pytest
import torch

from stepnet.config import load_config
from stepnet.data.blocks import Block, SessionData
from stepnet.data.windows import WindowDataset, channel_stats, random_rotation, session_arrays
from stepnet.models.cnn import StepCNN
from stepnet.train import detect_peaks, masked_bce, predict_session, train_model


def test_output_keeps_the_time_axis():
    model = StepCNN()
    assert model(torch.randn(2, 6, 256)).shape == (2, 256)
    assert model.receptive_field == 187
    assert sum(p.numel() for p in model.parameters()) < 40_000


def test_receptive_field_by_gradient():
    # d(output[t]) / d(input) is non-zero only within receptive_field/2 of t: that's what
    # "receptive field" means, measured with autograd instead of taken on trust.
    model = StepCNN(dropout=0.0)
    for p in model.parameters():  # make every weight positive so no path cancels out
        p.data = p.data.abs() + 0.01
    x = torch.zeros(1, 6, 400, requires_grad=True)
    model(x)[0, 200].backward()
    touched = torch.nonzero(x.grad.abs().sum(dim=1)[0]).squeeze()
    half = model.receptive_field // 2
    assert touched.min().item() == 200 - half and touched.max().item() == 200 + half


def synthetic_walk(n_strides=40, seed=0):
    """Fake walking: one L-gyro dip + heel-strike impact per second, then 20 s of standing."""
    rng = np.random.default_rng(seed)
    n = (n_strides + 25) * 100
    t = np.arange(n) / 100
    gyr = rng.normal(0, 2, (n, 3))
    acc = np.tile([1.0, 0, 0], (n, 1)) + rng.normal(0, 0.01, (n, 3))
    strides = []
    for k in range(n_strides):
        c = 2 + k + rng.uniform(-0.05, 0.05)
        gyr[:, 2] += -300 * np.exp(-((t - c) / 0.08) ** 2)
        acc[:, 0] += 2.0 * np.exp(-((t - c - 0.25) / 0.02) ** 2)
        strides.append(int(round(c * 100)))
    seq = np.arange(n)
    blocks = [Block("S001", 3, "walk_normal", "walk_normal", "bout", 100, (n_strides + 3) * 100, 2 * n_strides, np.array(strides)),
              Block("S001", 4, "stand", "stand", "hold", (n_strides + 4) * 100, n - 100, 0, np.array([], int))]
    return SessionData("S001", {}, seq, acc, gyr, blocks)


def test_targets_and_mask():
    s = synthetic_walk(5)
    x, target, mask = session_arrays(s, sigma=5.0)
    assert x.shape == (6, len(s.seq))
    assert target[s.blocks[0].strides].min() == pytest.approx(1.0)
    assert mask[:100].sum() == 0 and mask[150] == 1  # before the bout: unlabeled, no loss


def test_rotation_is_a_rotation():
    r = random_rotation(np.random.default_rng(0), 10)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.degrees(np.arccos((np.trace(r) - 1) / 2)) <= 10 + 1e-6


def test_masked_bce_ignores_unmasked_samples():
    logits, target = torch.zeros(1, 4), torch.tensor([[1.0, 0, 0, 0]])
    a = masked_bce(logits, target, torch.tensor([[1.0, 1, 0, 0]]), pos_weight=0)
    b = masked_bce(logits + torch.tensor([[0, 0, 9, 9]]), target, torch.tensor([[1.0, 1, 0, 0]]), pos_weight=0)
    assert a.item() == pytest.approx(b.item())


def test_learns_synthetic_walking_end_to_end():
    # Overfit check of the whole pipeline: train, slide over a session, pick peaks, count.
    cfg = load_config(overrides=["train.epochs=8", "data.windows_per_epoch=600", "model.dropout=0.0"])
    s = synthetic_walk(40)
    model, stats = train_model([s], cfg)
    prob = predict_session(model, stats, s)
    found, _ = detect_peaks(prob, s.seq, threshold=0.5)
    true = s.blocks[0].strides
    hits = sum(np.abs(true - f).min() <= 10 for f in found)
    assert hits >= 38 and len(found) <= 42, (hits, len(found))
