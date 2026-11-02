"""The streaming engine agrees with offline inference and reports what the live view needs."""

import numpy as np
import torch

from stepnet.data.blocks import SessionData
from stepnet.data.windows import channel_stats
from stepnet.infer import LiveEngine
from stepnet.io.protocol import ACCEL_LSB_PER_G, GYRO_LSB_PER_DPS
from stepnet.models.cnn import StepCNN
from stepnet.preprocess import to_body
from stepnet.train import predict_session

CAL = {"config": {"data": {"window": 256}}, "threshold": 0.5, "min_gap_s": 0.3, "sigma_delta2": 0.2,
       "q": {"0.9": 1.8}}


def raw_walk(seconds=20, seed=0):
    """Raw sensor counts: 3 s still (for the gyro bias), then sagittal swings once a second."""
    rng = np.random.default_rng(seed)
    n = seconds * 100
    t = np.arange(n) / 100
    raw = np.zeros((n, 8))
    raw[:, 0] = np.arange(n)
    raw[:, 1] = np.arange(n) * 10
    raw[:, 2] = -ACCEL_LSB_PER_G + rng.normal(0, 5, n)  # S = -x: standing
    swing = np.where(t > 4, -250 * np.sin(np.pi * (t % 1)) ** 8, 0)
    raw[:, 7] = (swing + rng.normal(0, 1, n)) * GYRO_LSB_PER_DPS  # L = +z
    return np.round(raw).astype(np.int64)


def test_streaming_matches_offline_and_reports_everything():
    torch.manual_seed(0)
    model = StepCNN(dropout=0.0).eval()
    raw = raw_walk()
    acc = to_body(raw[:, 2:5] / ACCEL_LSB_PER_G)
    gyr = to_body(raw[:, 5:8] / GYRO_LSB_PER_DPS)
    stats = channel_stats([SessionData("X", {}, raw[:, 0], acc, gyr)])
    engine = LiveEngine([(model, stats)], CAL, accel_cal=None)

    for k in range(0, len(raw), 25):  # like the relay: 25 samples per MQTT message
        if engine.feed(raw[k:k + 25]):
            engine.step()

    assert not engine.calibrating and abs(engine.bias).max() < 0.5  # bias from the still start
    live = engine.curves()[0]
    offline = predict_session(model, stats, SessionData("X", {}, raw[:, 0], acc, gyr - engine.bias), hop=50)
    covered = engine.weight > 0.05
    # the windows sit in different places live vs offline, so only edge effects may differ
    assert np.abs(live[covered] - offline[covered]).mean() < 0.01

    count = engine.count()
    assert {"n_hat", "low", "high", "sigma", "level"} <= set(count) and count["low"] <= count["n_hat"] <= count["high"]
    layers = engine.latest_activations()
    assert len(layers) == 5 and len(layers[0]["values"]) == 16 and len(layers[0]["values"][0]) == 64
    assert engine.final_len() == len(raw) - 128
