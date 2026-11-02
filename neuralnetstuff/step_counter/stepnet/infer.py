"""Streaming inference for the live view (milestone M7, PLAN.md D10, D15, D17).

Samples arrive in small batches (live over MQTT, or replayed from a recorded session). Every
HOP new samples the ensemble looks at the latest `window` samples (2.56 s), and its outputs
are stitched into one probability curve with the same Hann-weighted overlap-add as offline
inference. A stride only becomes final once the window has moved past it (LATENCY samples):
until then the CNN hasn't seen what comes after it. That is the ~1.3 s delay accepted in D12.

Forward hooks on the first member's ReLU layers capture what every layer "sees" in the
latest window: the network visualization in the live view.
"""
import numpy as np
import torch

from stepnet.calibration import load_accel_calibration
from stepnet.data.windows import normalize
from stepnet.motion import is_still
from stepnet.preprocess import to_body, to_physical
from stepnet.uncertainty.counting import estimate, peaks_in

HOP = 50  # samples between inference steps: 2 per second
CALIBRATION_SAMPLES = 300  # live: the first 3 s must be standing still (gyro bias)


class LiveEngine:
    def __init__(self, members, cal, gyro_bias_dps=None, accel_cal="default", level=0.9):
        self.members, self.cal = members, cal
        self.window = cal["config"]["data"]["window"]
        self.latency = self.window // 2  # a stride is final once it is at the window's centre
        self.accel_cal = load_accel_calibration() if accel_cal == "default" else accel_cal
        self.bias = None if gyro_bias_dps is None else np.asarray(gyro_bias_dps, float)
        self.level = level
        self.raw = []  # rows of (seq, t_ms, ax..gz) received before the bias is known
        self.seq, self.x = [], []  # body-frame samples: seq and (6,) acc+gyr
        self.prob = np.zeros((len(members), 0))
        self.weight = np.zeros(0)
        self.since_step = 0
        self.activations = []
        self._hooks = [m for m in members[0][0].body if isinstance(m, torch.nn.ReLU)]
        for layer in self._hooks:
            layer.register_forward_hook(self._capture)
        self._capturing = False

    # ---- input ----

    @property
    def calibrating(self):
        return self.bias is None

    def feed(self, samples):
        """Append raw samples (n, 8). Returns True when it's time for an inference step."""
        samples = np.asarray(samples)
        if self.bias is None:
            self.raw.extend(samples.tolist())
            if len(self.raw) < CALIBRATION_SAMPLES:
                return False
            raw = np.array(self.raw)
            _, gyr = to_physical(raw[:CALIBRATION_SAMPLES], self.accel_cal)
            bias = gyr.mean(axis=0)
            if not is_still(raw[:CALIBRATION_SAMPLES], bias):
                self.raw = self.raw[HOP:]  # not still yet: keep waiting
                return False
            self.bias = bias
            samples, self.raw = raw, []
        acc, gyr = to_physical(samples, self.accel_cal)
        x = np.concatenate([to_body(acc), to_body(gyr - self.bias)], axis=1)
        self.seq.extend(samples[:, 0].tolist())
        self.x.extend(x.tolist())
        grow = len(samples)
        self.prob = np.concatenate([self.prob, np.zeros((len(self.members), grow))], axis=1)
        self.weight = np.concatenate([self.weight, np.zeros(grow)])
        self.since_step += grow
        return self.since_step >= HOP and len(self.seq) >= self.window

    def _capture(self, module, inputs, output):
        if self._capturing:
            self.activations.append(output[0].detach().numpy())

    # ---- inference ----

    @torch.no_grad()
    def step(self):
        """Run the ensemble on the latest window and update the stitched curves."""
        self.since_step = 0
        n = len(self.seq)
        lo = n - self.window
        xw = np.asarray(self.x[lo:], dtype=np.float32).T
        w = np.hanning(self.window + 2)[1:-1]
        self.activations = []
        for k, (model, (mean, std)) in enumerate(self.members):
            self._capturing = k == 0
            p = torch.sigmoid(model(torch.from_numpy(normalize(xw, mean, std))[None])).numpy()[0]
            self.prob[k, lo:] += p * w
        self._capturing = False
        self.weight[lo:] += w

    def curves(self):
        return self.prob / np.maximum(self.weight, 1e-9)

    def final_len(self):
        """Samples whose strides can no longer change (the window has moved past them)."""
        return max(0, len(self.seq) - self.latency)

    def strides(self):
        mean = self.curves().mean(axis=0)
        idx, heights = peaks_in(mean, 0, self.final_len(), self.cal["threshold"], max(1, int(self.cal["min_gap_s"] * 100)))
        return np.asarray(self.seq)[idx], heights

    def count(self):
        """Steps so far with the conformal interval at self.level."""
        est = estimate(self.curves(), 0, self.final_len(), self.cal["sigma_delta2"], self.cal["threshold"], self.cal["min_gap_s"])
        q = self.cal["q"][str(self.level)]
        half = q * est["sigma"]
        return {**est, "level": self.level, "low": max(0.0, est["n_hat"] - half), "high": est["n_hat"] + half}

    def latest_activations(self, time_points=64):
        """Each ReLU layer of member 1 on the latest window, (channels, time_points) scaled to 0-255."""
        out = []
        for a in self.activations:
            a = a.reshape(a.shape[0], time_points, -1).mean(axis=2)
            top = float(a.max()) or 1.0
            out.append({"channels": int(a.shape[0]), "scale": top,
                        "values": np.round(255 * a / top).astype(np.uint8).tolist()})
        return out
