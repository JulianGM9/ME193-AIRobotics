"""Training windows for the network (PLAN.md D5, D10, D13).

Input: 6 channels (accel S/F/L in g, gyro S/F/L in deg/s) x `window` samples, each channel
normalized with mean/std from the TRAINING sessions only (not per window: absolute magnitude
is what separates a fidget from a step).

Target: a Gaussian bump (sigma ~50 ms) at every labeled stride, in [0, 1].
Mask: 1 inside the counted range of usable blocks, 0 elsewhere. Everything else in a session
is unlabeled (it may contain real strides with no count) and must never be taught as
"no stride", but it can still be seen as context at a window's edge.
"""
import numpy as np
import torch

from stepnet.data.blocks import SessionData


def session_arrays(s: SessionData, sigma: float):
    """(x (6, n) raw units, target (n,), mask (n,)) for a whole session."""
    x = np.concatenate([s.acc, s.gyr], axis=1).T.astype(np.float32)
    n = len(s.seq)
    target = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    idx = np.arange(n)
    for b in s.blocks:
        mask[(s.seq >= b.first) & (s.seq < b.last)] = 1.0
        for q in b.strides:
            c = np.searchsorted(s.seq, q)
            lo, hi = max(0, c - int(4 * sigma)), min(n, c + int(4 * sigma) + 1)
            target[lo:hi] = np.maximum(target[lo:hi], np.exp(-0.5 * ((idx[lo:hi] - c) / sigma) ** 2))
    return x, target, mask


def channel_stats(sessions) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over all samples of the given (training) sessions."""
    x = np.concatenate([np.concatenate([s.acc, s.gyr], axis=1) for s in sessions])
    return x.mean(axis=0).astype(np.float32), x.std(axis=0).astype(np.float32) + 1e-6


def normalize(x, mean, std):
    """x (6, n) or (batch, 6, n) in raw units -> normalized."""
    return (x - mean[:, None]) / std[:, None]


def random_rotation(rng, max_deg) -> np.ndarray:
    """A random 3x3 rotation of at most max_deg about a random axis (Rodrigues' formula)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class WindowDataset(torch.utils.data.Dataset):
    """Random windows centred on labeled (masked) samples, freshly drawn every epoch.

    Augmentation (D13): the same small rotation applied to accel and gyro (a slightly
    different strap angle), and +/-scale amplitude scaling. Random crops come for free.
    """

    def __init__(self, sessions, cfg_data, mean, std, augment=True, seed=0):
        self.cfg, self.mean, self.std, self.augment = cfg_data, mean, std, augment
        self.rng = np.random.default_rng(seed)
        self.items = [session_arrays(s, cfg_data.target_sigma) for s in sessions]
        half = cfg_data.window // 2
        # every masked sample that can be a window centre, as (session index, sample index)
        self.centres = np.concatenate([
            np.stack([np.full(len(c), k), c], axis=1)
            for k, (x, _, m) in enumerate(self.items)
            for c in [np.nonzero(m[half:len(m) - half])[0] + half]
        ])

    def __len__(self):
        return self.cfg.windows_per_epoch

    def __getitem__(self, _):
        k, c = self.centres[self.rng.integers(len(self.centres))]
        x, target, mask = self.items[k]
        w = self.cfg.window
        lo = c - w // 2 + self.rng.integers(-w // 4, w // 4 + 1)  # the labeled sample needn't be centred
        lo = int(np.clip(lo, 0, x.shape[1] - w))
        xw = x[:, lo:lo + w].copy()
        if self.augment:
            r = random_rotation(self.rng, self.cfg.rotate_deg)
            xw[:3] = r @ xw[:3]
            xw[3:] = r @ xw[3:]
            xw *= self.rng.uniform(1 - self.cfg.scale, 1 + self.cfg.scale)
        return (torch.from_numpy(normalize(xw, self.mean, self.std)),
                torch.from_numpy(target[lo:lo + w]), torch.from_numpy(mask[lo:lo + w]))
