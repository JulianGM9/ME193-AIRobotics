"""Training loop, whole-session inference and peak picking (PLAN.md D10).

Loss: binary cross-entropy between the logits and the Gaussian stride targets, weighted
1 + pos_weight x target (strides are ~1 sample in 100, so without the weight the network
could score well by predicting "no stride" everywhere), averaged over masked samples only.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

from stepnet.data.windows import WindowDataset, channel_stats, normalize
from stepnet.models.cnn import StepCNN


def masked_bce(logits, target, mask, pos_weight):
    weight = (1 + pos_weight * target) * mask
    loss = F.binary_cross_entropy_with_logits(logits, target, weight=weight, reduction="sum")
    return loss / mask.sum().clamp(min=1)


def pick_device(name="auto"):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def train_model(sessions, cfg, log=None):
    """Train a StepCNN on the given SessionData list. Returns (model on the CPU, (mean, std))."""
    device = pick_device(cfg.train.device)
    torch.manual_seed(cfg.train.seed)
    mean, std = channel_stats(sessions)
    data = WindowDataset(sessions, cfg.data, mean, std, augment=True, seed=cfg.train.seed)
    loader = torch.utils.data.DataLoader(data, batch_size=cfg.train.batch)
    model = StepCNN.from_config(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    for epoch in range(cfg.train.epochs):
        model.train()
        total, n = 0.0, 0
        for x, target, mask in loader:
            x, target, mask = x.to(device), target.to(device), mask.to(device)
            loss = masked_bce(model(x), target, mask, cfg.train.pos_weight)
            opt.zero_grad()
            loss.backward()  # autograd: d(loss)/d(every weight), through all 6 conv layers
            opt.step()  # Adam: move each weight against its gradient
            total, n = total + loss.item() * len(x), n + len(x)
        if log:
            log(epoch=epoch + 1, loss=total / n)
    return model.cpu(), (mean, std)


@torch.no_grad()
def predict_session(model, stats, session, window=256, hop=64, mc_dropout=False):
    """Stride probability for every sample of a session: sliding windows, overlaps averaged.

    mc_dropout=True keeps dropout on (for MC Dropout sampling in M6).
    """
    model.train(mc_dropout)
    mean, std = stats
    x = normalize(np.concatenate([session.acc, session.gyr], axis=1).T.astype(np.float32), mean, std)
    n = x.shape[1]
    starts = list(range(0, max(1, n - window + 1), hop))
    if starts[-1] != max(0, n - window):
        starts.append(max(0, n - window))
    prob, count = np.zeros(n), np.zeros(n)
    batch = torch.from_numpy(np.stack([x[:, s:s + window] for s in starts]))
    p = torch.sigmoid(model(batch)).numpy()
    for s, row in zip(starts, p):
        # the window's edges see less context, so they count less than its middle
        w = np.hanning(window + 2)[1:-1]
        prob[s:s + window] += row * w
        count[s:s + window] += w
    model.eval()
    return prob / np.maximum(count, 1e-9)


def detect_peaks(prob, seq, threshold=0.5, min_gap_s=0.3):
    """(stride seqs, their peak probabilities): local maxima of the probability curve."""
    idx, props = find_peaks(prob, height=threshold, distance=max(1, int(min_gap_s * 100)))
    return seq[idx], props["peak_heights"]
