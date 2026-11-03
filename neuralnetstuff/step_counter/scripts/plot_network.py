"""Block diagram of StepCNN with its forward and backward equations -> deliverables/network_diagram.png

    ../../.venv/bin/python scripts/plot_network.py

Top row: the forward pass, left to right, with each layer's shape, dilation, receptive field and
parameter count (all as in stepnet/models/cnn.py and configs/cnn.yaml). Below it: the gradient
flowing back, right to left. The two panels underneath give the equations for each direction.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "deliverables" / "network_diagram.png"

FWD, BWD = "#2a78d6", "#eb6834"  # categorical slots 1 and 3 of the data-viz reference palette
SURFACE, TEXT, TEXT2, BOX = "#fcfcfb", "#0b0b0b", "#52514e", "#f1f0ec"

# (title, body lines, parameter count); receptive field 1 + 6 * sum(dilations so far)
BLOCKS = [
    ("Input $x$", ["6 x 256", "accel S, F, L", "gyro S, F, L", "normalized"], None),
    ("Conv 1", ["k = 7, d = 1", "6 -> 16", "ReLU, dropout", "RF 7"], 688),
    ("Conv 2", ["k = 7, d = 2", "16 -> 32", "ReLU, dropout", "RF 19"], 3616),
    ("Conv 3", ["k = 7, d = 4", "32 -> 32", "ReLU, dropout", "RF 43"], 7200),
    ("Conv 4", ["k = 7, d = 8", "32 -> 32", "ReLU, dropout", "RF 91"], 7200),
    ("Conv 5", ["k = 7, d = 16", "32 -> 32", "ReLU, dropout", "RF 187 (1.87 s)"], 7200),
    ("Head", ["1x1 conv", "32 -> 1", "logits $z_t$"], 33),
    ("Sigmoid", ["$p_t = \\sigma(z_t)$", "1 x 256", "stride prob."], None),
    ("Loss $L$", ["weighted,", "masked BCE", "vs target $y_t$"], None),
]
SHAPES = ["6x256", "16x256", "32x256", "32x256", "32x256", "32x256", "1x256", "1x256"]
GRADS = ["", "$\\delta_1$", "$\\delta_2$", "$\\delta_3$", "$\\delta_4$", "$\\delta_5$",
         "$\\partial L/\\partial z$", "$\\partial L/\\partial p$"]

FORWARD_EQS = [
    ("Normalize (training-set mean, std)", r"$\tilde{x}_c = (x_c - \mu_c)\,/\,s_c$"),
    ("Conv layer $l$ (dilation $d_l$ = 1, 2, 4, 8, 16; taps $k$ = 0 ... 6)",
     r"$a_l[c,t] = b_l[c] + \sum_{c'}\sum_{k} W_l[c,c',k]\; h_{l-1}[c',\,t + d_l(k-3)]$"),
    ("ReLU, then dropout 0.1 (training only)", r"$h_l = \mathrm{Dropout}(\max(0,\, a_l))$"),
    ("Head and sigmoid", r"$z_t = b + \sum_c w_c\, h_5[c,t], \qquad p_t = 1\,/\,(1 + e^{-z_t})$"),
    ("Target: Gaussian bump on each labeled stride $t_j$ ($\\sigma$ = 5 samples = 50 ms)",
     r"$y_t = \max_j\; \exp\!\left(-(t - t_j)^2 / 2\sigma^2\right)$"),
    ("Loss: mask $m_t$ = 1 inside counted blocks; positives weighted 21x",
     r"$L = \frac{1}{\sum_t m_t} \sum_t m_t (1 + 20 y_t)\,\left[-y_t \ln p_t - (1 - y_t)\ln(1 - p_t)\right]$"),
]
BACKWARD_EQS = [
    ("Output error (sigmoid + BCE simplify to prediction minus target)",
     r"$\frac{\partial L}{\partial z_t} = \frac{m_t (1 + 20 y_t)}{\sum_t m_t}\,(p_t - y_t)$"),
    (r"Weight gradient, with $\delta_l = \partial L / \partial a_l$",
     r"$\frac{\partial L}{\partial W_l[c,c',k]} = \sum_t \delta_l[c,t]\; h_{l-1}[c',\, t + d_l(k-3)]$"),
    ("Pass back to the previous layer (ReLU gate; dropped units also pass 0)",
     r"$\delta_{l-1}[c',s] = \mathbf{1}[a_{l-1}[c',s] > 0] \sum_c \sum_k W_l[c,c',k]\; \delta_l[c,\, s - d_l(k-3)]$"),
    ("Adam, per weight $w$ (weight decay $\\lambda$ = 1e-4)",
     r"$g = \frac{\partial L}{\partial w} + \lambda w, \quad m \leftarrow 0.9\,m + 0.1\,g, \quad v \leftarrow 0.999\,v + 0.001\,g^2$"),
    ("Step (lr $\\eta$ = 0.001; batch 64; 1,410 steps; 5 seeds for the ensemble)",
     r"$w \leftarrow w - \eta\; \hat{m}\,/\,(\sqrt{\hat{v}} + \epsilon)$"),
]


def arrow(ax, x0, x1, y, color):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=14, lw=2, color=color))


def main():
    fig = plt.figure(figsize=(16, 9.5), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")

    ax.text(0.02, 0.965, "StepCNN: forward pass and backpropagation", fontsize=18, weight="bold", color=TEXT, va="top")
    ax.text(0.02, 0.925, "Dilated 1-D CNN, 25,937 parameters. Each output sample sees 187 input samples (1.87 s).",
            fontsize=11.5, color=TEXT2, va="top")

    n, left, right = len(BLOCKS), 0.02, 0.98
    gap = 0.018
    w = (right - left - gap * (n - 1)) / n
    yb, h = 0.66, 0.19
    xs = [left + i * (w + gap) for i in range(n)]

    ax.text(left, 0.875, "Forward pass  ->", fontsize=12, weight="bold", color=FWD)
    for i, (title, lines, params) in enumerate(BLOCKS):
        edge = BWD if i == n - 1 else FWD
        ax.add_patch(FancyBboxPatch((xs[i], yb), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                    fc=BOX, ec=edge, lw=1.6))
        cx = xs[i] + w / 2
        ax.text(cx, yb + h - 0.022, title, ha="center", va="top", fontsize=11, weight="bold", color=TEXT)
        for j, line in enumerate(lines):
            ax.text(cx, yb + h - 0.06 - j * 0.028, line, ha="center", va="top", fontsize=9, color=TEXT2)
        if params:
            ax.text(cx, yb - 0.012, f"{params:,} params", ha="center", va="top", fontsize=8.5, color=TEXT2)

    # forward arrows (top of the gap) with activation shapes, backward arrows (bottom) with gradients
    for i in range(n - 1):
        x0, x1 = xs[i] + w, xs[i + 1]
        arrow(ax, x0 + 0.001, x1 - 0.001, yb + h * 0.72, FWD)
        arrow(ax, x1 - 0.001, x0 + 0.001, yb + h * 0.28, BWD)
    for i, (shape, grad) in enumerate(zip(SHAPES, GRADS)):
        mid = xs[i] + w + gap / 2
        ax.text(mid, yb + h * 0.72 + 0.022, shape, ha="center", fontsize=7, color=FWD, rotation=90, va="bottom")
        if grad:
            ax.text(mid, yb + h * 0.28 - 0.022, grad, ha="center", va="top", fontsize=9, color=BWD, rotation=90)

    ax.text(left, 0.605, "<-  Backpropagation: the error at every sample flows back right to left; "
            "each layer turns it into $\\partial L/\\partial W_l$ for its own weights, then Adam updates every weight",
            fontsize=11, color=BWD, weight="bold", va="top")

    # equation panels
    panels = [(0.02, "Forward equations", FWD, FORWARD_EQS), (0.51, "Backward equations + Adam", BWD, BACKWARD_EQS)]
    for x0, heading, color, eqs in panels:
        ax.add_patch(FancyBboxPatch((x0, 0.015), 0.47, 0.535, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="white", ec=color, lw=1.4))
        ax.text(x0 + 0.015, 0.53, heading, fontsize=13, weight="bold", color=color, va="top")
        for j, (label, eq) in enumerate(eqs):
            y = 0.475 - j * 0.077
            ax.text(x0 + 0.015, y, label, fontsize=9.5, color=TEXT2, va="center")
            ax.text(x0 + 0.025, y - 0.033, eq, fontsize=12.5, color=TEXT, va="center")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
