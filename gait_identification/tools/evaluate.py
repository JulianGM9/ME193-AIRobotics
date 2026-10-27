"""Evaluate Model 1 vs Model 2 on the held-out last session (PLAN.md D14, D15).

    python3 tools/evaluate.py

Trains on every session except the last, picks k by leave-one-session-out on
those, tests on the last one, and writes reports/report.md plus confusion
matrix PNGs. Nothing from the held-out session is used to choose anything.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

import common as c  # noqa: E402
from common import g, np  # noqa: E402

LABEL_ORDER = g.CLASSES


def run(train, test, model, axis="S", stat="max", drop_group=None):
    """Fit on train, predict test. Returns (k, raw predictions, smoothed predictions)."""
    k, _ = c.choose_k(train, model, axis, stat, drop_group)
    knn = g.KNN(k).fit(c.features_for(train, model, axis, stat, drop_group), train.label)
    pred = knn.predict(c.features_for(test, model, axis, stat, drop_group))
    order = np.argsort(test.order)
    smoothed = np.empty_like(pred)
    smoothed[order] = g.smooth(pred[order])
    return k, pred, smoothed, knn


def accuracy(y, p):
    return float((y == p).mean()) if len(y) else float("nan")


def plot_confusion(y, p, title, path):
    cm = confusion_matrix(y, p, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), LABEL_ORDER, rotation=30)
    ax.set_yticks(range(4), LABEL_ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    sessions = c.find_sessions()
    if len(sessions) < 2:
        raise SystemExit("Need at least 2 non-test sessions (train on the rest, test on the last).")
    win = c.extract(sessions)
    holdout = int(win.session.max())
    is_class = np.isin(win.label, g.CLASSES)
    train = win.subset(is_class & (win.session != holdout))
    test = win.subset(is_class & (win.session == holdout))
    posture = win.subset(win.label == g.POSTURE_CHECK)
    c.REPORTS_DIR.mkdir(exist_ok=True)

    lines = ["# Gait classifier evaluation", "",
             f"Training sessions: {sorted(set(train.session.tolist()))} ({len(train.label)} windows). "
             f"Held-out test session: {holdout} ({len(test.label)} windows).", ""]

    # ---- main comparison
    main_results = {}
    lines += ["## Model 1 vs Model 2 (held-out session)", "",
              "| Model | k | Raw accuracy | Smoothed accuracy |", "|---|---|---|---|"]
    for model, name in ((1, "Model 1 - peak shin accel (1 feature)"), (2, "Model 2 - full IMU (18 features)")):
        k, pred, sm, knn = run(train, test, model)
        main_results[model] = (pred, knn)
        lines.append(f"| {name} | {k} | {accuracy(test.label, pred):.1%} | {accuracy(test.label, sm):.1%} |")
        plot_confusion(test.label, pred, f"Model {model} (raw, session {holdout})",
                       c.REPORTS_DIR / f"confusion_model{model}.png")
    lines.append("")

    # ---- per-class precision / recall
    lines += ["## Per-class precision / recall (raw)", "",
              "| Class | M1 precision | M1 recall | M2 precision | M2 recall | test windows |",
              "|---|---|---|---|---|---|"]
    pr = {m: precision_recall_fscore_support(test.label, main_results[m][0], labels=LABEL_ORDER,
                                             zero_division=0) for m in (1, 2)}
    for i, cl in enumerate(LABEL_ORDER):
        lines.append(f"| {cl} | {pr[1][0][i]:.0%} | {pr[1][1][i]:.0%} | {pr[2][0][i]:.0%} | {pr[2][1][i]:.0%} "
                     f"| {int(pr[1][3][i])} |")
    lines += ["", "Confusion matrices: `confusion_model1.png`, `confusion_model2.png`.", ""]

    # ---- Model 1 axis / statistic comparison
    lines += ["## Model 1: which axis? (D6, D11)", "",
              "| Axis | Peak (max) accuracy | 95th-percentile accuracy |", "|---|---|---|"]
    for axis in g.AXES:
        accs = [accuracy(test.label, run(train, test, 1, axis, stat)[1]) for stat in ("max", "p95")]
        lines.append(f"| {axis} | {accs[0]:.1%} | {accs[1]:.1%} |")
    lines.append("")

    # ---- Model 2 feature groups
    lines += ["## Model 2: drop one feature group at a time (D10)", "",
              "| Feature group removed | Accuracy | Change |", "|---|---|---|"]
    base = accuracy(test.label, main_results[2][0])
    for group in g.MODEL2_GROUPS:
        a = accuracy(test.label, run(train, test, 2, drop_group=group)[1])
        lines.append(f"| {group} | {a:.1%} | {a - base:+.1%} |")
    lines.append("")

    # ---- posture check
    lines += ["## Posture check: lying still on your back (D10)", "",
              "Never used for training. Should come out as **standing** (no motion), not biking.", ""]
    if len(posture.label):
        lines += ["| Model | " + " | ".join(LABEL_ORDER) + " |", "|---|" + "---|" * 4]
        for m in (1, 2):
            p = main_results[m][1].predict(c.features_for(posture, m))
            lines.append(f"| Model {m} | " + " | ".join(f"{(p == cl).mean():.0%}" for cl in LABEL_ORDER) + " |")
        lines.append(f"\n({len(posture.label)} lying-still windows from all sessions.)")
    else:
        lines.append("No lying-still windows recorded.")
    lines.append("")

    # ---- every-session view
    if len(sessions) >= 3:
        lines += ["## Leave-one-session-out across all sessions (k fixed at 5)", "",
                  "| Test session | Model 1 | Model 2 |", "|---|---|---|"]
        allw = win.subset(is_class)
        for s in sorted(set(allw.session.tolist())):
            tr, te = allw.subset(allw.session != s), allw.subset(allw.session == s)
            row = [accuracy(te.label, g.KNN(5).fit(c.features_for(tr, m), tr.label).predict(c.features_for(te, m)))
                   for m in (1, 2)]
            lines.append(f"| {s} | {row[0]:.1%} | {row[1]:.1%} |")
        lines.append("")

    out = c.REPORTS_DIR / "report.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
