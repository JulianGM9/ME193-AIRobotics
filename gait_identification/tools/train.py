"""Train the two deployable models on every recorded (non-test) session.

    python3 tools/train.py

Writes python/models/model1.npz (peak shin acceleration, D11) and
python/models/model2.npz (18 motion features, D10). They live inside python/
so they ship with the board app. k is chosen by leave-one-session-out (D12).
These use ALL sessions, so they're for the live app; honest accuracy numbers
come from tools/evaluate.py, which holds the last session out.
"""
import common as c
from common import g


def main():
    sessions = c.find_sessions()
    if not sessions:
        raise SystemExit(f"No non-test sessions in {c.DATA_DIR} - record some first.")
    win = c.extract(sessions)
    train = win.subset(c.np.isin(win.label, g.CLASSES))
    print(f"{len(sessions)} sessions, {len(train.label)} training windows: "
          + ", ".join(f"{cl} {int((train.label == cl).sum())}" for cl in g.CLASSES))

    c.MODELS_DIR.mkdir(exist_ok=True)
    for model, name in ((1, "model1"), (2, "model2")):
        k, scores = c.choose_k(train, model)
        knn = g.KNN(k).fit(c.features_for(train, model), train.label)
        path = c.MODELS_DIR / f"{name}.npz"
        knn.save(path, model=model, axis="S", stat="max", k=k,
                 sessions=sorted(set(train.session.tolist())),
                 features=["peak_acc_S"] if model == 1 else g.MODEL2_FEATURE_NAMES)
        cv = ", ".join(f"k={kk}: {s:.1%}" for kk, s in scores.items()) or "only one session - k=5"
        print(f"Model {model}: k={k} -> {path}\n   leave-one-session-out accuracy: {cv}")


if __name__ == "__main__":
    main()
