"""Loading the trained ensemble and its calibration (models/cnn_ensemble_v1, made by scripts/uncertainty.py)."""
import json

import torch

from stepnet.models.cnn import StepCNN
from stepnet.tracking import ROOT

MODEL_DIR = ROOT / "models" / "cnn_ensemble_v1"


def load_ensemble(model_dir=MODEL_DIR):
    """([(model, (mean, std)), ...], calibration dict)."""
    cal = json.loads((model_dir / "calibration.json").read_text())
    m = cal["config"]["model"]
    members = []
    for k in range(cal["members"]):
        ck = torch.load(model_dir / f"member{k}.pt", weights_only=False)
        model = StepCNN(m["channels"], m["dilations"], m["kernel"], m["dropout"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        members.append((model, (ck["mean"], ck["std"])))
    return members, cal
