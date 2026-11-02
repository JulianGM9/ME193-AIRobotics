"""Experiment tracking and dataset versions (PLAN.md D18).

  runs/<time>_<name>/   config.yaml, run.json (git commit, dirty flag, dataset hash, seed),
                        metrics.jsonl (one line per logged step), summary.json, model files
  data/datasets/<hash>.json   which sessions + label versions a run used, named by content hash

Same data -> same hash, so any result can be traced back to exactly what it was trained on.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml

from stepnet import labels as lab
from stepnet.config import to_dict

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
DATASETS_DIR = ROOT / "data" / "datasets"


def dataset_manifest(train, held_out=(), labels_dir=lab.LABELS_DIR) -> tuple[str, dict]:
    """(hash, manifest) for a split; also writes data/datasets/<hash>.json."""
    def version(s):
        labels = lab.load_latest_labels(s, labels_dir)
        return labels["version"] if labels else None

    manifest = {"train": {s: version(s) for s in train}, "held_out": {s: version(s) for s in held_out},
                "event": lab.EVENT}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    (DATASETS_DIR / f"{digest}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return digest, manifest


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Run:
    def __init__(self, cfg, dataset_hash, tag=""):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = RUNS_DIR / f"{stamp}_{cfg.name}{'_' + tag if tag else ''}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        (self.dir / "run.json").write_text(json.dumps({
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", ".")),
            "dataset": dataset_hash,
            "seed": cfg.train.seed,
        }, indent=2) + "\n")
        self._metrics = open(self.dir / "metrics.jsonl", "a")

    def log(self, **values):
        self._metrics.write(json.dumps(values) + "\n")
        self._metrics.flush()

    def save_model(self, model, stats, name="model.pt", **extra):
        mean, std = stats
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, **extra}, self.dir / name)

    def finish(self, summary: dict):
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        self._metrics.close()
