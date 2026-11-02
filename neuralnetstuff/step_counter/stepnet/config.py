"""Config-driven experiments (PLAN.md D18): YAML files checked against dataclasses.

  cfg = load_config("configs/cnn.yaml", ["model.channels=48", "train.epochs=40"])

A typo in a key fails immediately instead of being silently ignored.
"""
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    window: int = 256  # samples (2.56 s), PLAN.md D10
    target_sigma: float = 5.0  # Gaussian bump around each stride, in samples (50 ms)
    windows_per_epoch: int = 3000
    rotate_deg: float = 10.0  # random rotation, mimics re-strapping (D13)
    scale: float = 0.10  # random amplitude scaling +/-10%


@dataclass
class ModelConfig:
    channels: list = field(default_factory=lambda: [16, 32, 32, 32, 32])
    dilations: list = field(default_factory=lambda: [1, 2, 4, 8, 16])
    kernel: int = 7
    dropout: float = 0.1  # also what MC Dropout samples from in M6


@dataclass
class TrainConfig:
    epochs: int = 30
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float = 20.0  # BCE weight is 1 + pos_weight * target: strides are rare
    seed: int = 0


@dataclass
class DetectConfig:
    threshold: float = 0.5  # minimum peak probability for a stride
    min_gap_s: float = 0.3
    hop: int = 64  # sliding-window hop for whole-session inference


@dataclass
class Config:
    name: str = "cnn"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)


def _build(cls, values: dict, where: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"unknown config key(s) in {where or 'config'}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        kwargs[name] = _build(type(default), value, f"{where}.{name}".lstrip(".")) if is_dataclass(default) else value
    return cls(**kwargs)


def load_config(path=None, overrides=()) -> Config:
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    values = values or {}
    for item in overrides:
        key, _, raw = item.partition("=")
        node = values
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return _build(Config, values, "")


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)
