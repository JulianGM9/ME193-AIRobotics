from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
