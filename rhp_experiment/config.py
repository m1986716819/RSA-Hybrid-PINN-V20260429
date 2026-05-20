import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentConfig:
    def __init__(self, config_path: str, overrides: Optional[Dict] = None):
        self.config_path = Path(config_path)
        self.data: Dict[str, Any] = self._load(self.config_path)
        if overrides:
            self._apply_overrides(overrides)
        self._validate()

    def _load(self, path: Path) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _apply_overrides(self, overrides: Dict):
        for key, value in overrides.items():
            parts = key.split(".")
            target = self.data
            for p in parts[:-1]:
                if p not in target:
                    target[p] = {}
                target = target[p]
            target[parts[-1]] = value

    def _validate(self):
        env = self.data.get("env", {})
        if "name" not in env:
            raise ValueError("config must have env.name")
        if "start" not in env or "goal" not in env:
            raise ValueError("config must have env.start and env.goal")

    def save_snapshot(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False)

    def get_rhp_config_path(self) -> Path:
        return self.config_path

    @property
    def scenario(self) -> str:
        return self.data["env"]["name"]

    @property
    def num_seeds(self) -> int:
        cfg = self.data.get("experiment", {}) or {}
        return int(cfg.get("num_seeds", 5))

    @num_seeds.setter
    def num_seeds(self, val: int):
        if "experiment" not in self.data:
            self.data["experiment"] = {}
        self.data["experiment"]["num_seeds"] = val

    @property
    def device(self) -> str:
        return str(self.data.get("device", "cuda"))

    def to_dict(self) -> Dict:
        return dict(self.data)


def load_config(config_path: str, overrides: Optional[Dict] = None) -> ExperimentConfig:
    return ExperimentConfig(config_path, overrides)
