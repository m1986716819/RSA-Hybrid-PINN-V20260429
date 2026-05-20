from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..datasets.scene import PlanningResult, Scene


class BasePlanner(ABC):
    """Abstract interface for all benchmark planners.

    Subclasses must implement:
        plan(scene) -> PlanningResult
        name -> str
        reset()
    """

    @abstractmethod
    def plan(self, scene: Scene) -> PlanningResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def reset(self) -> None:
        """Reset planner state (called between scenes)."""
        pass

    def can_handle(self, scene: Scene) -> bool:
        """Check if this planner can handle the given scene size/resolution."""
        return True

    def config_snapshot(self) -> Dict[str, Any]:
        return {"planner": self.name}

    def requires_training(self) -> bool:
        """Whether this planner needs a training step before planning."""
        return False

    def train(self, scenes: Optional[np.ndarray] = None) -> None:
        """Optional training step (for learned planners)."""
        pass
