# code/training/state.py
"""Lightweight runtime state used by the experiment loop."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

@dataclass
class EvalStats:
    """Collect and summarize scalar metrics across repeated evaluations."""
    values: List[float] = field(default_factory=list)
    def add(self, v: float) -> None:
        self.values.append(float(v))
    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")
    @property
    def best(self) -> float:
        return float(np.min(self.values)) if self.values else float("nan")
    @property
    def std(self) -> float:
        return float(np.std(self.values)) if self.values else float("nan")
    def as_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "best": self.best, "std": self.std, "n": len(self.values),}


@dataclass
class RunState:
    """Mutable state shared across repeated runs of one model configuration."""

    model_printed: bool = False
    table_created: bool = False
    table: Optional[object] = None
    loop_time_list: List[float] = field(default_factory=list)
