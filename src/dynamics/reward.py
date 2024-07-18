from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias
from typing_extensions import override

from src.dynamics.state import State
from src.models.lr import logistic_loss
from src.types import FloatArray

Features: TypeAlias = FloatArray
Labels: TypeAlias = FloatArray

__all__ = [
    "Reward",
    "LogisticReward",
]


class Reward(ABC):
    @abstractmethod
    def calculate(self, *, state: State, action: FloatArray) -> float: ...


@dataclass(kw_only=True)
class LogisticReward(Reward):
    l2_penalty: float = 0.0

    @override
    def calculate(
        self,
        *,
        state: State,
        action: FloatArray,
    ) -> float:
        return logistic_loss(
            x=state.features,
            y=state.labels,
            weights=action,
            l2_penalty=self.l2_penalty,
        )
