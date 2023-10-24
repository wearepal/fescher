from dataclasses import dataclass
from typing import Protocol
from typing_extensions import TypeAlias, override

from src.dynamics.state import State
from src.models.lr import logistic_loss
from src.types import FloatArray

Features: TypeAlias = FloatArray
Labels: TypeAlias = FloatArray

__all__ = [
    "State",
    "RewardFn",
    "LogisticReward",
]


class RewardFn(Protocol):
    def __call__(self, *, state: State, action: FloatArray) -> float:
        ...


@dataclass(kw_only=True)
class LogisticReward(RewardFn):
    l2_penalty: float = 0.0

    @override
    def __call__(
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
