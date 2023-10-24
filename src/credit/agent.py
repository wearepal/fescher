import dataclasses
from functools import partial
from typing import Any, Callable, Generic, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.special import expit
from typing_extensions import override

FloatArray: TypeAlias = npt.NDArray[np.floating]

__all__ = ["Agent", "LrAgent", "Intervention"]


class Agent(Protocol):
    def update(self, features: FloatArray) -> FloatArray:
        ...

    def logits(self, features: FloatArray) -> FloatArray:
        ...

    def probs(self, x: FloatArray) -> FloatArray:
        ...


A = TypeVar("A", bound=Agent)


@dataclasses.dataclass(kw_only=True)
class Intervention(Generic[A]):
    time: int | None
    fn: Callable[[A], A]

    def __call__(self, *, time: int, agent: A) -> A:
        if (self.time is None) or (time >= self.time):
            agent = self.fn(agent)
        return agent


NoIntervention = partial(Intervention, time=None, fn=lambda x: x)


@dataclasses.dataclass
class LrAgent(Agent):
    changeable_features: npt.NDArray[np.integer] = dataclasses.field(
        default_factory=lambda: np.array([0, 5, 7])
    )

    #: Model how much the agent adapt her features in response to a classifier
    epsilon: float = 0.1

    #: Parameters for logistic regression classifier used by the institution
    theta: FloatArray = dataclasses.field(default_factory=lambda: np.ones((11, 1)))

    #: L2 penalty on the logistic regression loss
    l2_penalty: float = 0.0

    @override
    def update(
        self,
        features: FloatArray,
    ) -> Any:
        strategic_features = np.copy(features)
        theta_strat = self.theta[self.changeable_features].flatten()
        strategic_features[:, self.changeable_features] -= self.epsilon * theta_strat
        return strategic_features

    @override
    def logits(self, x: FloatArray) -> FloatArray:
        return x @ self.theta

    @override
    def probs(self, x: FloatArray) -> FloatArray:
        return expit(x @ self.theta)
