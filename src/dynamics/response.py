from dataclasses import dataclass
from typing import Protocol
from typing_extensions import override

import numpy as np
from ranzen import unwrap_or

from src.types import FloatArray, IntArray

__all__ = [
    "ResponseFn",
    "linear_response",
    "LinearResponse",
]


class ResponseFn(Protocol):
    def __call__(
        self,
        *,
        features: FloatArray,
        action: FloatArray,
    ) -> FloatArray:
        ...


def linear_response(
    *,
    features: FloatArray,
    action: FloatArray,
    epsilon: float = 1.0,
    changeable_features: IntArray | slice | None = None,
) -> FloatArray:
    changeable_features = unwrap_or(changeable_features, default=slice(None))
    strategic_features = np.copy(features)
    theta_strat = action[changeable_features].flatten()
    strategic_features[changeable_features] -= epsilon * theta_strat
    return strategic_features


@dataclass(kw_only=True)
class LinearResponse(ResponseFn):
    changeable_features: IntArray | slice | None = None
    epsilon: float = 1.0

    @override
    def __call__(
        self,
        *,
        features: FloatArray,
        action: FloatArray,
    ) -> FloatArray:
        return linear_response(
            features=features,
            action=action,
            epsilon=self.epsilon,
            changeable_features=self.changeable_features,
        )
