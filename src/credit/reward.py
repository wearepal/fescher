from typing import Protocol

import numpy as np
import numpy.typing as npt
from credit.simulator import State
from typing_extensions import TypeAlias

Features: TypeAlias = npt.NDArray[np.floating]
Labels: TypeAlias = npt.NDArray[np.floating]

__all__ = [
    "State",
    "logistic_loss",
]

FloatArray: TypeAlias = npt.NDArray[np.floating]


class RewardFn(Protocol):
    def __call__(self, *, state: State, theta: FloatArray) -> float:
        ...


def logistic_loss(
    *,
    state: State,
    theta: FloatArray,
    weight_decay: float = 0.0,
) -> float:
    """Evaluate the performative loss for logistic regression classifier."""
    # compute log likelihood
    num_samples = state.features.shape[0]
    logits = state.features @ theta
    log_likelihood = (1.0 / num_samples) * np.sum(
        -1.0 * np.multiply(state.labels, logits) + np.log(1 + np.exp(logits))
    )

    # Add regularization to thetas, excluding the last dimension correspondin
    # to the bias term
    regularization = (weight_decay / 2.0) * np.linalg.norm(theta[:-1]) ** 2

    return log_likelihood + regularization
