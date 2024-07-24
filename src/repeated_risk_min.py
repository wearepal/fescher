from dataclasses import dataclass

import numpy as np

from src.dynamics.env import DynamicEnv
from src.models.lr import Lr, logistic_loss


@dataclass(unsafe_hash=True, kw_only=True)
class EpisodeRecord:
    loss_start: list[float]
    loss_end: list[float]
    acc_start: list[float]
    acc_end: list[float]
    theta_gap: list[float]
    theta: list[np.ndarray]


def repeated_risk_minimization(
    *, env: DynamicEnv, num_steps: int, lr: Lr, l2_penalty: float
) -> EpisodeRecord:
    """Run repeated risk minimization for num_iters steps"""
    # Track loss and accuracy before/after updating model on new distribution
    loss_start = []
    loss_end = []
    acc_start = []
    acc_end = []
    theta_gap = []
    weight = []
    # Warm-start with baseline classifier
    theta = np.copy(lr.weights)
    for _ in range(num_steps):
        # Deploy classifier and observe strategic response
        observation = env.step(theta)[0]
        features_strat, labels = observation["features"], observation["labels"]

        # Evaluate loss and accuracy on the new distribution
        loss_start.append(
            logistic_loss(
                x=features_strat,
                y=labels,
                weights=theta,
                l2_penalty=l2_penalty,
            )
        )
        acc_start.append(lr.acc(features=features_strat, labels=labels))
        # Learn a new model on the induced distribution bootstrapped from the previous model.
        lr = lr.fit(x=features_strat, y=labels)
        # Evaluate loss and accuracy on the strategic distribution after training
        loss_end.append(
            lr.loss(
                x=features_strat,
                y=labels,
            )
        )
        acc_end.append(lr.acc(features=features_strat, labels=labels))
        # Track distance (in terms of Euclidean norm) between iterates
        theta_gap.append(np.linalg.norm(lr.weights - theta))
        weight.append(lr.weights)
        theta = np.copy(lr.weights)
    return EpisodeRecord(
        loss_start=loss_start,
        loss_end=loss_end,
        acc_start=acc_start,
        acc_end=acc_end,
        theta_gap=theta_gap,
        theta=weight,
    )
