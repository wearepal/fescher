"""Utility functions for performative prediction demo."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing_extensions import Self

from loguru import logger
import numpy as np
from ranzen import some
from scipy.special import expit  # type: ignore

from src.types import FloatArray, IntArray

__all__ = ["logistic_loss", "Lr"]


@dataclass(kw_only=True)
class Lr:
    l2_penalty: float = 0.0
    """Regularization coefficient. Use l2_penalty=0 for no regularization."""
    _weights: FloatArray | None = field(init=False, default=None)
    """A [num_features] vector of classifier parameters to use a initialization."""

    @property
    def weights(self) -> FloatArray:
        if self._weights is None:
            raise AttributeError("LR model must be fit before its weights can be retrieved.")
        return self._weights

    @weights.setter
    def weights(self, value: FloatArray) -> None:
        self._weights = value

    def logits(self, features: FloatArray) -> FloatArray:
        return features @ self.weights

    def preds(self, features: FloatArray) -> IntArray:
        return (self.logits(features) > 0).astype(np.uint8)

    def acc(self, *, features: FloatArray, labels: IntArray) -> float:
        return (self.preds(features) == labels).mean()

    def probs(self, features: FloatArray) -> FloatArray:
        return expit(self.logits(features))

    def fit(self, *, x: FloatArray, y: IntArray, tol: float = 1e-7) -> Self:
        """Fit a logistic regression model via gradient descent.

        :param x: A [num_samples, num_features] matrix of features.
            The last feature dimension is assumed to be the bias term.
        :param y: A [num_samples] vector of binary labels.
        :param tol:Stopping criteria for gradient descent

        :returns: The optimal [num_features] vector of classifier parameters.
        """
        x = np.copy(x)
        y = np.copy(y)
        n, d = x.shape

        # Smoothness of the logistic loss
        smoothness = np.sum(x**2) / (4.0 * n)

        # Optimal initial learning rate
        eta_init = 1.0 / (smoothness + self.l2_penalty)

        theta = np.copy(self._weights) if some(self._weights) else np.zeros(d)

        # Evaluate loss at initialization
        prev_loss = logistic_loss(
            x=x,
            y=y,
            weights=theta,
            l2_penalty=self.l2_penalty,
        )

        loss_list = [prev_loss]
        i = 0
        gap = 1e30

        eta = eta_init
        while gap > tol and i < 10:  # 5_000:
            # take gradients
            exp_tx = np.exp(x @ theta)
            c = exp_tx / (1 + exp_tx) - y
            gradient = 1.0 / n * np.sum(x * c[:, np.newaxis], axis=0) + self.l2_penalty * np.append(
                theta[:-1], 0
            )

            new_theta = theta - eta * gradient

            # compute new loss
            loss = logistic_loss(
                x=x,
                y=y,
                weights=new_theta,
                l2_penalty=self.l2_penalty,
            )

            # do backtracking line search
            if loss > prev_loss:
                eta *= 0.1
                gap = 1.0e30
                continue

            eta = eta_init
            theta = np.copy(new_theta)

            loss_list.append(loss)
            gap = prev_loss - loss
            prev_loss = loss

            i += 1
        logger.info(f"Trained LR in {i} iterations")
        self._weights = theta
        return self

    def loss(
        self,
        *,
        x: FloatArray,
        y: IntArray,
    ) -> float:
        return logistic_loss(
            x=x,
            y=y,
            weights=self.weights,
            l2_penalty=self.l2_penalty,
        )


def logistic_loss(
    *,
    x: FloatArray,
    y: IntArray,
    weights: FloatArray,
    l2_penalty: float = 0.0,
) -> float:
    """Compute the l2-penalized logistic loss function

    :param X: A [num_samples, num_features] matrix of features. The last
        feature dimension is assumed to be the bias term.
    :param y: A [num_samples] vector of binary labels.
    :param theta: A [num_features] vector of classifier parameters
    :param l2_penalty: egularization coefficient. Set to 0 to disable regularization.

    :returns: logistic loss.
    """
    n = x.shape[0]

    logits = x @ weights
    log_likelihood = 1.0 / n * np.sum(-1.0 * (y * logits) + np.log(1 + np.exp(logits)))

    regularization = (l2_penalty / 2.0) * np.linalg.norm(weights[:-1]) ** 2

    return log_likelihood + regularization  # type: ignore
