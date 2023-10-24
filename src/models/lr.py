"""Utility functions for performative prediction demo."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypeAlias
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from ranzen import some
from scipy.special import expit  # type: ignore

FloatArray: TypeAlias = npt.NDArray[np.floating]

__all__ = [
    "logistic_loss",
    "fit_lr_with_gd",
    "Lr",
]


@dataclass(kw_only=True)
class Lr:
    #: L2 penalty on the logistic regression loss
    l2_penalty: float = 0.0
    #: Parameters for logistic-regression classifier used by the institution
    _weights: FloatArray | None = field(init=False, default=None)

    @property
    def weights(self) -> FloatArray:
        if self._weights is None:
            raise AttributeError("LR model must be fit before weights can be retrieved.")
        return self.weights

    @weights.setter
    def weights(self, value: FloatArray) -> None:
        self._weights = value

    def logits(self, features: FloatArray) -> FloatArray:
        return features @ self.weights

    def probs(self, features: FloatArray) -> FloatArray:
        return expit(features @ self.weights)

    def fit(
        self,
        *,
        x: FloatArray,
        y: FloatArray,
        tol: float = 1e-7,
    ) -> Self:
        self._weights = fit_lr_with_gd(
            x=x,
            y=y,
            tol=tol,
            l2_penalty=self.l2_penalty,
            theta_init=self._weights,
        )
        return self

    def loss(
        self,
        *,
        x: FloatArray,
        y: FloatArray,
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
    y: FloatArray,
    weights: FloatArray,
    l2_penalty: float = 0.0,
) -> float:
    """Compute the l2-penalized logistic loss function

    Parameters
    ----------
        X: np.ndarray
            A [num_samples, num_features] matrix of features. The last
            feature dimension is assumed to be the bias term.
        Y: np.ndarray
            A [num_samples] vector of binary labels.
        theta: np.ndarray
            A [num_features] vector of classifier parameters
        l2_penalty: float
            Regularization coefficient. Use l2_penalty=0 for no regularization.

    Returns
    -------
        loss: float

    """
    n = x.shape[0]

    logits = x @ weights
    log_likelihood = 1.0 / n * np.sum(-1.0 * (y * logits) + np.log(1 + np.exp(logits)))

    regularization = (l2_penalty / 2.0) * np.linalg.norm(weights[:-1]) ** 2

    return log_likelihood + regularization


def fit_lr_with_gd(
    *,
    x: FloatArray,
    y: FloatArray,
    l2_penalty: float,
    tol: float = 1e-7,
    theta_init: FloatArray | None = None,
) -> FloatArray:
    """Fit a logistic regression model via gradient descent.

    Parameters
    ----------
        X: np.ndarray
            A [num_samples, num_features] matrix of features.
            The last feature dimension is assumed to be the bias term.
        Y: np.ndarray
            A [num_samples] vector of binary labels.
        l2_penalty: float
            Regularization coefficient. Use l2_penalty=0 for no regularization.
        tol: float
            Stopping criteria for gradient descent
        theta_init: np.ndarray
            A [num_features] vector of classifier parameters to use a
            initialization

    Returns
    -------
        theta: np.ndarray
            The optimal [num_features] vector of classifier parameters.

    """
    x = np.copy(x)
    y = np.copy(y)
    n, d = x.shape

    # Smoothness of the logistic loss
    smoothness = np.sum(x**2) / (4.0 * n)

    # Optimal initial learning rate
    eta_init = 1.0 / (smoothness + l2_penalty)

    theta = np.copy(theta_init) if some(theta_init) else np.zeros(d)

    # Evaluate loss at initialization
    prev_loss = logistic_loss(
        x=x,
        y=y,
        weights=theta,
        l2_penalty=l2_penalty,
    )

    loss_list = [prev_loss]
    i = 0
    gap = 1e30

    eta = eta_init
    while gap > tol:
        # take gradients
        exp_tx = np.exp(x @ theta)
        c = exp_tx / (1 + exp_tx) - y
        gradient = 1.0 / n * np.sum(x * c[:, np.newaxis], axis=0) + l2_penalty * np.append(
            theta[:-1], 0
        )

        new_theta = theta - eta * gradient

        # compute new loss
        loss = logistic_loss(
            x=x,
            y=y,
            weights=new_theta,
            l2_penalty=l2_penalty,
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

    return theta
