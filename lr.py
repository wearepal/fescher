"""Utility functions for performative prediction demo."""
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.floating]

__all__ = [
    "evaluate_logistic_loss",
    "fit_lr_with_gd",
]


def evaluate_logistic_loss(
    *,
    x: FloatArray,
    y: FloatArray,
    theta: FloatArray,
    l2_penalty: float,
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

    logits = x @ theta
    log_likelihood = 1.0 / n * np.sum(-1.0 * (y * logits) + np.log(1 + np.exp(logits)))

    regularization = (l2_penalty / 2.0) * np.linalg.norm(theta[:-1]) ** 2

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

    if theta_init is not None:
        theta = np.copy(theta_init)
    else:
        theta = np.zeros(d)

    # Evaluate loss at initialization
    prev_loss = evaluate_logistic_loss(
        x=x,
        y=y,
        theta=theta,
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
        gradient = 1.0 / n * np.sum(
            x * c[:, np.newaxis], axis=0
        ) + l2_penalty * np.append(theta[:-1], 0)

        new_theta = theta - eta * gradient

        # compute new loss
        loss = evaluate_logistic_loss(
            x=x,
            y=y,
            theta=new_theta,
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
