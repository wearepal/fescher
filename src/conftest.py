from __future__ import annotations
import sys
from typing import Final
from typing_extensions import Self

import numpy as np
import pytest

from src.dynamics.registration import make_env
from src.dynamics.state import State
from src.models.lr import Model
from src.types import FloatArray, IntArray


@pytest.fixture
def mock_state():
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    labels = np.array([1, 0], dtype=np.int64)
    return State(features=features, labels=labels)


@pytest.fixture
def mock_env(mock_state: State):
    return make_env(initial_state=mock_state, epsilon=0.1, memory=False, changeable_features=[0])


@pytest.fixture
def mock_model():
    class MockModel(Model):
        def __init__(self):
            self._weights = np.array([0.1, 0.2])

        @property
        def weights(self) -> FloatArray:
            return self._weights

        @weights.setter
        def weights(self, value: FloatArray) -> None:
            self._weights = value

        def fit(
            self,
            *,
            x: FloatArray,
            y: IntArray,
            tol: float = 1e-7,
        ) -> Self:
            self.weights = np.array([0.3, 0.4])
            return self

        def acc(self, *, features: FloatArray, labels: IntArray) -> float:
            return 0.5

        def loss(
            self,
            *,
            x: FloatArray,
            y: IntArray,
        ) -> float:
            return 0.6

        @property
        def l2_penalty(self) -> float:
            raise NotImplementedError

        def logits(self, features: FloatArray) -> FloatArray:
            raise NotImplementedError

        def preds(self, features: FloatArray) -> IntArray:
            raise NotImplementedError

        def probs(self, features: FloatArray) -> FloatArray:
            raise NotImplementedError

    return MockModel()
