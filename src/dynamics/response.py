from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from ranzen import unwrap_or

from src.conftest import TESTING
from src.types import FloatArray, IntArray

__all__ = ["Response", "LinearResponse"]


class Response(ABC):
    @abstractmethod
    def respond(self, *, features: FloatArray, action: FloatArray) -> FloatArray: ...


@dataclass(kw_only=True)
class LinearResponse(Response):
    changeable_features: IntArray | slice | list[int] | None = None
    epsilon: float = 1.0

    def respond(
        self,
        *,
        features: FloatArray,
        action: FloatArray,
    ) -> FloatArray:
        action = action.flatten()
        if len(action) != features.shape[1]:
            raise ValueError(
                "'action' should correspond to the feature weights and thus must have entries "
                "numbering the number of columns in 'features'"
            )
        changeable_features = unwrap_or(self.changeable_features, default=[])
        new_features = np.copy(features)

        neg_item_mask = np.dot(action[None, :], features[...].T)[0] < 0
        assert not isinstance(changeable_features, slice)
        for changeable_feature in changeable_features:
            new_features[np.nonzero(neg_item_mask), changeable_feature] += (
                self.epsilon / np.linalg.norm(action)
            ) * action[changeable_feature]
        return new_features


def rir_response(
    *,
    features: FloatArray,
    action: FloatArray,
    epsilon: float = 1.0,
    changeable_features: IntArray | slice | list[int] | None = None,
) -> FloatArray:
    r"""
    Respond with the Resample-If-Rejected (RIR) procedure proposed in
    `Performative Prediction with Neural Networks` wherein the strategic
    features of a given simple are replaced with the strategic features
    of another sample with probability equal to the classifier's
    probabilities plus some performative hyperparameter, \epsilon.

    .. _Performative Prediction with Neural Networks:
        https://proceedings.mlr.press/v206/mofakhami23a.html

    .. note::
        The incoming predictions (actions) are presumed to lie in the
        interval [0, 1] (i.e. be valid probabilities).
    """
    n = features.shape[0]
    resample_inds = np.random.default_rng(0).binomial(n=1, p=(action + epsilon).clip(max=1.0))
    # sample new indices by shifting by x ~ unif{1, n-1} and projecting the image
    # onto the Z/nZ ring.
    shift = np.random.default_rng(0).integers(
        low=1,
        high=n,
        size=(
            len(
                resample_inds,
            ),
        ),
    )
    transplant_inds = (resample_inds + shift) % n

    changeable_features = unwrap_or(changeable_features, default=slice(None))
    if isinstance(changeable_features, list):
        changeable_features = np.array(changeable_features)
    # Add broadcasting dimensions in event that the strategic-feature indices
    # are encoded by a numpy array (or by a list-turned-array)
    if isinstance(changeable_features, np.ndarray):
        resample_inds = resample_inds[:, None]
        transplant_inds = transplant_inds[:, None]
        changeable_features = changeable_features[None]

    new_strategic_features = features[transplant_inds, changeable_features]
    new_features = np.copy(features)
    new_features[resample_inds, changeable_features] = new_strategic_features

    return new_features


@dataclass(kw_only=True)
class RIRResponse:
    r"""
    Respond with the Resample-If-Rejected (RIR) procedure proposed in
    `Performative Prediction with Neural Networks` wherein the strategic
    features of a given simple are replaced with the strategic features
    of another sample with probability equal to the classifier's
    probabilities plus some performative hyperparameter, \epsilon.

    .. _Performative Prediction with Neural Networks:
        https://proceedings.mlr.press/v206/mofakhami23a.html

    .. note::
        The incoming predictions (actions) are presumed to lie in the
        interval [0, 1] (i.e. be valid probabilities).
    """

    changeable_features: IntArray | slice | list[int] | None = None
    epsilon: float = 1.0

    def __call__(
        self,
        *,
        features: FloatArray,
        action: FloatArray,
    ) -> FloatArray:
        return rir_response(
            features=features,
            action=action,
            epsilon=self.epsilon,
            changeable_features=self.changeable_features,
        )


if TESTING:
    import pytest

    def test_linear_response():
        rng = np.random.default_rng(0)
        n = 19
        c = 5
        features = rng.normal(loc=0, scale=1, size=(n, c))
        action = rng.normal(loc=0, scale=0.5, size=(c,))

        response_fn = LinearResponse(epsilon=1.0, changeable_features=None)
        new_features = response_fn.respond(features=features, action=action)
        assert new_features.shape == features.shape

        changeable_features = np.array([0, 2])
        response_fn = LinearResponse(epsilon=1.0, changeable_features=changeable_features)
        new_features = response_fn.respond(features=features, action=action)
        assert new_features.shape == features.shape

        missized_action = rng.normal(loc=0, scale=0.5, size=(n,))
        with pytest.raises(ValueError):
            response_fn.respond(features=features, action=missized_action)

    def test_rir_response():
        rng = np.random.default_rng(0)
        n = 19
        c = 5
        features = rng.integers(low=0, high=4, size=(n, c)).astype(np.float64)
        action = rng.uniform(low=0, high=1, size=(n,))

        response_fn = RIRResponse(epsilon=1.0, changeable_features=None)
        new_features = response_fn(features=features, action=action)
        with pytest.raises(AssertionError):
            assert np.testing.assert_array_equal(features, new_features)

        changeable_features = np.array([0, 2])
        response_fn = RIRResponse(epsilon=1.0, changeable_features=changeable_features)
        new_features = response_fn(features=features, action=action)

        assert new_features.shape == features.shape
