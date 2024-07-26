from abc import ABC, abstractmethod
from dataclasses import dataclass

from beartype import beartype
import numpy as np
from ranzen import unwrap_or

from src.types import FloatArray, IntArray

__all__ = ["Response", "LinearResponse"]


@beartype
class Response(ABC):
    @abstractmethod
    def respond(self, *, features: FloatArray, action: FloatArray) -> FloatArray: ...


@beartype
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


@beartype
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


@beartype
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
