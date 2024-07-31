from abc import ABC, abstractmethod
from dataclasses import dataclass

from beartype import beartype
import numpy as np
from ranzen import unwrap_or

from src.types import FloatArray

__all__ = ["Response", "LinearResponse"]


@beartype
class Response(ABC):
    @abstractmethod
    def respond(self, *, features: FloatArray, action: FloatArray) -> FloatArray: ...


@beartype
@dataclass(kw_only=True)
class LinearResponse(Response):
    epsilon: dict[int, float] | None = None

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
        epsilon_map = unwrap_or(self.epsilon, default={a: 0 for a in range(features.shape[1])})
        new_features = np.copy(features)

        neg_item_mask = np.dot(action[None, :], features[...].T)[0] < 0
        for feature, weight in epsilon_map.items():
            # TODO: Make a decision on the normalisation.
            # In Perdomo et al this is not present, but some sort of normalisation (not neccessarily this) might make sense.
            new_features[np.nonzero(neg_item_mask), feature] += (
                weight / np.linalg.norm(action)
            ) * action[feature]
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

    epsilon: dict[int, float] | None = None

    def __call__(
        self,
        *,
        features: FloatArray,
        action: FloatArray,
    ) -> FloatArray:
        epsilon_map = unwrap_or(self.epsilon, default={a: 0.0 for a in range(features.shape[1])})
        return self.rir_response(
            features=features,
            action=action,
            epsilon=epsilon_map,
        )

    @beartype
    def rir_response(
        self,
        *,
        features: FloatArray,
        action: FloatArray,
        epsilon: dict[int, float],
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
        modified_action = action.copy()
        for i in range(len(action)):
            modified_action[i] += epsilon.get(i, 0)
        resample_inds = np.random.default_rng(0).binomial(n=1, p=modified_action.clip(max=1.0))
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

        new_strategic_features = features[transplant_inds, np.array(list(epsilon.keys()))[:, None]]
        new_features = np.copy(features)
        new_features[resample_inds, np.array(list(epsilon.keys()))[:, None]] = (
            new_strategic_features
        )

        return new_features
