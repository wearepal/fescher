from dataclasses import dataclass, fields
from typing_extensions import TypeAlias

from gymnasium import spaces
import numpy as np
import numpy.typing as npt

from src.types import FloatArray

Features: TypeAlias = FloatArray
Labels: TypeAlias = FloatArray

__all__ = ["State"]


@dataclass(kw_only=True)
class State:
    """Simulative state."""

    features: Features
    labels: Labels

    def __len__(self) -> int:
        return len(fields(self))

    @property
    def num_features(self) -> int:
        return self.features.shape[-1]

    def asdict(self) -> dict[str, npt.NDArray[np.floating]]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def action_space(self) -> spaces.Box:
        num_features = self.features.shape[-1]
        return spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float64)

    @property
    def observation_space(self) -> spaces.Dict:
        """Return observation space for credit simulator.

        The observation space is the vector of possible datasets, which
        must have the same dimensions as the initial state.
        """
        return spaces.Dict(
            {
                "features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.features.shape,
                    dtype=np.float64,
                ),
                "labels": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.labels.shape,
                    dtype=np.float64,
                ),
            }
        )
