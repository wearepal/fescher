"""Load and preprocess Kaggle credit dataset."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn import preprocessing

from src.dynamics.state import State
from src.types import FloatArray, IntArray

__all__ = ["CreditData"]


class CreditData:
    """Class to lazily load the credit dataset."""

    def __init__(self, seed: int | None = None) -> None:
        self.filepath = (Path(__file__).parent / "credit_data").with_suffix(".zip")
        self._features = None
        self._labels = None
        self.seed = seed

    @classmethod
    def as_state(cls, seed: int | None = None) -> State:
        data = cls(seed=seed)
        return State(features=data.features, labels=data.labels)

    @property
    def features(self) -> FloatArray:
        """Return the dataset features."""
        if self._features is None:
            self._features, self._labels = self.load()
        return np.copy(self._features)

    @property
    def labels(self) -> IntArray:
        """Return the dataset labels."""
        if self._labels is None:
            self._features, self._labels = self.load()
        return np.copy(self._labels)

    @property
    def num_agents(self) -> int:
        """Compute number of agents in the dataset."""
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        """Compute number of features for each agent."""
        return self.features.shape[1]

    def load(self) -> tuple[FloatArray, FloatArray]:
        """Load, preprocess and class-balance the credit data."""
        from zipfile import ZipFile

        rng = np.random.default_rng(seed=self.seed)
        data = pl.read_csv(
            ZipFile(self.filepath).open("credit_data.csv", mode="r").read(),
        )
        data = data.drop("")
        # Replace "NA" with 'null'
        data = data.with_columns(
            pl.when(pl.col(pl.Utf8) != "NA").then(pl.col(pl.Utf8)).name.keep()
        )
        # Drop null-containing rows
        data = data.drop_nulls()
        data = data.cast(
            {f"{col}": pl.Int64 for col in data.columns if data[col].dtype == pl.Utf8}
        )
        outcomes = data.drop_in_place("SeriousDlqin2yrs")
        # zero mean, unit variance
        features = preprocessing.scale(data.to_numpy())
        # add bias term
        features = np.append(features, np.ones((features.shape[0], 1)), axis=1)  # type: ignore
        outcomes = outcomes.to_numpy()

        # balance classes
        default_indices = np.nonzero(outcomes == 1)[0]
        other_indices = rng.permutation(np.nonzero(outcomes == 0)[0])[:10000]
        indices = np.concatenate((default_indices, other_indices))

        features_balanced = features[indices]
        outcomes_balanced = outcomes[indices]

        # shuffle arrays
        shuffled = rng.permutation(len(indices))
        return features_balanced[shuffled], outcomes_balanced[shuffled]
