from dataclasses import dataclass
from functools import partial
from typing import Callable, TypeVar

from src.types import FloatArray

__all__ = ["Intervention", "NoIntervention"]


@dataclass(kw_only=True)
class Intervention:
    time: int | None
    fn: Callable[[FloatArray], FloatArray]

    def __call__(self, *, time: int, features: FloatArray) -> FloatArray:
        if (self.time is None) or (time >= self.time):
            features = self.fn(features)
        return features


T = TypeVar("T")


def identity(x: T) -> T:
    return x


NoIntervention = partial(Intervention, time=None, fn=identity)
