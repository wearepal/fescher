from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "Action",
    "FloatArray",
    "IntArray",
]

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.uint8]
Action: TypeAlias = npt.NDArray[np.float64]
