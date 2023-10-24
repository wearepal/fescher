from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "Action",
    "FloatArray",
    "IntArray",
]

FloatArray: TypeAlias = npt.NDArray[np.floating]
IntArray: TypeAlias = npt.NDArray[np.integer]
Action: TypeAlias = npt.NDArray[np.floating]
