from __future__ import annotations
import sys
from typing import Final

__all__ = ["TESTING"]

TESTING: Final[bool] = "pytest" in sys.modules
