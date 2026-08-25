from __future__ import annotations

import math
from typing import cast


def positive_finite_seconds(value: object, field_name: str) -> float:
    """Validate a public provider timeout without accepting booleans or infinities."""
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    try:
        normalized = float(cast("int | float", value))
    except OverflowError:
        raise ValueError(f"{field_name} must be finite and greater than zero.") from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")
    return normalized
