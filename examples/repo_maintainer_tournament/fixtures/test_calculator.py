import pytest
from calculator import divide


def test_divide() -> None:
    assert divide(6, 3) == 2


def test_divide_by_zero_has_domain_error() -> None:
    with pytest.raises(ValueError, match="denominator cannot be zero"):
        divide(1, 0)
