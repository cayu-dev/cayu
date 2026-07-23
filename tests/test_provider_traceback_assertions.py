from __future__ import annotations

import pytest

from tests.provider_traceback_assertions import is_cayu_source_filename


@pytest.mark.parametrize(
    "filename",
    (
        "/workspace/src/cayu/providers/openai.py",
        r"C:\workspace\src\cayu\providers\openai.py",
    ),
)
def test_cayu_source_filename_detection_is_platform_independent(filename: str) -> None:
    assert is_cayu_source_filename(filename) is True
