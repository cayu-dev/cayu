from __future__ import annotations

from pathlib import Path

import pytest

import cayu
from tests.provider_traceback_assertions import (
    assert_cayu_traceback_does_not_retain,
    is_cayu_source_filename,
)


@pytest.mark.parametrize(
    "filename",
    (
        "/workspace/src/cayu/providers/openai.py",
        r"C:\workspace\src\cayu\providers\openai.py",
    ),
)
def test_cayu_source_filename_detection_is_platform_independent(filename: str) -> None:
    assert is_cayu_source_filename(filename) is True


def test_traceback_retention_check_inspects_the_active_installation() -> None:
    filename = str(Path(cayu.__file__).resolve().parent / "_retention_probe.py")
    namespace = {}
    exec(
        compile("def probe(value):\n    raise RuntimeError('probe')\n", filename, "exec"), namespace
    )
    retained = object()
    with pytest.raises(RuntimeError) as caught:
        namespace["probe"](retained)
    with pytest.raises(AssertionError):
        assert_cayu_traceback_does_not_retain(caught.value, retained)
    assert_cayu_traceback_does_not_retain(caught.value, object())
    assert not is_cayu_source_filename(str(Path(filename).parent.with_name("cayu_other") / "x.py"))
