"""Clarification must preserve real pending human-input authority."""

import pytest
from tests.core import test_clarification_public as journey


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "postgres"))
async def test_public_side_service_preserves_human_input(backend, tmp_path, request, monkeypatch):
    await journey.test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=True,
        human_paused=True,
    )
