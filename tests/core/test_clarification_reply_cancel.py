"""Question terminal election through real reply production and public entrances."""

import pytest
from tests.core import test_clarification_public as journey


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "postgres"))
@pytest.mark.parametrize("cancel_first", (False, True))
async def test_public_reply_and_cancellation_elect_once(
    backend, cancel_first, tmp_path, request, monkeypatch
):
    await journey.test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=False,
        reply_cancel_before=cancel_first,
    )
