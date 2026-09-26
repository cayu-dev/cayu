"""Genuine retained final production does not consume a temporary service continuation."""

from dataclasses import replace

import pytest
from tests.core import _clarification_final_flow, test_clarification_public
from tests.core.producer_clarification_latch import finish_with_native_producer
from tests.core.test_clarification_public import (
    test_public_question_uses_real_assistant_export as clarification_flow,
)


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("final_source", ["existing", "producer"])
async def test_retained_producer_final_latches_during_clarification_service(
    backend, final_source, tmp_path, request, monkeypatch
):
    registration = test_clarification_public.registration

    def combined_registration(**kwargs):
        original = registration(**kwargs)
        # This journey retains both clarification/service evidence and a separate
        # producer's mandatory settlement reservations in the same namespace.
        return replace(
            original,
            bootstrap=original.bootstrap.model_copy(
                update={
                    "limits": original.bootstrap.limits.model_copy(
                        update={
                            "operations": 512,
                            "events": 1024,
                            "retained_bytes": 32 * 1024 * 1024,
                        }
                    )
                }
            ),
        )

    monkeypatch.setattr(test_clarification_public, "registration", combined_registration)
    if final_source == "producer":
        setup = test_clarification_public.setup

        async def request_only(*args, **kwargs):
            values = list(await setup(*args, **kwargs))
            values[4] = values[4].model_copy(update={"cancellation": "stop"})
            return tuple(values)

        monkeypatch.setattr(test_clarification_public, "setup", request_only)
        monkeypatch.setattr(
            _clarification_final_flow, "finish_original_wait", finish_with_native_producer
        )
    await clarification_flow(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=False,
        post_admission=True,
        side_session=False,
        finish_request=True,
        final_latch_timing="during_return",
    )
