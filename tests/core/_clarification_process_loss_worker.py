"""Real public journey stopped before independent receiving reconciliation."""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from tests.core import _clarification_recovery_flow
from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export


async def main(value):
    class BackendFixtures:
        def getfixturevalue(self, name):
            if name != "postgres_dsn":
                raise AssertionError("Unexpected backend fixture requested.")
            return value["dsn"]

    async def before_reconciliation(initial, **kwargs):
        # The default call follows native RELEASE and lost settlement ACK.
        # The dispatched-phase hook instead stops inside the real transport.
        print(
            json.dumps({"pid": os.getpid(), "initial": initial.model_dump(mode="json")}), flush=True
        )
        await asyncio.Event().wait()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            _clarification_recovery_flow, "cleanup_in_fresh_process", before_reconciliation
        )
        if value.get("phase") == "dispatched":
            from cayu.runtime._temporary_continuation_permits import TemporaryServicePermitAuthority

            initializations = []
            register = TemporaryServicePermitAuthority.register
            dispatch = httpx.MockTransport.handle_async_request
            dispatch_count = 0

            async def registered(authority, candidate):
                result = await register(authority, candidate)
                initializations.append(authority.initialized)
                return result

            async def in_flight(transport, request):
                nonlocal dispatch_count
                dispatch_count += 1
                if dispatch_count == 3:
                    assert len(initializations) == 1
                    assert b"Which API version?" in request.content
                    # The real adapter has serialized and dispatched the
                    # authorized peer input. No response or RELEASE exists.
                    await before_reconciliation(initializations[0])
                return await dispatch(transport, request)

            patch.setattr(TemporaryServicePermitAuthority, "register", registered)
            patch.setattr(httpx.MockTransport, "handle_async_request", in_flight)
        await test_public_question_uses_real_assistant_export(
            value["backend"],
            Path(value["directory"]),
            BackendFixtures(),
            patch,
            temporary_service=True,
            public_reply=True,
            post_admission=True,
            side_session=False,
            maintenance_recovery="process_cleanup",
            budget_boundary=0 if value.get("phase") == "dispatched" else None,
        )
    raise AssertionError("The process-loss barrier unexpectedly returned.")


if __name__ == "__main__":
    asyncio.run(main(json.loads(sys.stdin.read())))
