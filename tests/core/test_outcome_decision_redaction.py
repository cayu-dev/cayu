"""A terminal decision holds only what its events would store: secrets redacted.

The decision and the transition receipt that installs it are durable before the
events they owe. A before-stop interruption whose reason names a registered
secret must leave the secret nowhere in the store, as its event never did.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest
from pydantic import SecretStr

from cayu import AgentSpec, CayuApp, Message, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.loop_policies import BeforeStopDecision, LoopPolicy
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor

_SECRET = "sk-outcome-decision-secret-4242"
_SESSION = "redacted-decision"


class _InterruptNamingSecret(LoopPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:outcome-redaction:interrupt",
            behavior_version="1",
            implementation_version="1",
        )

    async def before_stop(self, context) -> BeforeStopDecision:
        return BeforeStopDecision.interrupt(f"stopped with {_SECRET}")


def _app(store: Any) -> CayuApp:
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor([_SECRET]),
        loop_policies=[_InterruptNamingSecret()],
    )
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


def _codec() -> PublicAuthorityAliasCodec:
    key = base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
    )


async def _durable_text(store: Any, backend: str, tmp_path) -> str:
    if backend == "memory":
        return repr(vars(store))
    await store.close()
    return "".join(
        path.read_bytes().decode("utf-8", "replace") for path in tmp_path.glob("redaction.sqlite*")
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_terminal_decision_never_stores_a_redacted_secret(tmp_path, backend: str) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore(public_authority_alias_codec=_codec())
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "redaction.sqlite", public_authority_alias_codec=_codec()
            )
        )
        app = _app(store)
        try:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=_SESSION,
                    messages=[Message.text("user", "go")],
                )
            ):
                pass
            assert (await store.load(_SESSION)).status is SessionStatus.INTERRUPTED
        finally:
            await app.aclose()
        assert _SECRET not in await _durable_text(store, backend, tmp_path)

    asyncio.run(scenario())
