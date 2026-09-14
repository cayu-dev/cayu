"""Public pending queries discover child-owned actions without mirroring content."""

import asyncio
from copy import deepcopy

import pytest
from pydantic import SecretStr
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.user_input import UserInputResponse
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.base import (
    InMemorySessionStore,
    InterruptSessionRequest,
    PendingActionKind,
    PendingActionQuery,
    RunRequest,
    SessionQuery,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


def _check_delegated_discovery(
    tmp_path,
    backend,
    caplog,
    capsys,
    *,
    action_kind,
    server_check=False,
    repeat_pause=False,
    lost_ack_monkeypatch=None,
    redacted_child_id=False,
):
    async def scenario():
        codec = (
            PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="test", keys={"test": SecretStr("A" * 43)}
                )
            )
            if redacted_child_id
            else None
        )
        store = (
            InMemorySessionStore(public_authority_alias_codec=codec)
            if backend == "memory"
            else SQLiteSessionStore(
                tmp_path / "pending-child.sqlite", public_authority_alias_codec=codec
            )
        )
        child_question = "private-child-question-canary"
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="action",
                        name="ask_user" if action_kind == "user_input" else "record",
                        arguments={"question": child_question}
                        if action_kind == "user_input"
                        else {"value": 7},
                    ),
                    ModelStreamEvent.completed(),
                ],
            ]
            + (
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="action", name="ask_user", arguments={"question": child_question}
                        ),
                        ModelStreamEvent.completed(),
                    ]
                ]
                if repeat_pause
                else []
            )
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor("cayu-child") if redacted_child_id else None,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("pending-child-discovery"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[UserInputTool()] if action_kind == "user_input" else [_RecordingTool()],
            tool_policy=None
            if action_kind == "user_input"
            else AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
            assert len(children.sessions) == 1
            child = children.sessions[0]
            if redacted_child_id and action_kind == "user_input":
                from cayu.approvals.user_input import pending_user_input_from_checkpoint
                from cayu.runtime._checkpoint_redaction import require_secret_free_durable_object

                private_checkpoint = await store.load_checkpoint(child.id)
                assert private_checkpoint is not None
                redactor = SecretRedactor("cayu-child")
                with pytest.raises(ValueError, match="workload secret"):
                    pending_user_input_from_checkpoint(private_checkpoint, redactor=redactor)
                trusted_pending = pending_user_input_from_checkpoint(
                    private_checkpoint, redactor=redactor, runtime_session=child
                )
                assert trusted_pending is not None
                assert trusted_pending.session_id == child.id
                # The session owner cannot authorize another incarnation or
                # exempt secret-bearing neighboring user/tool content.
                for candidate in (
                    {
                        **private_checkpoint,
                        "pending_user_input": {
                            **private_checkpoint["pending_user_input"],
                            "session_instance_id": "different-incarnation",
                        },
                    },
                    {
                        **private_checkpoint,
                        "pending_user_input": {
                            **private_checkpoint["pending_user_input"],
                            "question": "cayu-child",
                        },
                    },
                ):
                    with pytest.raises(ValueError, match="workload secret"):
                        require_secret_free_durable_object(
                            candidate,
                            redactor=redactor,
                            field_name="checkpoint",
                            runtime_session=child,
                        )
            child_page = await store.query_pending_actions(PendingActionQuery(session_id=child.id))
            assert child_page.issues == [] and len(child_page.actions) == 1, "\n".join(
                str(event.payload.get("error") or event.payload.get("reason") or event.type)
                for event in (await store.load_events(child.id))[-3:]
            )
            child_action = child_page.actions[0]
            if repeat_pause:
                before_checkpoint = await store.load_checkpoint("parent")
                assert before_checkpoint is not None
                before_wait = before_checkpoint["foreground_child_wait"]
                assert before_wait["revision"] == 1
                lost_ack = []
                if lost_ack_monkeypatch is not None:
                    publish = app._runtime_session_store.publish_session_operation

                    async def lose_refresh_ack(session_id, **kwargs):
                        result = await publish(session_id, **kwargs)
                        if (
                            kwargs["idempotency_key"].startswith("foreground-action:")
                            and not lost_ack
                        ):
                            lost_ack.append(kwargs["idempotency_key"])
                            raise ConnectionError("Refresh committed but acknowledgement lost")
                        return result

                    lost_ack_monkeypatch.setattr(
                        app._runtime_session_store, "publish_session_operation", lose_refresh_ack
                    )
                previous_action_id = child_action.input_id
                assert previous_action_id is not None
                _ = [
                    event
                    async for event in app.resolve_user_input(
                        UserInputResponse(
                            session_id=child.id, input_id=previous_action_id, answer="first answer"
                        )
                    )
                ]
                if lost_ack_monkeypatch is not None:
                    assert len(lost_ack) == 1
                    await app.recover_persisted_event_side_effects()
                    await app.recover_persisted_event_side_effects()
                after_checkpoint = await store.load_checkpoint("parent")
                assert after_checkpoint is not None
                after_wait = after_checkpoint["foreground_child_wait"]
                assert after_wait["revision"] == 2
                assert after_wait["parent_effect"] == before_wait["parent_effect"]
                refresh_events = [
                    event
                    for event in await store.load_events("parent")
                    if event.id.startswith("foreground-action:")
                ]
                assert len(refresh_events) == 1
                child_page = await store.query_pending_actions(
                    PendingActionQuery(session_id=child.id)
                )
                assert child_page.issues == [] and len(child_page.actions) == 1
                child_action = child_page.actions[0]
                assert child_action.input_id != previous_action_id
            assert child_action.kind == action_kind
            child_action_id = (
                child_action.input_id if action_kind == "user_input" else child_action.approval_id
            )
            page = await store.query_pending_actions(PendingActionQuery(session_id="parent"))
            assert page.issues == [] and len(page.actions) == 1
            action = page.actions[0]
            assert action.kind == "delegated_action", (
                list((await store.load_checkpoint("parent")) or {}),
                [
                    event.payload
                    for event in await store.load_events("parent")
                    if event.type == "session.interrupted"
                ],
            )
            from cayu import HumanAttentionRequest

            assert action.attention_id is None
            assert HumanAttentionRequest.from_pending_action(action) is None
            assert child_action.attention_id is not None
            assert HumanAttentionRequest.from_pending_action(child_action) is not None
            assert action.input_id is None and action.approval_id is None
            assert action.question is None and action.arguments is None
            public = action.model_dump(mode="json")
            assert public["delegated_action"] == {
                "child_session_id": child.id,
                "action_kind": action_kind,
                "action_id": child_action_id,
                "status": "waiting_on_child_action",
            }
            assert child_question not in action.model_dump_json()
            assert len(provider.requests) == (3 if repeat_pause else 2)
            if server_check:
                import httpx

                from cayu.server import ServerConfig, create_server

                server = create_server(app, config=ServerConfig.local_development())
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://127.0.0.1"
                ) as client:
                    response = await client.get(
                        "/api/pending-actions",
                        params={"session_id": "parent", "kind": "delegated_action"},
                    )
                    if redacted_child_id:
                        assert response.status_code == 200, response.text
                        reference = response.json()["actions"][0]["delegated_action"]
                        assert reference["child_session_id"] == app.project_session_id_for_exposure(
                            child.id
                        )
                        child_response = await client.get(
                            "/api/pending-actions",
                            params={"session_id": reference["child_session_id"]},
                        )
                        assert child_response.status_code == 200, child_response.text
                        assert len(child_response.json()["actions"]) == 1
                assert response.status_code == 200, response.text
                api_actions = response.json()["actions"]
                assert len(api_actions) == 1
                assert api_actions[0]["kind"] == "delegated_action"
                expected_reference = {
                    **public["delegated_action"],
                    "child_session_id": app.project_session_id_for_exposure(child.id),
                }
                assert api_actions[0]["delegated_action"] == expected_reference
                assert api_actions[0]["approval_id"] is None and api_actions[0]["input_id"] is None
                assert child_question not in response.text
            filtered = await store.query_pending_actions(
                PendingActionQuery(session_id="parent", kind=PendingActionKind.DELEGATED_ACTION)
            )
            assert filtered.actions == page.actions and filtered.issues == []
            source = await store.load_checkpoint("parent")
            assert source is not None
            original_wait = deepcopy(source["foreground_child_wait"])
            invalid_identity = "private-invalid-child-identity-canary"
            for field, value in (
                ("child_session_id", invalid_identity),
                ("child_action_id", True),
                ("child_action_kind", "unknown_future_action"),
            ):

                def replace_wait(_session, checkpoint, *, field=field, value=value):
                    return {
                        **checkpoint,
                        "foreground_child_wait": {**original_wait, field: value},
                    }

                try:
                    await store.transform_checkpoint("parent", replace_wait)
                    rejected = await store.query_pending_actions(
                        PendingActionQuery(session_id="parent")
                    )
                    assert rejected.actions == []
                    assert len(rejected.issues) == 1
                    assert rejected.issues[0].code == "source_invalid"
                    assert invalid_identity not in rejected.model_dump_json()
                finally:
                    await store.transform_checkpoint(
                        "parent",
                        lambda _session, checkpoint: {
                            **checkpoint,
                            "foreground_child_wait": deepcopy(original_wait),
                        },
                    )
            captured = capsys.readouterr()
            assert invalid_identity not in captured.out + captured.err + caplog.text
            restored = await store.query_pending_actions(PendingActionQuery(session_id="parent"))
            assert restored.actions == page.actions and restored.issues == []
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="parent", reason="Stop delegation")
                )
            ]
            after = await store.query_pending_actions(PendingActionQuery(session_id="parent"))
            assert not any(action.kind == "delegated_action" for action in after.actions)
            still_child_owned = await store.query_pending_actions(
                PendingActionQuery(session_id=child.id)
            )
            assert len(still_child_owned.actions) == 1
            assert still_child_owned.actions[0].input_id == child_action.input_id
            assert still_child_owned.actions[0].approval_id == child_action.approval_id
        finally:
            assert await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action_kind", ["user_input", "tool_approval"])
def test_parent_pending_query_exposes_only_delegated_child_action(
    tmp_path, backend, caplog, capsys, action_kind
):
    _check_delegated_discovery(tmp_path, backend, caplog, capsys, action_kind=action_kind)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action_kind", ["user_input", "tool_approval"])
def test_parent_pending_server_exposes_only_delegated_child_action(
    tmp_path, backend, caplog, capsys, action_kind
):
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    pytest.importorskip("httpx")
    _check_delegated_discovery(
        tmp_path, backend, caplog, capsys, action_kind=action_kind, server_check=True
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action_kind", ["user_input", "tool_approval"])
def test_parent_pending_server_child_link_survives_redaction(
    tmp_path, backend, caplog, capsys, action_kind
):
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    pytest.importorskip("httpx")
    _check_delegated_discovery(
        tmp_path,
        backend,
        caplog,
        capsys,
        action_kind=action_kind,
        server_check=True,
        redacted_child_id=True,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_parent_pending_query_tracks_repeated_child_input(tmp_path, backend, caplog, capsys):
    _check_delegated_discovery(
        tmp_path, backend, caplog, capsys, action_kind="user_input", repeat_pause=True
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_parent_action_refresh_survives_lost_ack(tmp_path, backend, caplog, capsys, monkeypatch):
    _check_delegated_discovery(
        tmp_path,
        backend,
        caplog,
        capsys,
        action_kind="user_input",
        repeat_pause=True,
        lost_ack_monkeypatch=monkeypatch,
    )
