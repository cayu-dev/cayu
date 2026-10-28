"""Reading a selected argument whole when it is longer than the review view's limits."""

from __future__ import annotations

import asyncio
import html
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from tests.core.test_human_review import (
    ApprovalPolicy,
    DynamicFactory,
    Provider,
    RecordingTool,
    decision,
    identity,
    pause,
    resolve,
)

from cayu import AgentSpec, CayuApp
from cayu.approvals import ToolArgumentReview
from cayu.approvals.review import (
    HumanReviewCall,
    HumanReviewConflict,
    HumanReviewContext,
    HumanReviewDenied,
    HumanReviewDisclosure,
    HumanReviewField,
    HumanReviewLongField,
    HumanReviewPolicy,
    HumanReviewSource,
    build_long_field,
    build_review,
)
from cayu.environments import EnvironmentSpec
from cayu.storage import SQLiteSessionStore
from cayu.vaults import SecretRedactor

KEY = b"test-fixture-only-binding-key-32-bytes"
CONTEXT = HumanReviewContext(recipient="operator", purpose="approvals")
BODY = "".join(f"line {n} <&> of a long issue body\n" for n in range(2000))


def _source(arguments: dict, *, tool: str = "file_issue") -> HumanReviewSource:
    return HumanReviewSource(
        kind="tool_approval",
        interaction_id="approval-1",
        tool_round_id="round-1",
        tool_call_id="call-1",
        secret_resolution_scope="static",
        calls=(HumanReviewCall(tool_call_id="call-1", tool_name=tool, on_grant="eligible"),),
        arguments_by_call={"call-1": arguments},
    )


def _view(policy, source, redactor=None):
    return build_review(
        policy=policy,
        context=CONTEXT,
        source=source,
        session_id="session-1",
        session_instance_id="instance-1",
        authoritative_content={"round": "round-1"},
        redactor=redactor or SecretRedactor(),
        now=datetime.now(UTC),
    )


def _read(policy, source, key, redactor=None):
    view = _view(policy, source, redactor)
    return build_long_field(
        policy=policy,
        context=CONTEXT,
        source=source,
        view=view,
        key=key,
        redactor=redactor or SecretRedactor(),
    )


def test_a_value_the_view_shows_in_part_is_listed_and_read_whole() -> None:
    policy = ToolArgumentReview({"file_issue": ("title", "body")}, binding_key=KEY)
    source = _source({"title": "Lost job", "body": BODY, "token": "unselected"})
    view = _view(policy, source)

    assert view.status == "permitted"
    # The title fits one field, so only the body is a long field.
    assert [item.model_dump() for item in view.long_fields] == [
        {"key": "1.2", "label": "body", "characters": len(BODY)}
    ]
    field = _read(policy, source, "1.2")
    assert field.status == "permitted"
    assert html.unescape(field.text) == BODY
    assert field.characters == len(BODY)
    assert len(field.text) > 16 * 1024


def test_only_listed_fields_open_and_unselected_arguments_never_do() -> None:
    policy = ToolArgumentReview({"file_issue": ("title", "body")}, binding_key=KEY)
    source = _source({"title": "Lost job", "body": BODY, "token": "unselected"})
    # The title is shown whole in the view; it is not a long field.
    assert _read(policy, source, "1.1").status == "unavailable"
    for key in ("1.3", "2.1", "0.1", "1.02", "token", ""):
        assert _read(policy, source, key).status == "unavailable"
        assert policy.project_long_field(CONTEXT, source, key) is None
    unlisted = _source({"command": "x" * 20000}, tool="run_command")
    assert _view(policy, unlisted).long_fields == ()


def test_a_known_secret_anywhere_in_the_complete_value_withholds_it() -> None:
    secret = "credential-private-123456789"

    class SplitPolicy(HumanReviewPolicy):
        version = "split-v1"
        binding_key = KEY

        def authorize(self, context, *, session_id, session_metadata, action):
            return True

        def project(self, context, source):
            # Shows a clean prefix; the secret sits only past the shown part.
            return HumanReviewDisclosure(
                status="permitted",
                fields=(HumanReviewField(label="body", text="clean prefix"),),
                long_fields=(HumanReviewLongField(key="body", label="body", characters=99),),
            )

        def project_long_field(self, context, source, key):
            return "clean prefix " + "x" * 50000 + secret

    field = _read(SplitPolicy(), _source({}), "body", SecretRedactor(secret))
    assert field.status == "redacted"
    assert field.text is None


def test_a_redacted_view_lists_no_long_fields() -> None:
    policy = ToolArgumentReview({"file_issue": ("body",)}, binding_key=KEY)
    dynamic = replace(_source({"body": BODY}), secret_resolution_scope="dynamic")
    view = _view(policy, dynamic)
    assert view.status == "redacted"
    assert view.long_fields == ()
    assert _read(policy, dynamic, "1.1").status == "unavailable"


def test_long_fields_bind_into_the_reference() -> None:
    policy = ToolArgumentReview({"file_issue": ("body",)}, binding_key=KEY)
    short = _view(policy, _source({"body": "short"}))
    assert short.long_fields == ()
    longer = _view(policy, _source({"body": BODY}))
    shorter_long = _view(policy, _source({"body": BODY[:-1]}))
    assert longer.reference.content_tag != shorter_long.reference.content_tag


def test_the_existing_view_limits_still_hold() -> None:
    policy = ToolArgumentReview({"file_issue": ("body",)}, binding_key=KEY)
    view = _view(policy, _source({"body": BODY * 4}))
    assert len(view.fields) <= 32
    assert all(len(item.text) <= 4096 for item in view.fields)


def test_too_long_values_are_unavailable() -> None:
    policy = ToolArgumentReview({"file_issue": ("body",)}, binding_key=KEY)
    source = _source({"body": "x" * (1024 * 1024 + 1)})
    assert _view(policy, source).long_fields[0].key == "1.1"
    assert _read(policy, source, "1.1").status == "unavailable"


def _app(store, policy, value):
    provider = Provider([("call-1", "side_effect", {"value": value})])
    app = CayuApp(session_store=store, human_review_policy=policy, enable_logging=False)
    app.register_provider(provider, default=True)
    tool = RecordingTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"), tools=[tool], tool_policy=ApprovalPolicy()
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic", execution_profile_identity=identity("review-environment")),
        DynamicFactory(),
        default=True,
    )
    return app, tool


def test_an_approver_reads_a_long_argument_whole_then_approves(tmp_path) -> None:
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ToolArgumentReview(
            {"side_effect": ("value",)}, binding_key=KEY, attest_listed_arguments=True
        )
        app, tool = _app(store, policy, BODY)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "permitted", view
        [long_field] = view.long_fields
        field = await app.read_human_review_long_field(
            "review-session", reference=view.reference, key=long_field.key
        )
        assert html.unescape(field.text) == BODY
        await resolve(app, decision(view, approval=True))
        assert tool.calls == [{"value": BODY}]
        # The approval is settled, so its reference reads nothing any more.
        with pytest.raises(HumanReviewConflict):
            await app.read_human_review_long_field(
                "review-session", reference=view.reference, key=long_field.key
            )
        await store.close()

    asyncio.run(scenario())


def test_reading_needs_decide_permission_and_a_current_reference(tmp_path) -> None:
    def inspect_only(context, *, session_id, session_metadata, action):
        return action == "inspect" or context.recipient == "operator"

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ToolArgumentReview(
            {"side_effect": ("value",)},
            binding_key=KEY,
            attest_listed_arguments=True,
            authorize_recipient=inspect_only,
        )
        app, _ = _app(store, policy, BODY)
        await pause(app)
        guest = CONTEXT.model_copy(update={"recipient": "guest"})
        guest_view = await app.inspect_human_review("review-session", context=guest)
        assert guest_view.status == "permitted"
        with pytest.raises(HumanReviewDenied):
            await app.read_human_review_long_field(
                "review-session", reference=guest_view.reference, key="1.1"
            )
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        stale = view.reference.model_copy(update={"content_tag": "0" * 64})
        with pytest.raises(HumanReviewConflict):
            await app.read_human_review_long_field("review-session", reference=stale, key="1.1")
        await store.close()

    asyncio.run(scenario())


def test_a_large_round_stays_permitted_with_at_most_32_long_fields() -> None:
    calls = tuple(
        HumanReviewCall(tool_call_id=f"call-{n}", tool_name="file_issue", on_grant="eligible")
        for n in range(1, 65)
    )
    source = replace(
        _source({}),
        calls=calls,
        arguments_by_call={f"call-{n}": {"title": f"Issue {n}"} for n in range(1, 65)},
    )
    policy = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY)
    view = _view(policy, source)
    assert view.status == "permitted"
    assert len(view.long_fields) == 32
    assert html.unescape(view.fields[-1].label) == "Not shown"
    assert html.unescape(_read(policy, source, view.long_fields[0].key).text).startswith("Issue")
