from __future__ import annotations

import asyncio
import html
from datetime import UTC, datetime

import pytest
from tests.core.test_human_review import QUESTION, decision, make_app, pause, resolve

from cayu.approvals import ToolArgumentReview
from cayu.approvals.review import (
    HumanReviewCall,
    HumanReviewContext,
    HumanReviewSource,
    build_review,
)
from cayu.storage import SQLiteSessionStore
from cayu.vaults import SecretRedactor

KEY = b"test-fixture-only-binding-key-32-bytes"
CONTEXT = HumanReviewContext(recipient="operator", purpose="approvals")


def _source(*calls: tuple[str, dict], scope: str = "static") -> HumanReviewSource:
    return HumanReviewSource(
        kind="tool_approval",
        interaction_id="approval-1",
        tool_round_id="round-1",
        tool_call_id="call-1",
        secret_resolution_scope=scope,
        calls=tuple(
            HumanReviewCall(tool_call_id=f"call-{n}", tool_name=name, on_grant="eligible")
            for n, (name, _) in enumerate(calls, start=1)
        ),
        arguments_by_call={f"call-{n}": args for n, (_, args) in enumerate(calls, start=1)},
    )


def _view(policy: ToolArgumentReview, source: HumanReviewSource, *, redactor=None):
    return build_review(
        policy=policy,
        context=CONTEXT,
        source=source,
        session_id="session-1",
        session_instance_id="instance-1",
        authoritative_content={"round": "round-1"},
        redactor=redactor if redactor is not None else SecretRedactor(),
        now=datetime.now(UTC),
    )


def _shown(view) -> list[tuple[str, str]]:
    return [(html.unescape(item.label), html.unescape(item.text)) for item in view.fields]


def test_an_approver_reads_the_listed_arguments_and_approves(tmp_path) -> None:
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        # The fixture's environment comes from a factory, so its rounds are in a
        # dynamic secret scope: listed arguments show only with attestation.
        policy = ToolArgumentReview(
            {"side_effect": ("value",)}, binding_key=KEY, attest_listed_arguments=True
        )
        app, _, tool = make_app(store, policy, approval=True)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "permitted", view
        assert _shown(view) == [("value", "morning")]
        await resolve(app, decision(view, approval=True))
        assert tool.calls == [{"value": "morning"}]
        await store.close()

    asyncio.run(scenario())


def test_without_attestation_a_dynamic_round_can_only_be_denied(tmp_path) -> None:
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ToolArgumentReview({"side_effect": ("value",)}, binding_key=KEY)
        app, _, _ = make_app(store, policy, approval=True)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "redacted"
        assert "morning" not in str(view.model_dump())
        await store.close()

    asyncio.run(scenario())


def test_tools_it_does_not_name_stay_approvable_without_their_arguments(tmp_path) -> None:
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY)
        app, _, tool = make_app(store, policy, approval=True)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "permitted", view
        assert _shown(view) == [("side_effect", "Arguments are not shown for this tool.")]
        assert "morning" not in str(view.model_dump())
        await resolve(app, decision(view, approval=True))
        assert tool.calls == [{"value": "morning"}]
        await store.close()

    asyncio.run(scenario())


def test_questions_in_dynamic_scope_require_explicit_attestation(tmp_path) -> None:
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        policy = ToolArgumentReview(
            {"file_issue": ("title",)}, binding_key=KEY, attest_questions=True
        )
        app, _, _ = make_app(store, policy)
        await pause(app)
        view = await app.inspect_human_review("review-session", context=CONTEXT)
        assert view.status == "permitted", view
        assert _shown(view) == [("Question", QUESTION)]
        await resolve(app, decision(view))
        await store.close()

    asyncio.run(scenario())


def test_long_arguments_are_split_and_the_rest_is_counted() -> None:
    body = "".join(f"line {n} <&> of the issue body\n" for n in range(4000))
    policy = ToolArgumentReview({"file_issue": ("title", "body")}, binding_key=KEY)
    view = _view(policy, _source(("file_issue", {"title": "[scout] Lost job", "body": body})))

    assert view.status == "permitted", view
    shown = _shown(view)
    assert shown[0] == ("title", "[scout] Lost job")
    assert shown[1][0].startswith("body (1/")
    assert shown[-1][0] == "Not shown"
    shown_body = "".join(text for label, text in shown[1:-1])
    assert body.startswith(shown_body)
    left_out = int(shown[-1][1].split(" ", 1)[0])
    assert len(shown_body) + left_out == len(body)
    assert len(view.fields) <= 32
    assert all(len(item.text) <= 4096 for item in view.fields)


def test_a_round_of_several_calls_names_each_call() -> None:
    policy = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY)
    view = _view(
        policy,
        _source(("file_issue", {"title": "One"}), ("run_command", {"command": "rm -rf /"})),
    )
    assert _shown(view) == [
        ("file_issue (call 1): title", "One"),
        ("run_command (call 2)", "Arguments are not shown for this tool."),
    ]


def test_listed_arguments_in_a_dynamic_scope_need_the_applications_attestation() -> None:
    source = _source(("file_issue", {"title": "One"}), scope="dynamic")
    plain = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY)
    attested = ToolArgumentReview(
        {"file_issue": ("title",)}, binding_key=KEY, attest_listed_arguments=True
    )
    assert _view(plain, source).status == "redacted"
    assert _shown(_view(attested, source)) == [("title", "One")]
    # Nothing private is shown for an unlisted tool, so no attestation is needed.
    unlisted = _source(("run_command", {"command": "ls"}), scope="dynamic")
    assert _view(plain, unlisted).status == "permitted"


def test_authorization_follows_purpose_and_the_optional_recipient_rule() -> None:
    def operators_only(context, *, session_id, session_metadata, action):
        return context.recipient == "operator"

    policy = ToolArgumentReview(
        {"file_issue": ("title",)}, binding_key=KEY, authorize_recipient=operators_only
    )
    kwargs = {"session_id": "s", "session_metadata": {}, "action": "decide"}
    assert policy.authorize(CONTEXT, **kwargs)
    assert not policy.authorize(CONTEXT.model_copy(update={"recipient": "guest"}), **kwargs)
    assert not policy.authorize(CONTEXT.model_copy(update={"purpose": "other"}), **kwargs)
    assert policy.dashboard_purpose == "approvals"


def test_version_changes_with_what_it_shows() -> None:
    one = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY)
    two = ToolArgumentReview({"file_issue": ("title", "body")}, binding_key=KEY)
    again = ToolArgumentReview({"file_issue": ("title",)}, binding_key=b"x" * 32)
    assert one.version != two.version
    assert one.version == again.version
    assert (
        one.version
        != ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY, revision="2").version
    )


@pytest.mark.parametrize(
    ("fields", "options"),
    [
        ({}, {}),
        ({"file_issue": ()}, {}),
        ({"file_issue": "title"}, {}),
        ({"file_issue": ("title",)}, {"binding_key": b"short"}),
        ({"file_issue": ("title",)}, {"purpose": " approvals"}),
    ],
)
def test_invalid_settings_are_refused(fields, options) -> None:
    with pytest.raises(ValueError):
        ToolArgumentReview(fields, **{"binding_key": KEY, **options})


@pytest.mark.parametrize("scope", ["static", "dynamic", "unknown"])
@pytest.mark.parametrize("attested", [False, True])
def test_questions_obey_secret_scope_and_explicit_question_attestation(scope, attested):
    from dataclasses import replace

    source = replace(
        _source(scope=scope),
        kind="user_input",
        question="Choose a destination",
        options=("one", "two"),
    )
    policy = ToolArgumentReview(
        {"file_issue": ("title",)}, binding_key=KEY, attest_questions=attested
    )
    assert _view(policy, source).status == (
        "permitted" if scope == "static" or attested else "redacted"
    )


@pytest.mark.parametrize("character", ['"', "'", "&", "<", "a"])
def test_escaping_heavy_text_splits_with_forward_progress(character):
    from cayu.approvals.tool_argument_review import _pieces

    text = character * 10000
    pieces = _pieces(text)
    assert "".join(pieces) == text
    assert all(piece and len(html.escape(piece)) <= 4096 for piece in pieces)


@pytest.mark.parametrize("kind", ["tool", "question", "option", "nested"])
@pytest.mark.parametrize("padding", [0, 4090, 20000])
def test_known_secrets_are_scanned_before_formatting_splitting_or_truncation(kind, padding):
    from dataclasses import replace

    secret = 'credential-"private"-123456789'
    text = "x" * padding + secret
    policy = ToolArgumentReview(
        {"file_issue": ("body",)},
        binding_key=KEY,
        attest_listed_arguments=True,
        attest_questions=True,
    )
    if kind in {"question", "option"}:
        source = replace(
            _source(scope="dynamic"),
            kind="user_input",
            question=text if kind == "question" else "Choose",
            options=(text,) if kind == "option" else (),
        )
    else:
        source = _source(("file_issue", {"body": {"value": text} if kind == "nested" else text}))
    view = _view(policy, source, redactor=SecretRedactor(secret))
    assert view.status == "redacted"
    assert view.fields == ()
