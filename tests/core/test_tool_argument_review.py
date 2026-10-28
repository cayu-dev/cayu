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


def _keyring(seed: bytes = b"k" * 32, active: str = "one"):
    import base64

    from pydantic import SecretStr

    from cayu.runtime.public_authority import PublicAuthorityAliasKeyring

    encoded = base64.urlsafe_b64encode(seed).decode().rstrip("=")
    return PublicAuthorityAliasKeyring(active_key_id=active, keys={active: SecretStr(encoded)})


def test_without_a_binding_key_the_app_derives_one_from_its_alias_keyring() -> None:
    from cayu import CayuApp
    from cayu.approvals import derive_human_review_binding_key

    keyring = _keyring()
    first = ToolArgumentReview({"file_issue": ("title",)})
    CayuApp(human_review_policy=first, public_authority_alias_keyring=keyring)
    # Another worker or a restart with the same keyring derives the same key.
    second = ToolArgumentReview({"file_issue": ("title",)})
    CayuApp(human_review_policy=second, public_authority_alias_keyring=_keyring())
    assert first.binding_key == second.binding_key
    assert first.binding_key == derive_human_review_binding_key(keyring, purpose="approvals")
    assert len(first.binding_key) == 32

    other_purpose = ToolArgumentReview({"file_issue": ("title",)}, purpose="filing")
    CayuApp(human_review_policy=other_purpose, public_authority_alias_keyring=keyring)
    other_app = ToolArgumentReview({"file_issue": ("title",)})
    CayuApp(human_review_policy=other_app, public_authority_alias_keyring=_keyring(b"o" * 32))
    keys = {first.binding_key, other_purpose.binding_key, other_app.binding_key}
    assert len(keys) == 3
    # The derived key is not the alias key itself.
    assert b"k" * 32 not in keys


def test_a_durable_stores_keyring_serves_and_a_store_without_one_needs_a_key(tmp_path) -> None:
    from cayu import CayuApp
    from cayu.approvals import derive_human_review_binding_key
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec

    keyed = SQLiteSessionStore(
        tmp_path / "keyed.sqlite",
        public_authority_alias_codec=PublicAuthorityAliasCodec(_keyring()),
    )
    policy = ToolArgumentReview({"file_issue": ("title",)})
    CayuApp(session_store=keyed, human_review_policy=policy)
    assert policy.binding_key == derive_human_review_binding_key(_keyring(), purpose="approvals")

    # A durable store without a keyring has no key material to derive from.
    plain = SQLiteSessionStore(tmp_path / "plain.sqlite")
    with pytest.raises(ValueError, match="binding_key"):
        CayuApp(session_store=plain, human_review_policy=ToolArgumentReview({"file_issue": ("t",)}))
    explicit = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY)
    CayuApp(session_store=plain, human_review_policy=explicit)
    assert explicit.binding_key == KEY
    with pytest.raises(ValueError, match="binding_key"):
        ToolArgumentReview({"file_issue": ("title",)}).binding_key  # noqa: B018


def test_an_explicit_binding_key_wins_over_the_keyring() -> None:
    from cayu import CayuApp

    explicit = ToolArgumentReview({"file_issue": ("title",)}, binding_key=KEY)
    CayuApp(human_review_policy=explicit, public_authority_alias_keyring=_keyring())
    assert explicit.binding_key == KEY


def test_one_derived_policy_cannot_serve_apps_with_different_keyrings() -> None:
    from cayu import CayuApp

    policy = ToolArgumentReview({"file_issue": ("title",)})
    CayuApp(human_review_policy=policy, public_authority_alias_keyring=_keyring())
    CayuApp(human_review_policy=policy, public_authority_alias_keyring=_keyring())
    with pytest.raises(ValueError, match="another app"):
        CayuApp(human_review_policy=policy, public_authority_alias_keyring=_keyring(b"o" * 32))


def test_the_derived_key_follows_the_active_key() -> None:
    import base64

    from pydantic import SecretStr

    from cayu.approvals import derive_human_review_binding_key

    keyring = _keyring()
    rotated = keyring.rotated(
        active_key_id="two",
        key=SecretStr(base64.urlsafe_b64encode(b"t" * 32).decode().rstrip("=")),
    )
    assert derive_human_review_binding_key(
        keyring, purpose="approvals"
    ) != derive_human_review_binding_key(rotated, purpose="approvals")
    with pytest.raises(TypeError):
        derive_human_review_binding_key(b"x" * 32, purpose="approvals")
