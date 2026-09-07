from __future__ import annotations

import asyncio
import hashlib
import json
import traceback
import warnings
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from cayu.artifacts import ArtifactReadResult, LocalArtifactStore
from cayu.github_delivery import (
    GitHubCheckBundle,
    GitHubCheckObservation,
    GitHubCheckPolicy,
    GitHubCheckState,
    GitHubConnectorProfile,
    GitHubCredentials,
    GitHubDeliveryAdmissionError,
    GitHubDeliveryLimits,
    GitHubDeliveryReconstructionRequiredError,
    GitHubDeliveryRepository,
    GitHubDeliveryState,
    GitHubFeedbackObservation,
    GitHubLifecycleReceipt,
    GitHubOperation,
    GitHubProviderError,
    GitHubPullRequestConnector,
    GitHubPullRequestDeliveryRequest,
    GitHubPullRequestMetadata,
    GitHubPullRequestSnapshot,
    GitHubRepositoryAuthority,
    GitHubRepositoryConfig,
    GitHubRestTransport,
    GitHubReviewBundle,
    GitHubReviewPolicy,
    GitHubReviewState,
    GitHubSecurityAuthority,
    GitHubSourceAuthority,
    approve_github_delivery,
    github_connector_behavior_fingerprint,
    github_follow_up_coding_input,
    github_pull_request_delivery_request,
)
from cayu.vaults import REDACTED_SECRET, SecretRef, StaticVault


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _artifact_id(*parts: str) -> str:
    return "art_" + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:32]


def _request(
    *,
    mode: str = "create",
    number: int | None = None,
    labels=(),
    reviewers=(),
    draft: bool = True,
):
    operations = [
        GitHubOperation.CREATE_PULL_REQUEST
        if mode == "create"
        else GitHubOperation.UPDATE_PULL_REQUEST
    ]
    if labels:
        operations.append(GitHubOperation.SET_LABELS)
    if mode == "update" and not draft:
        operations.append(GitHubOperation.MARK_READY)
    if reviewers:
        operations.append(GitHubOperation.REQUEST_REVIEWERS)
    return GitHubPullRequestDeliveryRequest(
        connector_run_id="github-run",
        session_id="github-session",
        idempotency_key="github-idempotency",
        requested_at="2026-08-30T12:00:00+00:00",
        source=GitHubSourceAuthority(
            product_result_artifact_id="art_11111111111111111111111111111111",
            product_result_sha256=_digest("product"),
            product_request_fingerprint=_digest("product-request"),
            product_run_id="product-run",
            source_workspace_id="source-workspace",
            source_revision=_digest("source"),
            diff_artifact_id="art_33333333333333333333333333333333",
            diff_sha256=_digest("diff"),
            remote_result_artifact_id="art_22222222222222222222222222222222",
            remote_result_sha256=_digest("remote"),
            remote_request_fingerprint=_digest("remote-request"),
            remote_delivery_id="remote-delivery",
        ),
        repository=GitHubRepositoryAuthority(
            repository_id="repository",
            repository_alias="github",
            installation_id="installation",
            account_id="account",
            base_ref="refs/heads/main",
            expected_base_commit="1" * 40,
            head_ref="refs/heads/cayu/change",
            head_commit="2" * 40,
        ),
        mode=mode,
        existing_pull_request_number=number,
        metadata=GitHubPullRequestMetadata(
            title="Repair source",
            body="Bound to exact Cayu evidence.",
            labels=tuple(labels),
            reviewers=tuple(reviewers),
            draft=draft,
        ),
        checks=GitHubCheckPolicy(required_checks=("test",)),
        reviews=GitHubReviewPolicy(
            approval_required=True,
            required_approvers=("reviewer",),
            allow_follow_up=True,
        ),
        security=GitHubSecurityAuthority(
            connector_id="github-connector",
            connector_behavior_fingerprint=github_connector_behavior_fingerprint(),
            credential_profile_id="github-token",
            egress_profile_id="github-only",
            policy_fingerprint=_digest("policy"),
            approval_policy_fingerprint=_digest("approval"),
            redaction_profile_fingerprint=_digest("redaction"),
            allowed_operations=tuple(sorted(operations)),
        ),
        limits=GitHubDeliveryLimits(max_polls=3, max_elapsed_seconds=7200),
    )


def _pr(request, *, number=7, state="open", head_commit=None, labels=()):
    return GitHubPullRequestSnapshot(
        number=number,
        node_id=f"node-{number}",
        url=f"https://github.example/pr/{number}",
        state=state,
        draft=request.metadata.draft,
        base_ref=request.repository.base_ref,
        base_commit=request.repository.expected_base_commit,
        head_ref=request.repository.head_ref,
        head_commit=head_commit or request.repository.head_commit,
        title=request.metadata.title,
        body=request.metadata.body,
        labels=tuple(labels),
    )


class FakeTransport:
    def __init__(self, request):
        self.request = request
        self.pull_request = None
        self.create_calls = 0
        self.update_calls = 0
        self.ready_calls = 0
        self.reviewer_calls = 0
        self.check_bundles = [
            GitHubCheckBundle(
                head_commit=request.repository.head_commit,
                checks=(
                    GitHubCheckObservation(
                        provider_id="check-1",
                        name="test",
                        head_commit=request.repository.head_commit,
                        status="in_progress",
                    ),
                ),
            )
        ]
        self.review_bundle = GitHubReviewBundle(feedback=())
        self.error = None
        self.lose_create_ack = False

    async def observe_ref(self, config, ref, limits):
        del config, limits
        if self.error:
            raise self.error
        return (
            self.request.repository.expected_base_commit
            if ref == self.request.repository.base_ref
            else self.request.repository.head_commit
        )

    async def find_pull_requests(self, config, *, base_ref, head_ref, limits):
        del config, base_ref, head_ref, limits
        return () if self.pull_request is None else (self.pull_request,)

    async def get_pull_request(self, config, number, limits):
        del config, number, limits
        assert self.pull_request is not None
        return self.pull_request

    async def create_pull_request(self, config, request):
        del config
        self.create_calls += 1
        self.pull_request = _pr(request)
        if self.lose_create_ack:
            self.lose_create_ack = False
            raise GitHubProviderError("lost_ack", ambiguous=True)
        return self.pull_request, "request-create"

    async def update_pull_request(self, config, request, number):
        del config
        self.update_calls += 1
        self.pull_request = _pr(request, number=number)
        return self.pull_request, "request-update"

    async def set_labels(self, config, number, labels, limits):
        del config, number, limits
        assert self.pull_request is not None
        self.pull_request = self.pull_request.model_copy(update={"labels": labels})
        return "request-labels"

    async def request_reviewers(self, config, number, reviewers, teams, limits):
        del config, number, reviewers, teams, limits
        self.reviewer_calls += 1
        return "request-reviewers"

    async def mark_ready(self, config, number, limits):
        del config, number, limits
        self.ready_calls += 1
        assert self.pull_request is not None
        self.pull_request = self.pull_request.model_copy(update={"draft": False})
        return "request-ready"

    async def observe_checks(self, config, head_commit, limits):
        del config, head_commit, limits
        return self.check_bundles.pop(0) if len(self.check_bundles) > 1 else self.check_bundles[0]

    async def observe_reviews(self, config, number, limits):
        del config, number, limits
        return self.review_bundle


def _connector(
    tmp_path,
    request,
    transport,
    *,
    token="github-secret-token",
    validate_inputs=False,
):
    vault = StaticVault({"token": token})
    credentials = GitHubCredentials(
        credential_profile_id="github-token",
        token=SecretRef(name="token"),
        resolver=vault,
    )
    config = GitHubRepositoryConfig(
        alias="github",
        repository_id="repository",
        installation_id="installation",
        account_id="account",
        api_base_url="https://api.github.example",
        owner="cayu",
        name="runtime",
        credential_profile_id="github-token",
        egress_profile_id="github-only",
        credentials=credentials,
    )
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="github-artifacts")
    instant = datetime(2026, 8, 30, 12, 1, tzinfo=UTC)

    def clock():
        nonlocal instant
        instant += timedelta(seconds=60)
        return instant

    connector = GitHubPullRequestConnector(
        GitHubConnectorProfile(
            connector_id="github-connector",
            behavior_fingerprint=github_connector_behavior_fingerprint(),
            repositories={"github": config},
        ),
        repository=GitHubDeliveryRepository(store),
        transport=transport,
        clock=clock,
    )

    async def accept_inputs(*args, **kwargs):
        del args, kwargs

    if not validate_inputs:
        connector._validate_inputs = accept_inputs
    return connector, config


def test_create_requires_approval_then_polls_exact_head_idempotently(tmp_path):
    request = _request(labels=("cayu",))
    transport = FakeTransport(request)
    transport.check_bundles.append(
        GitHubCheckBundle(
            head_commit=request.repository.head_commit,
            checks=(
                GitHubCheckObservation(
                    provider_id="check-1",
                    name="test",
                    head_commit=request.repository.head_commit,
                    status="completed",
                    conclusion="success",
                ),
            ),
        )
    )
    transport.review_bundle = GitHubReviewBundle(
        feedback=(
            GitHubFeedbackObservation(
                provider_id="review-1",
                kind="review",
                head_commit=request.repository.head_commit,
                author_login="reviewer",
                author_type="User",
                state="approved",
                created_at="2026-08-30T12:00:30Z",
                body="Approved",
            ),
        )
    )
    connector, _ = _connector(tmp_path, request, transport)

    waiting = asyncio.run(connector.run(request, object(), object()))
    assert waiting.result.state == GitHubDeliveryState.APPROVAL_REQUIRED
    assert transport.create_calls == 0
    approval = approve_github_delivery(request, approval_id="approval-1")
    pending = asyncio.run(connector.run(request, object(), object(), approval=approval))
    settled = asyncio.run(connector.run(request, object(), object()))

    assert pending.result.state == GitHubDeliveryState.APPROVED
    assert pending.result.checks_state == GitHubCheckState.PENDING
    assert pending.result.next_poll_after_seconds == request.limits.poll_interval_seconds
    assert settled.result.state == GitHubDeliveryState.APPROVED
    assert settled.result.checks_state == GitHubCheckState.PASSED
    assert settled.result.pull_request.head_commit == request.repository.head_commit
    assert settled.result.session_id == request.session_id
    assert settled.result.idempotency_key == request.idempotency_key
    assert settled.result.required_checks == request.checks.required_checks
    assert settled.result.allowed_operations == request.security.allowed_operations
    assert settled.result.check_policy_fingerprint == request.checks.fingerprint
    assert settled.result.review_policy_fingerprint == request.reviews.fingerprint
    assert settled.result.approval_id == approval.approval_id
    assert settled.result.approval_fingerprint == approval.fingerprint
    assert settled.result.redaction_profile_fingerprint == (
        request.security.redaction_profile_fingerprint
    )
    assert transport.create_calls == 1
    assert asyncio.run(connector.run(request, object(), object(), approval=approval)) == settled


@pytest.mark.parametrize("reviewed_head", [None, "3" * 40, "2" * 40])
def test_review_settlement_requires_exact_reviewed_commit(tmp_path, reviewed_head):
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    transport.review_bundle = GitHubReviewBundle(
        feedback=(
            GitHubFeedbackObservation(
                provider_id="review-exact",
                kind="review",
                author_login="reviewer",
                author_type="User",
                state="approved",
                created_at="2026-08-30T12:00:30Z",
                body="Approved",
                head_commit=reviewed_head,
            ),
        )
    )
    connector, _ = _connector(tmp_path, request, transport)
    outcome = asyncio.run(connector.run(request, object(), object()))
    assert (outcome.result.review_state is GitHubReviewState.APPROVED) == (
        reviewed_head == request.repository.head_commit
    )


def test_restart_after_reviewer_effect_without_ack_never_repeats(tmp_path, monkeypatch):
    request = _request(reviewers=("reviewer",))
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    connector, _ = _connector(tmp_path, request, transport)
    publish = connector.repository.publish

    async def fail_settlement(current, result):
        if result.operations and result.operations[-1].status == "succeeded":
            raise OSError("receipt unavailable")
        return await publish(current, result)

    monkeypatch.setattr(connector.repository, "publish", fail_settlement)
    with pytest.raises(GitHubProviderError, match="provider_extension_failure"):
        asyncio.run(
            connector.run(
                request,
                object(),
                object(),
                approval=approve_github_delivery(request, approval_id="reviewer-approval"),
            )
        )
    assert transport.reviewer_calls == 1

    restarted, _ = _connector(tmp_path, request, transport)
    restarted.clock = lambda: datetime(2026, 8, 30, 12, 20, tzinfo=UTC)
    outcome = asyncio.run(restarted.run(request, object(), object()))
    assert outcome.result.state is GitHubDeliveryState.AMBIGUOUS
    assert transport.reviewer_calls == 1


@pytest.mark.parametrize("operation", ["create", "reviewers"])
@pytest.mark.parametrize("failure_code", ["provider_unavailable", "rate_limited"])
def test_recovery_read_failure_cannot_erase_unresolved_effect(
    tmp_path, monkeypatch, operation, failure_code
):
    request = _request(reviewers=("reviewer",))
    transport = FakeTransport(request)
    if operation == "reviewers":
        transport.pull_request = _pr(request)
    connector, _ = _connector(tmp_path, request, transport)
    now = datetime(2026, 8, 30, 12, 1, tzinfo=UTC)
    connector.clock = lambda: now
    publish = connector.repository.publish

    async def fail_settlement(current, result):
        if result.operations and result.operations[-1].status == "succeeded":
            raise OSError("settlement unavailable after provider effect")
        return await publish(current, result)

    monkeypatch.setattr(connector.repository, "publish", fail_settlement)

    async def scenario():
        nonlocal now
        with pytest.raises(GitHubProviderError):
            await connector.run(
                request,
                object(),
                object(),
                approval=approve_github_delivery(request, approval_id="recovery-approval"),
            )
        intent = await connector.repository.latest(request)
        assert intent.result.reason_code == "provider_mutation_in_flight"
        transport.error = GitHubProviderError(failure_code, retryable=True)
        restarted, _ = _connector(tmp_path, request, transport)
        restarted.clock = lambda: now
        now += timedelta(seconds=60)
        unavailable = await restarted.run(request, object(), object())
        assert unavailable.result.state is (
            GitHubDeliveryState.RATE_LIMITED
            if failure_code == "rate_limited"
            else GitHubDeliveryState.PROVIDER_UNAVAILABLE
        )
        assert unavailable.result.operations == intent.result.operations
        transport.error = None
        recovered, _ = _connector(tmp_path, request, transport)
        recovered.clock = lambda: now
        now += timedelta(seconds=60)
        outcome = await recovered.run(request, object(), object())
        if operation == "reviewers":
            assert outcome.result.state is GitHubDeliveryState.AMBIGUOUS
            assert outcome.result.operations[-1].status == "ambiguous"
        else:
            assert outcome.result.state is GitHubDeliveryState.CHECKS_PENDING
            assert any(item.status == "reconciled" for item in outcome.result.operations)
        assert transport.create_calls == (1 if operation == "create" else 0)
        assert transport.reviewer_calls == 1

    asyncio.run(scenario())


def test_reconciled_create_resumes_remaining_approved_effects(tmp_path):
    request = _request(labels=("cayu",), reviewers=("reviewer",))
    transport = FakeTransport(request)
    transport.lose_create_ack = True
    connector, _ = _connector(tmp_path, request, transport)
    first = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="all-effects"),
        )
    )
    assert first.result.state is GitHubDeliveryState.PR_UPDATED
    assert transport.reviewer_calls == 0
    second = asyncio.run(connector.run(request, object(), object()))
    assert second.result.state is GitHubDeliveryState.CHECKS_PENDING
    assert transport.pull_request.labels == ("cayu",)
    assert transport.create_calls == 1
    assert transport.reviewer_calls == 1


def test_documented_scheduler_finishes_reconciled_progress(tmp_path):
    request = _request(labels=("cayu",), reviewers=("reviewer",))
    transport = FakeTransport(request)
    transport.lose_create_ack = True
    transport.check_bundles = [
        GitHubCheckBundle(
            head_commit=request.repository.head_commit,
            checks=(
                GitHubCheckObservation(
                    provider_id="check-run:1",
                    name="test",
                    head_commit=request.repository.head_commit,
                    status="completed",
                    conclusion="success",
                ),
            ),
        )
    ]
    transport.review_bundle = GitHubReviewBundle(
        feedback=(
            GitHubFeedbackObservation(
                provider_id="review:1",
                kind="review",
                head_commit=request.repository.head_commit,
                author_login="reviewer",
                author_type="User",
                state="approved",
                created_at="2026-08-30T12:00:30Z",
                body="Approved",
            ),
        )
    )
    connector, _ = _connector(tmp_path, request, transport)
    now = datetime(2026, 8, 30, 12, 1, tzinfo=UTC)
    connector.clock = lambda: now

    def clock():
        return now

    async def scenario():
        nonlocal now, connector
        outcome = await connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="scheduled-approval"),
        )
        assert outcome.result.state is GitHubDeliveryState.PR_UPDATED
        for _ in range(4):
            # Follow the generated application rule, including after restart.
            delay = outcome.result.next_poll_after_seconds
            if delay is None:
                break
            assert outcome.result.next_poll_at is not None
            connector, _ = _connector(tmp_path, request, transport)
            connector.clock = clock
            early = await connector.run(request, object(), object())
            assert early == outcome
            now += timedelta(seconds=delay)
            outcome = await connector.run(request, object(), object())
        assert outcome.result.state is GitHubDeliveryState.APPROVED
        assert outcome.result.checks_state is GitHubCheckState.PASSED
        assert outcome.result.next_poll_after_seconds is None
        assert transport.pull_request.labels == ("cayu",)
        assert transport.create_calls == transport.reviewer_calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("metadata_changes", [False, True])
def test_ready_pr_rejects_requested_draft_before_effects(tmp_path, metadata_changes):
    request = _request(mode="update", number=7, reviewers=("reviewer",), labels=("cayu",))
    transport = FakeTransport(request)
    transport.pull_request = _pr(request).model_copy(
        update={
            "draft": False,
            "title": "Old title" if metadata_changes else request.metadata.title,
        }
    )
    original = transport.pull_request
    connector, _ = _connector(tmp_path, request, transport)
    outcome = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="draft-approval"),
        )
    )
    assert outcome.result.state is GitHubDeliveryState.DENIED
    assert outcome.result.reason_code == "ready_to_draft_not_supported"
    assert outcome.result.operations == ()
    assert not transport.pull_request.draft
    assert transport.pull_request == original
    assert (
        transport.create_calls
        == transport.update_calls
        == transport.reviewer_calls
        == transport.ready_calls
        == 0
    )


def test_poll_cadence_survives_connector_restart(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    connector, _ = _connector(tmp_path, request, transport)
    instant = datetime(2026, 8, 30, 12, 1, tzinfo=UTC)
    connector.clock = lambda: instant
    first = asyncio.run(connector.run(request, object(), object()))
    assert first.result.poll_count == 1
    restarted, _ = _connector(tmp_path, request, transport)
    restarted.clock = lambda: instant
    early = asyncio.run(restarted.run(request, object(), object()))
    assert early == first
    instant += timedelta(seconds=request.limits.poll_interval_seconds)
    next_poll = asyncio.run(restarted.run(request, object(), object()))
    assert next_poll.result.poll_count == 2


@pytest.mark.parametrize("field,value", [("number", 8), ("head_commit", "3" * 40)])
def test_mutation_reply_cannot_redirect_following_effects(tmp_path, field, value):
    request = _request(mode="update", number=7, reviewers=("reviewer",))
    transport = FakeTransport(request)
    transport.pull_request = _pr(request).model_copy(update={"body": "old body"})

    async def wrong_reply(config, current, number):
        del config, number
        transport.update_calls += 1
        return _pr(current).model_copy(update={field: value}), "update-ack"

    transport.update_pull_request = wrong_reply
    connector, _ = _connector(tmp_path, request, transport)
    outcome = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="wrong-reply"),
        )
    )
    assert outcome.result.state is GitHubDeliveryState.AMBIGUOUS
    assert transport.reviewer_calls == 0


@pytest.mark.parametrize("signal", [SystemExit, KeyboardInterrupt, GeneratorExit])
@pytest.mark.parametrize("grouped", [False, True])
def test_owner_delivers_detached_process_signal_to_normal_handler(tmp_path, signal, grouped):
    request = _request()
    transport = FakeTransport(request)
    secret = "private-signal-canary"

    async def fail(*args):
        del args
        if grouped:
            raise BaseExceptionGroup(secret, [signal(secret), RuntimeError(secret)])
        raise signal(secret)

    transport.observe_ref = fail
    connector, _ = _connector(tmp_path, request, transport, token=secret)

    async def exercise():
        try:
            await connector.run(request, object(), object())
        except signal as error:
            assert secret not in repr(error)
            assert secret not in repr(error.__cause__)
        else:
            pytest.fail("process signal was not propagated")

    asyncio.run(exercise())


def test_generic_transport_failure_is_safe_and_durably_ambiguous(tmp_path, caplog, capsys):
    request = _request()
    transport = FakeTransport(request)
    token = "generic-transport-canary"

    async def fail(config, current):
        del config
        transport.pull_request = _pr(current)
        raise RuntimeError(token)

    transport.create_pull_request = fail
    connector, _ = _connector(tmp_path, request, transport, token=token)
    result = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="failure-approval"),
        )
    )
    assert result.result.state is GitHubDeliveryState.AMBIGUOUS
    assert token not in result.result.model_dump_json()
    assert token not in caplog.text
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err


def test_mutated_provider_snapshot_emits_no_serializer_secret_warning(tmp_path, caplog, capsys):
    request = _request()
    token = "mutated-provider-snapshot-secret"
    transport = FakeTransport(request)
    transport.pull_request = _pr(request).model_copy(update={"number": token})
    connector, _ = _connector(tmp_path, request, transport, token=token)
    with (
        warnings.catch_warnings(record=True) as captured,
        pytest.raises(GitHubProviderError) as failure,
    ):
        asyncio.run(connector.run(request, object(), object()))
    output = capsys.readouterr()
    diagnostic = "".join(traceback.format_exception(failure.value))
    assert token not in diagnostic + caplog.text + output.out + output.err + str(captured)
    assert transport.create_calls == 0


def test_final_observation_preserves_closed_pr_outcome(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)

    async def reviews(*args):
        transport.pull_request = transport.pull_request.model_copy(update={"state": "closed"})
        return transport.review_bundle

    transport.observe_reviews = reviews
    connector, _ = _connector(tmp_path, request, transport)
    result = asyncio.run(connector.run(request, object(), object()))
    assert result.result.state is GitHubDeliveryState.CLOSED
    assert result.result.pull_request.state == "closed"
    assert result.result.reason_code == "pull_request_closed_during_observation"
    assert asyncio.run(connector.repository.latest(request)) == result


@pytest.mark.parametrize("during_reconciliation", [False, True])
def test_cancellation_publication_failure_keeps_signal(
    tmp_path, caplog, capsys, during_reconciliation
):
    request = _request()
    transport = FakeTransport(request)
    token = "cancel-publication-secret"
    connector, _ = _connector(tmp_path, request, transport, token=token)
    original_publish = connector.repository.publish
    failed_publications = []

    async def publish(current, result):
        if result.reason_code == "provider_mutation_cancelled_ambiguous":
            failed_publications.append(result)
            raise RuntimeError(token)
        return await original_publish(current, result)

    connector.repository.publish = publish

    async def scenario():
        entered = asyncio.Event()
        owners = []

        async def block():
            owners.append(asyncio.current_task())
            entered.set()
            await asyncio.Event().wait()

        async def create(*args):
            transport.create_calls += 1
            if during_reconciliation:
                raise GitHubProviderError("lost_ack", ambiguous=True)
            await block()

        original_find = transport.find_pull_requests

        async def find(*args, **kwargs):
            if transport.create_calls:
                await block()
            return await original_find(*args, **kwargs)

        transport.create_pull_request = create
        transport.find_pull_requests = find
        task = asyncio.create_task(
            connector.run(
                request,
                object(),
                object(),
                approval=approve_github_delivery(request, approval_id="cancel-approval"),
            )
        )
        await entered.wait()
        owners[0].cancel()
        assert owners[0].cancelling() == 1
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled()
        assert len(failed_publications) == 1
        assert caught.value.__cause__ is not None
        diagnostic = "".join(traceback.format_exception(caught.value))
        assert token not in diagnostic
        latest = await connector.repository.latest(request)
        assert latest.result.reason_code == "provider_mutation_in_flight"
        assert transport.create_calls == 1

    with warnings.catch_warnings(record=True) as captured:
        asyncio.run(scenario())
    output = capsys.readouterr()
    assert token not in caplog.text + output.out + output.err + str(captured)


def test_rest_cleanup_failure_does_not_expose_response_credentials(tmp_path, caplog, capsys):
    request = _request()
    token = "cleanup-secret-canary"
    _, config = _connector(tmp_path, request, FakeTransport(request), token=token)

    class BrokenClose(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps({"object": {"sha": "1" * 40}}).encode()

        async def aclose(self):
            raise RuntimeError(token)

    async def handler(current):
        del current
        return httpx.Response(200, stream=BrokenClose())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(GitHubProviderError) as captured:
                await GitHubRestTransport(client).observe_ref(
                    config, request.repository.base_ref, request.limits
                )
            assert token not in "".join(traceback.format_exception(captured.value))
            assert captured.value.__context__ is None

    with warnings.catch_warnings(record=True) as recorded:
        asyncio.run(exercise())
    assert all(token not in str(item.message) for item in recorded)
    assert token not in caplog.text
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err


@pytest.mark.parametrize("repository", [None, {"full_name": "other/runtime"}])
def test_rest_pr_must_prove_the_configured_head_repository(tmp_path, repository):
    request = _request()
    _, config = _connector(tmp_path, request, FakeTransport(request))

    async def handler(current):
        del current
        return httpx.Response(
            200,
            json={"base": {"repo": {"full_name": "cayu/runtime"}}, "head": {"repo": repository}},
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(GitHubProviderError, match="repository_mismatch"):
                await GitHubRestTransport(client).get_pull_request(config, 7, request.limits)

    asyncio.run(exercise())


def test_request_factory_rejects_untyped_upstream_placeholders():
    template = _request()
    with pytest.raises(TypeError, match="CodingProductPublication"):
        github_pull_request_delivery_request(
            object(),
            object(),
            connector_run_id=template.connector_run_id,
            session_id=template.session_id,
            idempotency_key=template.idempotency_key,
            requested_at=template.requested_at,
            repository_alias=template.repository.repository_alias,
            installation_id=template.repository.installation_id,
            account_id=template.repository.account_id,
            mode="create",
            existing_pull_request_number=None,
            metadata=template.metadata,
            checks=template.checks,
            reviews=template.reviews,
            security=template.security,
            limits=template.limits,
        )


def test_lost_create_ack_reconciles_without_duplicate_pull_request(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    transport.lose_create_ack = True
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="approval-1")

    result = asyncio.run(connector.run(request, object(), object(), approval=approval))

    assert result.result.state == GitHubDeliveryState.PR_UPDATED
    assert transport.create_calls == 1
    assert result.result.operations[-1].status == "reconciled"


def test_concurrent_same_run_calls_cross_one_mutation_boundary(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="approval-concurrent")

    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_create(config, current):
            del config
            transport.create_calls += 1
            entered.set()
            await release.wait()
            transport.pull_request = _pr(current)
            return transport.pull_request, "request-create"

        transport.create_pull_request = blocked_create
        first = asyncio.create_task(connector.run(request, object(), object(), approval=approval))
        await entered.wait()
        second = asyncio.create_task(connector.run(request, object(), object(), approval=approval))
        with pytest.raises(GitHubDeliveryAdmissionError, match="unsettled"):
            await second
        release.set()
        return await first

    first = asyncio.run(exercise())

    assert first.result.pull_request is not None
    assert transport.create_calls == 1


def test_failed_lost_ack_reconciliation_is_durable_and_recoverable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = _request()
    transport = FakeTransport(request)
    transport.lose_create_ack = True
    original_find = transport.find_pull_requests
    find_calls = 0

    async def fail_second_find(*args, **kwargs):
        nonlocal find_calls
        find_calls += 1
        if find_calls == 2:
            raise GitHubProviderError("provider_unavailable", retryable=True)
        return await original_find(*args, **kwargs)

    monkeypatch.setattr(transport, "find_pull_requests", fail_second_find)
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="approval-reconcile")

    ambiguous = asyncio.run(connector.run(request, object(), object(), approval=approval))
    assert ambiguous.result.state is GitHubDeliveryState.AMBIGUOUS
    assert ambiguous.result.reason_code == "provider_mutation_reconciliation_unavailable"
    assert ambiguous.result.next_poll_after_seconds == request.limits.poll_interval_seconds

    monkeypatch.setattr(transport, "find_pull_requests", original_find)
    recovered = asyncio.run(connector.run(request, object(), object(), approval=approval))
    assert recovered.result.pull_request is not None
    assert recovered.result.state is not GitHubDeliveryState.AMBIGUOUS
    assert recovered.result.operations[-1].status == "reconciled"
    assert transport.create_calls == 1


def test_lost_label_ack_reconciles_exact_state_without_duplicate_write(tmp_path):
    request = _request(labels=("cayu",))
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    label_calls = 0

    async def lose_label_ack(config, number, labels, limits):
        nonlocal label_calls
        del config, number, limits
        label_calls += 1
        transport.pull_request = transport.pull_request.model_copy(update={"labels": labels})
        raise GitHubProviderError("lost_label_ack", ambiguous=True)

    transport.set_labels = lose_label_ack
    connector, _ = _connector(tmp_path, request, transport)
    result = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="approval-label"),
        )
    )

    assert result.result.state is GitHubDeliveryState.PR_UPDATED
    assert label_calls == 1
    assert result.result.operations[-1].operation is GitHubOperation.SET_LABELS
    assert result.result.operations[-1].status == "reconciled"


def test_dynamic_unapproved_update_of_matching_create_target_is_denied(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request).model_copy(update={"body": "provider drift"})
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="approval-create-only"),
        )
    )

    assert result.result.state is GitHubDeliveryState.DENIED
    assert result.result.reason_code == "provider_operation_not_allowed"
    assert transport.update_calls == 0


def test_mismatched_pull_request_is_rejected_before_metadata_mutation(tmp_path):
    request = _request(mode="update", number=7)
    transport = FakeTransport(request)
    transport.pull_request = _pr(request, head_commit="3" * 40).model_copy(
        update={"body": "provider drift"}
    )
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="approval-mismatch"),
        )
    )

    assert result.result.state is GitHubDeliveryState.SUPERSEDED
    assert result.result.reason_code == "pull_request_binding_changed_before_mutation"
    assert transport.update_calls == 0


def test_failed_provider_mutation_retains_explicit_operation_evidence(tmp_path):
    request = _request(labels=("cayu",))
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)

    async def reject_labels(config, number, labels, limits):
        del config, number, labels, limits
        raise GitHubProviderError("provider_rejected")

    transport.set_labels = reject_labels
    connector, _ = _connector(tmp_path, request, transport)
    result = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="approval-label"),
        )
    )

    assert result.result.state is GitHubDeliveryState.FAILED
    assert result.result.operations[-1].operation is GitHubOperation.SET_LABELS
    assert result.result.operations[-1].status == "failed"


def test_ambiguous_reviewer_request_is_terminal_and_never_repeated(tmp_path):
    request = _request(reviewers=("reviewer",))
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)

    async def lose_reviewer_ack(config, number, reviewers, teams, limits):
        del config, number, reviewers, teams, limits
        transport.reviewer_calls += 1
        raise GitHubProviderError("lost_reviewer_ack", ambiguous=True)

    transport.request_reviewers = lose_reviewer_ack
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="approval-reviewer")

    ambiguous = asyncio.run(connector.run(request, object(), object(), approval=approval))
    recovered = asyncio.run(connector.run(request, object(), object(), approval=approval))

    assert ambiguous.result.state is GitHubDeliveryState.AMBIGUOUS
    assert ambiguous.result.operations[-1].operation is GitHubOperation.REQUEST_REVIEWERS
    assert ambiguous.result.operations[-1].status == "ambiguous"
    assert recovered == ambiguous
    assert transport.reviewer_calls == 1


@pytest.mark.parametrize(
    ("error", "state"),
    [
        (GitHubProviderError("rate_limited", retryable=True), GitHubDeliveryState.RATE_LIMITED),
        (
            GitHubProviderError("provider_unavailable", retryable=True),
            GitHubDeliveryState.PROVIDER_UNAVAILABLE,
        ),
        (GitHubProviderError("permission_denied"), GitHubDeliveryState.PERMISSION_DENIED),
        (GitHubProviderError("provider_timeout", ambiguous=True), GitHubDeliveryState.AMBIGUOUS),
    ],
)
def test_provider_failure_classification_is_truthful(tmp_path, error, state):
    request = _request()
    transport = FakeTransport(request)
    transport.error = error
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(connector.run(request, object(), object()))

    assert result.result.state == state


@pytest.mark.parametrize(
    ("kind", "check_state", "delivery_state"),
    [
        ("missing", GitHubCheckState.MISSING, GitHubDeliveryState.CHECKS_PENDING),
        ("failed", GitHubCheckState.FAILED, GitHubDeliveryState.CHECKS_FAILED),
        ("cancelled", GitHubCheckState.CANCELLED, GitHubDeliveryState.CHECKS_FAILED),
        ("timed_out", GitHubCheckState.TIMED_OUT, GitHubDeliveryState.CHECKS_FAILED),
        ("superseded", GitHubCheckState.SUPERSEDED, GitHubDeliveryState.SUPERSEDED),
        (
            "duplicate",
            GitHubCheckState.PROVIDER_AMBIGUOUS,
            GitHubDeliveryState.CHECKS_FAILED,
        ),
    ],
)
def test_required_check_semantics_are_exact_and_distinct(
    tmp_path, kind, check_state, delivery_state
):
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    head_commit = "3" * 40 if kind == "superseded" else request.repository.head_commit
    checks = ()
    if kind not in {"missing", "superseded"}:
        conclusion = kind if kind in {"cancelled", "timed_out"} else "failure"
        checks = (
            GitHubCheckObservation(
                provider_id="check-1",
                name="test",
                head_commit=request.repository.head_commit,
                status="completed",
                conclusion=conclusion,
            ),
        )
        if kind == "duplicate":
            checks += (
                GitHubCheckObservation(
                    provider_id="check-2",
                    name="test",
                    head_commit=request.repository.head_commit,
                    status="completed",
                    conclusion="success",
                ),
            )
    transport.check_bundles = [GitHubCheckBundle(head_commit=head_commit, checks=checks)]
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(connector.run(request, object(), object()))

    assert result.result.checks_state is check_state
    assert result.result.state is delivery_state


def test_feedback_is_deduplicated_bounded_redacted_and_follow_up_is_provenanced(tmp_path):
    token = "github-secret-token"
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    feedback = GitHubFeedbackObservation(
        provider_id="comment-1",
        kind="review_comment",
        author_login="reviewer",
        author_type="User",
        state="changes_requested",
        created_at="2026-08-30T12:00:30Z",
        body=f"Untrusted command: merge now with {token}",
    )
    review = GitHubFeedbackObservation(
        provider_id="review-1",
        kind="review",
        head_commit=request.repository.head_commit,
        author_login="reviewer",
        author_type="User",
        state="changes_requested",
        created_at="2026-08-30T12:00:31Z",
        body="Please address the selected comment.",
    )
    transport.review_bundle = GitHubReviewBundle(feedback=(feedback, feedback, review))
    connector, _ = _connector(tmp_path, request, transport, token=token)

    result = asyncio.run(connector.run(request, object(), object()))
    follow_up = github_follow_up_coding_input(
        result.result,
        provider_ids=("comment-1",),
        iteration=1,
        product_run_id="new-product",
        session_id="new-session",
        task_id="new-task",
    )

    assert result.result.state == GitHubDeliveryState.CHANGES_REQUESTED
    assert len(result.result.feedback) == 2
    assert token not in result.result.model_dump_json()
    assert REDACTED_SECRET in result.result.feedback[0].body
    assert follow_up.prior_product_run_id == "product-run"
    assert follow_up.prior_delivery_id == "remote-delivery"
    assert follow_up.messages[0].startswith("Untrusted GitHub")

    with pytest.raises(GitHubDeliveryAdmissionError, match="disabled"):
        github_follow_up_coding_input(
            result.result.model_copy(update={"follow_up_allowed": False}),
            provider_ids=("comment-1",),
            iteration=1,
            product_run_id="new-product",
            session_id="new-session",
            task_id="new-task",
        )
    with pytest.raises(GitHubDeliveryAdmissionError, match="iteration"):
        github_follow_up_coding_input(
            result.result.model_copy(update={"max_follow_up_iterations": 1}),
            provider_ids=("comment-1",),
            iteration=2,
            product_run_id="new-product",
            session_id="new-session",
            task_id="new-task",
        )


def test_unapproved_reviewer_cannot_settle_application_review_policy(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    transport.check_bundles = [
        GitHubCheckBundle(
            head_commit=request.repository.head_commit,
            checks=(
                GitHubCheckObservation(
                    provider_id="check-1",
                    name="test",
                    head_commit=request.repository.head_commit,
                    status="completed",
                    conclusion="success",
                ),
            ),
        )
    ]
    transport.review_bundle = GitHubReviewBundle(
        feedback=(
            GitHubFeedbackObservation(
                provider_id="review-outsider",
                kind="review",
                head_commit=request.repository.head_commit,
                author_login="outsider",
                author_type="User",
                state="approved",
                created_at="2026-08-30T12:00:30Z",
                body="Approved without application authority",
            ),
        )
    )
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(connector.run(request, object(), object()))

    assert result.result.checks_state is GitHubCheckState.PASSED
    assert result.result.review_state is GitHubReviewState.COMMENTED
    assert result.result.state is GitHubDeliveryState.CHECKS_PENDING


def test_superseded_head_and_closed_pr_never_overclaim(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)

    async def newer_head(config, ref, limits):
        del config, limits
        return (
            request.repository.expected_base_commit
            if ref == request.repository.base_ref
            else "3" * 40
        )

    transport.observe_ref = newer_head
    superseded = asyncio.run(connector.run(request, object(), object()))
    assert superseded.result.state == GitHubDeliveryState.SUPERSEDED
    assert transport.create_calls == 0

    closed_request = request.model_copy(
        update={"connector_run_id": "closed-run", "idempotency_key": "closed-run"}
    )
    closed_transport = FakeTransport(closed_request)
    closed_transport.pull_request = _pr(closed_request, state="closed")
    closed_connector, _ = _connector(tmp_path / "closed", closed_request, closed_transport)
    closed = asyncio.run(closed_connector.run(closed_request, object(), object()))
    assert closed.result.state == GitHubDeliveryState.CLOSED
    assert closed.result.pull_request.merged is False


def test_update_marks_one_exact_bound_draft_ready_with_explicit_authority(tmp_path):
    request = _request(mode="update", number=7, draft=False)
    transport = FakeTransport(request)
    transport.pull_request = _pr(request).model_copy(update={"draft": True})
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="approval-ready"),
        )
    )

    assert result.result.pull_request is not None
    assert result.result.pull_request.number == 7
    assert result.result.pull_request.draft is False
    assert transport.create_calls == 0
    assert transport.update_calls == 0
    assert transport.ready_calls == 1
    assert result.result.operations[-1].operation is GitHubOperation.MARK_READY


def test_conflicting_approval_is_denied_without_provider_mutation(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="approval-conflict").model_copy(
        update={"policy_fingerprint": _digest("different-policy")}
    )

    result = asyncio.run(connector.run(request, object(), object(), approval=approval))

    assert result.result.state is GitHubDeliveryState.DENIED
    assert transport.create_calls == 0


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_public_waiter_stop_retains_dispatched_owner_and_blocks_more_effects(tmp_path, stop):
    request = _request(reviewers=("reviewer",))
    request = request.model_copy(
        update={"limits": request.limits.model_copy(update={"timeout_seconds": 1})}
    )
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)
    approval = approve_github_delivery(request, approval_id="owned-approval")

    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed_create(config, current):
            del config
            transport.create_calls += 1
            entered.set()
            await release.wait()
            transport.pull_request = _pr(current)
            return transport.pull_request, "delayed-create"

        transport.create_pull_request = delayed_create
        caller = asyncio.create_task(connector.run(request, object(), object(), approval=approval))
        await asyncio.wait_for(entered.wait(), timeout=5)
        if stop == "cancel":
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled()
            assert caller.cancelling() == 1
        else:
            with pytest.raises(GitHubProviderError, match="provider_timeout"):
                await caller
            assert not caller.cancelled()
        owner = connector._owners[request.connector_run_id][1]
        assert not owner.done()
        with pytest.raises(GitHubDeliveryAdmissionError, match="unsettled"):
            await connector.run(request, object(), object(), approval=approval)
        release.set()
        await asyncio.wait_for(asyncio.shield(owner), timeout=5)
        assert transport.create_calls == 1
        assert transport.reviewer_calls == 0
        latest = await connector.repository.latest(request)
        assert latest.result.state is GitHubDeliveryState.CANCELLED
        assert any(item.status == "succeeded" for item in latest.result.operations)

    asyncio.run(exercise())


def test_cancellation_is_durably_recorded_before_propagation(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)

    async def cancel(config, ref, limits):
        del config, ref, limits
        raise asyncio.CancelledError

    transport.observe_ref = cancel
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(connector.run(request, object(), object()))

    latest = asyncio.run(connector.repository.latest(request))
    assert latest is not None
    assert latest.result.state is GitHubDeliveryState.CANCELLED


def test_cancellation_during_create_is_ambiguous_and_reconciled_before_retry(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport)
    original_create = transport.create_pull_request

    async def create_then_cancel(config, current):
        del config
        transport.create_calls += 1
        transport.pull_request = _pr(current)
        raise asyncio.CancelledError

    transport.create_pull_request = create_then_cancel
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            connector.run(
                request,
                object(),
                object(),
                approval=approve_github_delivery(request, approval_id="approval-cancelled"),
            )
        )

    ambiguous = asyncio.run(connector.repository.latest(request))
    assert ambiguous is not None
    assert ambiguous.result.state is GitHubDeliveryState.AMBIGUOUS
    assert ambiguous.result.reason_code == "provider_mutation_cancelled_ambiguous"
    assert ambiguous.result.operations[-1].operation is GitHubOperation.CREATE_PULL_REQUEST
    assert ambiguous.result.operations[-1].status == "ambiguous"

    transport.create_pull_request = original_create
    recovered = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="approval-cancelled"),
        )
    )
    assert recovered.result.pull_request is not None
    assert recovered.result.state is not GitHubDeliveryState.AMBIGUOUS
    assert transport.create_calls == 1


def test_provider_pr_and_request_evidence_are_redacted_before_persistence(tmp_path):
    token = "github-secret-token"
    request = _request()
    transport = FakeTransport(request)

    async def create(config, current):
        del config
        transport.create_calls += 1
        transport.pull_request = _pr(current).model_copy(
            update={"body": f"provider echoed {token}"}
        )
        return transport.pull_request, f"request-{token}"

    transport.create_pull_request = create
    connector, _ = _connector(tmp_path, request, transport, token=token)
    result = asyncio.run(
        connector.run(
            request,
            object(),
            object(),
            approval=approve_github_delivery(request, approval_id="approval-redaction"),
        )
    )

    serialized = result.result.model_dump_json()
    assert token not in serialized
    assert REDACTED_SECRET in serialized


def test_request_containing_configured_credential_is_rejected_before_persistence(tmp_path):
    token = "host-side-github-secret"
    template = _request()
    request = template.model_copy(
        update={
            "metadata": template.metadata.model_copy(
                update={"body": f"accidental credential: {token}"}
            )
        }
    )
    transport = FakeTransport(request)
    connector, _ = _connector(tmp_path, request, transport, token=token)

    with pytest.raises(GitHubDeliveryAdmissionError, match="authority"):
        asyncio.run(connector.run(request, object(), object()))

    persisted_files = [path for path in (tmp_path / "artifacts").rglob("*") if path.is_file()]
    assert all(token.encode() not in path.read_bytes() for path in persisted_files)


def test_truncated_provider_observation_settles_partial_explicitly(tmp_path):
    request = _request()
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    transport.review_bundle = GitHubReviewBundle(feedback=(), truncated=True)
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(connector.run(request, object(), object()))

    assert result.result.state is GitHubDeliveryState.PARTIAL
    assert result.result.feedback_truncated is True


def test_repository_rejects_wrong_session_artifact_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    request = _request()
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="github-artifacts")
    repository = GitHubDeliveryRepository(store)
    asyncio.run(repository.ensure_request(request))
    request_artifact_id = _artifact_id("github-request", request.connector_run_id)
    original_read = store.read_bytes

    async def wrong_session_read(artifact_id, *, max_bytes=None):
        result = await original_read(artifact_id, max_bytes=max_bytes)
        if artifact_id != request_artifact_id:
            return result
        return ArtifactReadResult(
            metadata=result.metadata.model_copy(update={"session_id": "different-session"}),
            content=result.content,
            total_bytes=result.total_bytes,
            truncated=result.truncated,
            source_bytes_read=result.source_bytes_read,
            redaction_truncated=result.redaction_truncated,
        )

    monkeypatch.setattr(store, "read_bytes", wrong_session_read)
    with pytest.raises(
        GitHubDeliveryReconstructionRequiredError,
        match="artifact authority",
    ):
        asyncio.run(repository.ensure_request(request))


def test_repository_detects_lifecycle_reconstruction_gap(tmp_path):
    request = _request()
    store = LocalArtifactStore(tmp_path / "artifacts", store_id="github-artifacts")
    repository = GitHubDeliveryRepository(store)
    receipt = GitHubLifecycleReceipt(
        connector_run_id=request.connector_run_id,
        request_fingerprint=request.fingerprint,
        ordinal=2,
        state=GitHubDeliveryState.CHECKS_PENDING,
        result_sha256=_digest("result"),
        poll_count=1,
    )
    asyncio.run(
        repository._ensure(
            receipt,
            _artifact_id("github-receipt", request.connector_run_id, "2"),
            "github-lifecycle-2.json",
            request.session_id,
            "github_lifecycle",
        )
    )

    with pytest.raises(
        GitHubDeliveryReconstructionRequiredError,
        match="reconstruction gap",
    ):
        asyncio.run(repository.receipts(request))


def test_identical_publication_does_not_consume_an_extra_lifecycle_slot(tmp_path):
    request = _request()
    repository = GitHubDeliveryRepository(
        LocalArtifactStore(tmp_path / "artifacts", store_id="github-artifacts")
    )
    connector, _ = _connector(tmp_path / "connector", request, FakeTransport(request))
    result = connector._result(
        request,
        GitHubDeliveryState.APPROVAL_REQUIRED,
        reason="durable_provider_approval_required",
    )

    first = asyncio.run(repository.publish(request, result))
    second = asyncio.run(repository.publish(request, result))

    assert second == first
    assert len(asyncio.run(repository.receipts(request))) == 1


def test_review_authority_is_case_insensitive_without_double_counting(tmp_path):
    template = _request()
    with pytest.raises(ValidationError, match="case-insensitive"):
        GitHubReviewPolicy(
            approval_required=True,
            required_approvers=("Reviewer", "reviewer"),
            minimum_approvals=2,
        )

    request = template.model_copy(
        update={
            "reviews": GitHubReviewPolicy(
                approval_required=True,
                required_approvers=("Reviewer",),
            )
        }
    )
    transport = FakeTransport(request)
    transport.pull_request = _pr(request)
    transport.review_bundle = GitHubReviewBundle(
        feedback=(
            GitHubFeedbackObservation(
                provider_id="review:1",
                kind="review",
                head_commit=request.repository.head_commit,
                author_login="reviewer",
                author_type="User",
                state="approved",
                created_at="2026-08-30T12:00:30Z",
                body="Approved",
            ),
        )
    )
    connector, _ = _connector(tmp_path, request, transport)

    result = asyncio.run(connector.run(request, object(), object()))

    assert result.result.review_state is GitHubReviewState.APPROVED


def test_closed_request_schema_forbids_merge_and_unapproved_metadata_operations():
    with pytest.raises(ValidationError, match="explicit approver"):
        GitHubReviewPolicy(approval_required=True)

    request = _request()
    payload = request.model_dump(mode="python")
    payload["merge"] = True
    with pytest.raises(ValidationError):
        GitHubPullRequestDeliveryRequest.model_validate(payload)

    payload = request.model_dump(mode="python")
    payload["metadata"] = request.metadata.model_copy(update={"reviewers": ("reviewer",)})
    with pytest.raises(ValidationError, match="request_reviewers"):
        GitHubPullRequestDeliveryRequest.model_validate(payload)

    update = _request(mode="update", number=7)
    payload = update.model_dump(mode="python")
    payload["metadata"] = update.metadata.model_copy(update={"draft": False})
    with pytest.raises(ValidationError, match="mark_ready"):
        GitHubPullRequestDeliveryRequest.model_validate(payload)


def test_rest_transport_uses_vault_token_only_in_fixed_authorization_header(tmp_path):
    token = "rest-secret-token"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {token}"
        assert request.url.host == "api.github.example"
        if request.url.path.endswith("/statuses"):
            return httpx.Response(200, json=[])
        return httpx.Response(
            200, json={"object": {"sha": "1" * 40}}, headers={"x-github-request-id": "request-1"}
        )

    request = _request()
    fake = FakeTransport(request)
    _, config = _connector(tmp_path, request, fake, token=token)
    transport = GitHubRestTransport(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    observed = asyncio.run(
        transport.observe_ref(config, request.repository.base_ref, request.limits)
    )
    assert observed == "1" * 40


def test_rest_transport_never_forwards_credentials_across_redirects(tmp_path):
    token = "redirect-secret-token"
    seen = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        seen.append((http_request.url.host, http_request.headers.get("authorization")))
        if http_request.url.host == "api.github.example":
            return httpx.Response(
                307,
                headers={"location": "https://attacker.example/capture"},
            )
        pytest.fail("The GitHub credential followed a provider redirect")

    request = _request()
    fake = FakeTransport(request)
    _, config = _connector(tmp_path, request, fake, token=token)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )
    transport = GitHubRestTransport(client)

    with pytest.raises(GitHubProviderError, match="provider_redirect_forbidden"):
        asyncio.run(transport.observe_ref(config, request.repository.base_ref, request.limits))
    assert seen == [("api.github.example", f"Bearer {token}")]


@pytest.mark.parametrize("termination", ["cancel", "deadline"])
@pytest.mark.parametrize("client_failure", [False, True])
def test_rest_cleanup_preserves_active_signal(
    tmp_path, monkeypatch, caplog, capsys, termination, client_failure
):
    import cayu.github_delivery as module

    secret = "github-cleanup-signal-secret"
    request = _request()
    _, config = _connector(tmp_path, request, FakeTransport(request), token=secret)

    async def scenario():
        entered = asyncio.Event()
        closed = []

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b""

            async def aclose(self):
                closed.append("response")
                raise RuntimeError(secret)

        class Client(httpx.AsyncClient):
            async def aclose(self):
                closed.append("client")
                await super().aclose()
                if client_failure:
                    raise RuntimeError(secret)

        async def handler(_request):
            return httpx.Response(200, stream=Stream())

        client = Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kwargs: client)
        transport = GitHubRestTransport()
        limits = request.limits.model_copy(
            update={"timeout_seconds": 0.05 if termination == "deadline" else 10}
        )
        task = asyncio.create_task(transport.observe_ref(config, "refs/heads/main", limits))
        await entered.wait()
        if termination == "cancel":
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
        else:
            with pytest.raises(GitHubProviderError, match="provider_timeout") as caught:
                await task
            assert not task.cancelled()
            assert task.cancelling() == 0
        assert closed == ["response", "client"]
        error = caught.value
        cleanup = error.__cause__
        assert isinstance(cleanup, BaseExceptionGroup)

        def leaves(current):
            if isinstance(current, BaseExceptionGroup):
                return [leaf for child in current.exceptions for leaf in leaves(child)]
            return [current]

        evidence = leaves(cleanup)
        assert len(evidence) == 1 + client_failure
        assert all(
            type(item) is GitHubProviderError and item.code == "provider_extension_failure"
            for item in evidence
        )
        return "".join(traceback.format_exception(error))

    with warnings.catch_warnings(record=True) as captured:
        diagnostic = asyncio.run(scenario())
    output = capsys.readouterr()
    assert secret not in diagnostic + caplog.text + output.out + output.err + str(captured)


def test_rest_transport_retains_latest_commit_status_without_false_truncation(tmp_path):
    request = _request()

    async def handler(http_request: httpx.Request) -> httpx.Response:
        if http_request.url.path.endswith("/check-runs"):
            return httpx.Response(200, json={"total_count": 0, "check_runs": []})
        assert http_request.url.path.endswith("/statuses")
        return httpx.Response(
            200,
            json=[
                {
                    "id": 9,
                    "context": "test",
                    "state": "success",
                    "created_at": "2026-08-30T12:00:00Z",
                    "updated_at": "2026-08-30T12:01:00Z",
                    "target_url": "https://github.example/checks/9",
                },
                {
                    "id": 8,
                    "context": "test",
                    "state": "pending",
                    "created_at": "2026-08-30T11:59:00Z",
                    "updated_at": "2026-08-30T11:59:00Z",
                    "target_url": "https://github.example/checks/8",
                },
            ],
        )

    fake = FakeTransport(request)
    _, config = _connector(tmp_path, request, fake)
    transport = GitHubRestTransport(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    observed = asyncio.run(
        transport.observe_checks(
            config,
            request.repository.head_commit,
            request.limits.model_copy(update={"max_checks": 2}),
        )
    )

    assert observed.truncated is False
    assert len(observed.checks) == 1
    assert observed.checks[0].provider_id == "commit-status:9"
    assert observed.checks[0].conclusion == "success"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("failure", GitHubCheckState.FAILED),
        ("pending", GitHubCheckState.PENDING),
        ("success", GitHubCheckState.PASSED),
    ],
)
def test_rest_connector_requires_both_same_named_signal_families(tmp_path, status, expected):
    request = _request(mode="update", number=7)

    async def handler(http_request):
        assert http_request.method == "GET"
        path = http_request.url.path
        if "/git/ref/" in path:
            return httpx.Response(
                200,
                json={
                    "object": {
                        "sha": "1" * 40 if path.endswith("/main") else "2" * 40,
                    }
                },
            )
        if path.endswith("/pulls/7"):
            return httpx.Response(
                200,
                json={
                    "number": 7,
                    "node_id": "node-7",
                    "html_url": "https://github.example/pr/7",
                    "state": "open",
                    "draft": True,
                    "base": {"ref": "main", "sha": "1" * 40, "repo": {"full_name": "cayu/runtime"}},
                    "head": {
                        "ref": "cayu/change",
                        "sha": "2" * 40,
                        "repo": {"full_name": "cayu/runtime"},
                    },
                    "title": request.metadata.title,
                    "body": request.metadata.body,
                    "labels": [],
                },
            )
        if path.endswith("/check-runs"):
            return httpx.Response(
                200,
                json={
                    "total_count": 1,
                    "check_runs": [
                        {
                            "id": 1,
                            "name": "test",
                            "head_sha": "2" * 40,
                            "status": "completed",
                            "conclusion": "success",
                        }
                    ],
                },
            )
        if path.endswith("/statuses"):
            return httpx.Response(
                200,
                json=[
                    {"id": 3, "context": "test", "state": status},
                    {"id": 2, "context": "test", "state": "error"},
                ],
            )
        assert path.endswith(("/reviews", "/comments"))
        return httpx.Response(200, json=[])

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = GitHubRestTransport(client)
            connector, _ = _connector(tmp_path, request, transport)
            outcome = await connector.run(request, object(), object())
            assert outcome.result.checks_state is expected
            assert {item.provider_id for item in outcome.result.checks} == {
                "check-run:1",
                "commit-status:3",
            }
            assert await connector.repository.latest(request) == outcome

    asyncio.run(scenario())


def test_rest_transport_marks_ready_through_one_fixed_graphql_mutation(tmp_path):
    token = "rest-secret-token"
    request = _request(mode="update", number=7, draft=False)
    seen = []

    async def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.headers["authorization"] == f"Bearer {token}"
        seen.append((http_request.method, http_request.url.path))
        if http_request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "number": 7,
                    "node_id": "node-7",
                    "html_url": "https://github.example/pr/7",
                    "state": "open",
                    "draft": True,
                    "base": {"ref": "main", "sha": "1" * 40, "repo": {"full_name": "cayu/runtime"}},
                    "head": {
                        "ref": "cayu/change",
                        "sha": "2" * 40,
                        "repo": {"full_name": "cayu/runtime"},
                    },
                    "title": request.metadata.title,
                    "body": request.metadata.body,
                    "labels": [],
                },
            )
        assert http_request.url.path == "/graphql"
        payload = json.loads(http_request.content)
        assert payload["variables"] == {"pullRequestId": "node-7"}
        assert "markPullRequestReadyForReview" in payload["query"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "markPullRequestReadyForReview": {
                        "pullRequest": {"id": "node-7", "isDraft": False}
                    }
                }
            },
            headers={"x-github-request-id": "request-ready"},
        )

    fake = FakeTransport(request)
    _, config = _connector(tmp_path, request, fake, token=token)
    transport = GitHubRestTransport(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    request_id = asyncio.run(transport.mark_ready(config, 7, request.limits))

    assert request_id == "request-ready"
    assert seen == [("GET", "/repos/cayu/runtime/pulls/7"), ("POST", "/graphql")]
