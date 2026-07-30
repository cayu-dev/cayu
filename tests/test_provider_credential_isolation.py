from __future__ import annotations

import asyncio
import json
import threading
import traceback
from pathlib import Path
from typing import Any

import pytest

from cayu import CayuApp
from cayu.core.messages import Message
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers._http import (
    credential_safe_error_event,
    credential_safe_provider_exception,
    sanitize_provider_cancellation,
)
from cayu.providers.anthropic import AnthropicProvider
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProviderError,
    ModelRequest,
    ModelStreamEventType,
)
from cayu.providers.chat_completions import ChatCompletionsProvider
from cayu.providers.openai import OpenAIProvider
from cayu.providers.openai_subscription import (
    OpenAISubscriptionAuth,
    OpenAISubscriptionAuthStore,
    OpenAISubscriptionCredentials,
    OpenAISubscriptionProvider,
)
from cayu.providers.vertex import VertexProvider
from cayu.proxies import PassthroughProxy
from cayu.runners import ExecCommand, ExecResult, LocalRunner, Runner
from cayu.testing import (
    ProviderCredentialIsolationViolation,
    verify_provider_credential_isolation,
)
from cayu.vaults import SecretRef, StaticVault
from tests.provider_traceback_assertions import (
    assert_cayu_traceback_does_not_retain,
    is_cayu_source_filename,
)


class _RecordingProbeRunner(Runner):
    isolation = "test"

    def __init__(self, *, leak: tuple[str, str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.leak = leak

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": dict(env or {}),
                "timeout_s": timeout_s,
                "stdin": stdin,
                "output_limit_bytes": output_limit_bytes,
            }
        )
        observed = {
            "environment": dict(env or {}),
            "auth_paths": {},
            "auth_scan_complete": True,
            "provider_canary_matches": [],
            "detector_control_match": True,
        }
        stdout = json.dumps(observed, sort_keys=True)
        stderr = ""
        artifacts: list[dict[str, Any]] = []
        if self.leak is not None:
            projection, value = self.leak
            if projection == "stdout":
                stdout = f"prefix:{value}:suffix"
            elif projection == "stderr":
                stderr = f"prefix:{value}:suffix"
            elif projection == "environment":
                observed["environment"]["PROVIDER_LEAK"] = value
                stdout = json.dumps(observed, sort_keys=True)
            elif projection == "auth_paths":
                observed["auth_paths"]["/root/.cayu/auth.json"] = value
                stdout = json.dumps(observed, sort_keys=True)
            else:
                artifacts = [{"diagnostic": f"prefix:{value}:suffix"}]
        return ExecResult(stdout=stdout, stderr=stderr, artifacts=artifacts)


class _ExecutingIsolatedProbeRunner(Runner):
    isolation = "test"

    def __init__(self, workspace: Path) -> None:
        self._local = LocalRunner(workspace)

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        return await self._local.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )


class _RedactingLocalProbeRunner(Runner):
    isolation = "test"

    def __init__(self, workspace: Path, *, env_name: str, secret_value: str) -> None:
        self._local = LocalRunner(
            workspace,
            secret_env={env_name: SecretRef(name="probe_secret")},
            secret_resolver=StaticVault({"probe_secret": secret_value}),
        )

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        return await self._local.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )


def _canaries() -> dict[str, str]:
    return {
        "openai_api_key": "cayu-provider-openai-canary-0123456789",
        "oauth_access_token": "cayu-provider-access-canary-0123456789",
        "oauth_refresh_token": "cayu-provider-refresh-canary-0123456789",
        "account_id": "cayu-provider-account-canary-0123456789",
        "authorization_header": "Bearer cayu-provider-header-canary-0123456789",
    }


class _CredentialEchoingOpenAITransport:
    def __init__(self, message: str | Exception) -> None:
        self.message = message

    async def stream_response_events(self, **kwargs: Any):
        if isinstance(self.message, Exception):
            raise self.message
        raise RuntimeError(self.message)
        yield {}


class _CredentialEchoingAnthropicTransport:
    def __init__(self, message: str | Exception) -> None:
        self.message = message

    async def stream_message_events(self, **kwargs: Any):
        if isinstance(self.message, Exception):
            raise self.message
        raise RuntimeError(self.message)
        yield {}


class _CredentialEchoingChatTransport:
    def __init__(self, message: str | Exception) -> None:
        self.message = message

    async def stream_chat_completions(self, **kwargs: Any):
        if isinstance(self.message, Exception):
            raise self.message
        raise RuntimeError(self.message)
        yield {}


class _CredentialThrowingOpenAITransport:
    def __init__(self, failure_kind: str, canary: str) -> None:
        self.failure_kind = failure_kind
        self.canary = canary

    async def stream_response_events(self, **kwargs: Any):
        if self.failure_kind == "cancel":
            raise asyncio.CancelledError(f"cancelled with {kwargs['headers']} {self.canary}")
        error = ModelContextOverflowError(
            f"overflow near {kwargs['headers']} {self.canary}",
            provider="openai",
            status_code=400,
            error_code=self.canary,
            request_id=self.canary,
            response_body=self.canary,
        )
        error.headers = kwargs["headers"]
        error.add_note(f"transport retained {self.canary}")
        raise error
        yield {}


class _CredentialThrowingAnthropicTransport:
    def __init__(self, failure_kind: str, canary: str) -> None:
        self.failure_kind = failure_kind
        self.canary = canary

    async def stream_message_events(self, **kwargs: Any):
        if self.failure_kind == "cancel":
            raise asyncio.CancelledError(f"cancelled with {kwargs['headers']} {self.canary}")
        error = ModelContextOverflowError(
            f"overflow near {kwargs['headers']} {self.canary}",
            provider="anthropic",
            status_code=400,
            error_code=self.canary,
            request_id=self.canary,
            response_body=self.canary,
        )
        error.headers = kwargs["headers"]
        error.add_note(f"transport retained {self.canary}")
        raise error
        yield {}


class _CredentialThrowingChatTransport:
    def __init__(self, failure_kind: str, canary: str) -> None:
        self.failure_kind = failure_kind
        self.canary = canary

    async def stream_chat_completions(self, **kwargs: Any):
        if self.failure_kind == "cancel":
            raise asyncio.CancelledError(f"cancelled with {kwargs['headers']} {self.canary}")
        error = ModelContextOverflowError(
            f"overflow near {kwargs['headers']} {self.canary}",
            provider="chat_completions",
            status_code=400,
            error_code=self.canary,
            request_id=self.canary,
            response_body=self.canary,
        )
        error.headers = kwargs["headers"]
        error.add_note(f"transport retained {self.canary}")
        raise error
        yield {}


class _StaticVertexCredentials:
    def __init__(self, token: str) -> None:
        self.token = token
        self.valid = True


class _StaticSubscriptionAuth:
    async def credentials(self) -> OpenAISubscriptionCredentials:
        return OpenAISubscriptionCredentials(
            access_token="subscription-access-control-0123456789",
            refresh_token="subscription-refresh-control-0123456789",
            expires_at=2_000_000_000,
            account_id="subscription-account-control-0123456789",
        )


class _CanarySubscriptionAuth:
    def __init__(self, canary: str) -> None:
        self.canary = canary

    async def credentials(self) -> OpenAISubscriptionCredentials:
        return OpenAISubscriptionCredentials(
            access_token=self.canary,
            refresh_token=f"refresh-{self.canary}",
            expires_at=2_000_000_000,
            account_id=f"account-{self.canary}",
        )


@pytest.mark.parametrize(
    ("provider_factory", "model"),
    [
        (
            lambda credential, error: OpenAIProvider(
                api_key=credential,
                transport=_CredentialEchoingOpenAITransport(error),
            ),
            "gpt-test",
        ),
        (
            lambda credential, error: AnthropicProvider(
                api_key=credential,
                transport=_CredentialEchoingAnthropicTransport(error),
            ),
            "claude-test",
        ),
        (
            lambda credential, error: ChatCompletionsProvider(
                api_key=credential,
                name="gemini",
                transport=_CredentialEchoingChatTransport(error),
            ),
            "gemini-test",
        ),
    ],
)
def test_api_key_provider_errors_never_project_the_provider_credential(
    provider_factory,
    model: str,
    provider_credential_canaries,
) -> None:
    credential = provider_credential_canaries.values[
        {
            "gpt-test": "openai_api_key",
            "claude-test": "anthropic_api_key",
            "gemini-test": "gemini_api_key",
        }[model]
    ]
    header = f"Authorization: Bearer {credential}"
    provider = provider_factory(credential, f"transport rejected {header}")
    request = ModelRequest(model=model, messages=[Message.text("user", "hello")])

    events = asyncio.run(_collect_provider_events(provider, request))

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    retained = repr(events[0]) + events[0].model_dump_json()
    assert credential not in retained
    assert header not in retained
    assert (
        events[0].payload["error"]
        == {
            "gpt-test": "RuntimeError: OpenAI provider failed",
            "claude-test": "RuntimeError: Anthropic provider failed",
            "gemini-test": "RuntimeError: Chat Completions provider failed",
        }[model]
    )


@pytest.mark.parametrize(
    ("provider_factory", "model"),
    [
        (
            lambda header, error: OpenAIProvider(
                api_key="openai-primary-control-0123456789",
                extra_headers={"x-provider-authorization": header},
                transport=_CredentialEchoingOpenAITransport(error),
            ),
            "gpt-test",
        ),
        (
            lambda header, error: AnthropicProvider(
                api_key="anthropic-primary-control-0123456789",
                extra_headers={"x-provider-authorization": header},
                transport=_CredentialEchoingAnthropicTransport(error),
            ),
            "claude-test",
        ),
        (
            lambda header, error: ChatCompletionsProvider(
                api_key="chat-primary-control-0123456789",
                name="gemini",
                extra_headers={"x-provider-authorization": header},
                transport=_CredentialEchoingChatTransport(error),
            ),
            "gemini-test",
        ),
        (
            lambda header, error: OpenAISubscriptionProvider(
                auth=_StaticSubscriptionAuth(),
                extra_headers={"x-provider-authorization": header},
                transport=_CredentialEchoingOpenAITransport(error),
            ),
            "subscription-test",
        ),
    ],
)
def test_provider_errors_never_project_distinct_extra_header_credentials(
    provider_factory,
    model: str,
) -> None:
    canary = f"provider-extra-header-{model}-canary-0123456789"
    error = ModelProviderError(
        f"request failed near {canary}",
        provider=model,
        status_code=401,
        error_type=canary,
        error_code=canary,
        request_id=canary,
        retryable=False,
        response_body=canary,
    )
    error.headers = {"x-provider-authorization": canary}
    error.add_note(f"transport retained {canary}")
    provider = provider_factory(canary, error)

    events = asyncio.run(
        _collect_provider_events(
            provider,
            ModelRequest(model=model, messages=[Message.text("user", "hello")]),
        )
    )

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    retained = repr(events[0]) + events[0].model_dump_json()
    assert canary not in retained


@pytest.mark.parametrize(
    ("provider_factory", "model"),
    [
        (
            lambda credential, kind: OpenAIProvider(
                api_key=credential,
                transport=_CredentialThrowingOpenAITransport(kind, credential),
            ),
            "gpt-test",
        ),
        (
            lambda credential, kind: AnthropicProvider(
                api_key=credential,
                transport=_CredentialThrowingAnthropicTransport(kind, credential),
            ),
            "claude-test",
        ),
        (
            lambda credential, kind: ChatCompletionsProvider(
                api_key=credential,
                name="gemini",
                transport=_CredentialThrowingChatTransport(kind, credential),
            ),
            "gemini-test",
        ),
        (
            lambda credential, kind: VertexProvider(
                project_id="credential-isolation-project",
                region="us-east5",
                credentials=_StaticVertexCredentials(credential),
                transport=_CredentialThrowingAnthropicTransport(kind, credential),
            ),
            "vertex-test",
        ),
        (
            lambda credential, kind: OpenAISubscriptionProvider(
                auth=_CanarySubscriptionAuth(credential),
                transport=_CredentialThrowingOpenAITransport(kind, credential),
            ),
            "subscription-test",
        ),
    ],
)
@pytest.mark.parametrize("failure_kind", ["cancel", "overflow"])
def test_api_key_provider_failure_tracebacks_drop_transport_credential_locals(
    provider_factory,
    model: str,
    failure_kind: str,
) -> None:
    canary = f"provider-{model}-traceback-canary-0123456789"
    provider = provider_factory(canary, failure_kind)
    request = ModelRequest(model=model, messages=[Message.text("user", "hello")])
    expected = asyncio.CancelledError if failure_kind == "cancel" else ModelContextOverflowError

    with pytest.raises(expected) as exc_info:
        asyncio.run(_collect_provider_events(provider, request))

    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert_cayu_traceback_does_not_retain(exc_info.value, provider)
    retained = repr([(frame.name, frame.locals) for frame in cayu_frames])
    assert canary not in retained
    assert all(
        frame.name
        not in {
            "stream_response_events",
            "stream_message_events",
            "stream_chat_completions",
        }
        for frame in captured.stack
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_unresolved_provider_auth_failure_uses_a_content_free_error_projection() -> None:
    canary = "cayu-unresolved-provider-canary-0123456789"

    event = credential_safe_error_event(
        RuntimeError(f"credential resolver failed near {canary}"),
        provider_label="Anthropic",
        provider_name="anthropic",
        credential_values=(),
    )

    retained = repr(event) + event.model_dump_json()
    assert canary not in retained
    assert event.payload == {
        "error": "RuntimeError: Anthropic provider failed",
        "error_type": "RuntimeError",
    }


def test_shared_provider_cancellation_drops_notes_and_custom_metadata() -> None:
    canary = "provider-cancellation-canary-0123456789"
    error = asyncio.CancelledError(f"cancelled with {canary}")
    error.artifacts = [{"credential": canary}]
    error.headers = {"Authorization": f"Bearer {canary}"}
    error.add_note(f"transport retained {canary}")

    safe = sanitize_provider_cancellation(
        error,
        provider_label="OpenAI",
        credential_values=(canary,),
    )

    rendered = repr(safe) + repr(vars(safe))
    assert canary not in rendered
    assert safe.artifacts == []
    assert not hasattr(safe, "headers")
    assert not hasattr(safe, "__notes__")
    assert safe.__cause__ is None
    assert safe.__context__ is None


def test_shared_provider_cancellation_bypasses_hostile_artifact_descriptor() -> None:
    canary = "provider-cancellation-descriptor-canary-0123456789"

    class HostileCancellation(asyncio.CancelledError):
        @property
        def artifacts(self):
            raise RuntimeError(canary)

    safe = sanitize_provider_cancellation(
        HostileCancellation(canary),
        provider_label="OpenAI",
        credential_values=(canary,),
    )

    rendered = repr(safe) + repr(vars(safe))
    assert canary not in rendered
    assert not hasattr(safe, "artifacts")


def test_shared_typed_provider_error_drops_notes_and_custom_metadata() -> None:
    canary = "provider-overflow-canary-0123456789"
    error = ModelContextOverflowError(
        f"overflow near {canary}",
        provider="openai",
        status_code=400,
        error_type=canary,
        error_code=canary,
        request_id=canary,
        response_body=canary,
    )
    error.headers = {"Authorization": f"Bearer {canary}"}
    error.add_note(f"transport retained {canary}")

    safe = credential_safe_provider_exception(
        error,
        provider_label="OpenAI",
        provider_name="openai",
        credential_values=(canary,),
    )

    rendered = str(safe) + repr(safe) + repr(vars(safe))
    assert canary not in rendered
    assert isinstance(safe, ModelContextOverflowError)
    assert safe.status_code == 400
    assert safe.response_body is None
    assert not hasattr(safe, "headers")
    assert not hasattr(safe, "__notes__")
    assert safe.__cause__ is None
    assert safe.__context__ is None


async def _collect_provider_events(provider: Any, request: ModelRequest) -> list[Any]:
    return [event async for event in provider.stream(request)]


def test_provider_credential_verifier_returns_content_free_positive_evidence() -> None:
    runner = _RecordingProbeRunner()

    evidence = asyncio.run(
        verify_provider_credential_isolation(
            runner,
            adapter="test_remote",
            scope="isolated_guest",
            provider_canaries=_canaries(),
            operational_env={"CAYU_PROBE_VISIBLE": "operational-control-0123456789"},
            workload_env={"CAYU_WORKLOAD_TOKEN": "workload-control-0123456789"},
        )
    )

    assert evidence.model_dump(mode="json") == {
        "schema_version": "cayu.provider_credential_isolation.v1",
        "status": "verified",
        "adapter": "test_remote",
        "scope": "isolated_guest",
        "canary_labels": [
            "account_id",
            "authorization_header",
            "oauth_access_token",
            "oauth_refresh_token",
            "openai_api_key",
        ],
        "auth_search_labels": [
            "current_working_directory",
            "guest_home",
            "workspace_root",
        ],
        "projections": ["artifacts", "auth_paths", "environment", "stderr", "stdout"],
        "positive_controls": ["CAYU_PROBE_VISIBLE", "CAYU_WORKLOAD_TOKEN"],
    }
    rendered = repr(evidence) + json.dumps(evidence.model_dump(mode="json"))
    assert all(value not in rendered for value in _canaries().values())
    assert runner.calls[0]["env"]["CAYU_PROBE_VISIBLE"] == ("operational-control-0123456789")
    assert runner.calls[0]["env"]["CAYU_WORKLOAD_TOKEN"] == ("workload-control-0123456789")
    assert all(value not in repr(runner.calls[0]) for value in _canaries().values())


def test_provider_credential_verifier_detects_runner_secret_before_result_redaction(
    tmp_path: Path,
) -> None:
    canary = _canaries()["openai_api_key"]

    with pytest.raises(ProviderCredentialIsolationViolation) as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                _RedactingLocalProbeRunner(
                    tmp_path,
                    env_name="ACCIDENTAL_PROVIDER_KEY",
                    secret_value=canary,
                ),
                adapter="redacting_isolated",
                scope="isolated_guest",
                provider_canaries={"openai_api_key": canary},
                operational_env={
                    "HOME": str(tmp_path),
                    "VISIBLE": "positive-control-0123456789",
                },
            )
        )

    assert exc_info.value.canary_label == "openai_api_key"
    assert exc_info.value.projection == "environment"
    assert canary not in str(exc_info.value)


def test_provider_credential_verifier_detects_embedded_secret_before_result_redaction(
    tmp_path: Path,
) -> None:
    canary = _canaries()["authorization_header"]
    injected_value = f"Bearer prefix-{canary}-suffix"

    with pytest.raises(ProviderCredentialIsolationViolation) as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                _RedactingLocalProbeRunner(
                    tmp_path,
                    env_name="ACCIDENTAL_AUTHORIZATION",
                    secret_value=injected_value,
                ),
                adapter="redacting_isolated",
                scope="isolated_guest",
                provider_canaries={"authorization_header": canary},
                operational_env={
                    "HOME": str(tmp_path),
                    "VISIBLE": "positive-control-0123456789",
                },
            )
        )

    assert exc_info.value.canary_label == "authorization_header"
    assert exc_info.value.projection == "environment"
    assert canary not in str(exc_info.value)


@pytest.mark.parametrize(
    "projection",
    ["environment", "auth_paths", "stdout", "stderr", "artifacts"],
)
def test_provider_credential_verifier_inspects_raw_results_without_echoing_canary(
    projection: str,
) -> None:
    canaries = _canaries()
    leaked = canaries["oauth_refresh_token"]
    runner = _RecordingProbeRunner(leak=(projection, leaked))

    with pytest.raises(ProviderCredentialIsolationViolation) as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                runner,
                adapter="test_remote",
                scope="isolated_guest",
                provider_canaries=canaries,
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )

    assert exc_info.value.canary_label == "oauth_refresh_token"
    assert exc_info.value.projection == projection
    assert leaked not in str(exc_info.value)
    assert leaked not in repr(exc_info.value)
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert leaked not in repr([(frame.name, frame.locals) for frame in cayu_frames])


def test_provider_credential_verifier_rejects_a_vacuous_probe() -> None:
    class EmptyRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            return ExecResult(stdout="{}")

    with pytest.raises(RuntimeError, match="malformed output"):
        asyncio.run(
            verify_provider_credential_isolation(
                EmptyRunner(),
                adapter="empty",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )


def test_provider_credential_verifier_requires_exact_structured_positive_controls() -> None:
    control = "positive-control-0123456789"

    class MisleadingRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            return ExecResult(
                stdout=json.dumps(
                    {
                        "environment": {},
                        "auth_paths": {},
                        "auth_scan_complete": True,
                        "provider_canary_matches": [],
                        "detector_control_match": True,
                        "untrusted_note": control,
                    }
                ),
                stderr=control,
                artifacts=[{"also_not_environment": control}],
            )

    with pytest.raises(RuntimeError, match="positive control"):
        asyncio.run(
            verify_provider_credential_isolation(
                MisleadingRunner(),
                adapter="misleading",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={"VISIBLE": control},
            )
        )


def test_provider_credential_verifier_never_passes_a_canary_as_a_control() -> None:
    canary = _canaries()["oauth_access_token"]
    runner = _RecordingProbeRunner()

    with pytest.raises(ValueError, match="must not contain provider credential canaries"):
        asyncio.run(
            verify_provider_credential_isolation(
                runner,
                adapter="overlap",
                scope="isolated_guest",
                provider_canaries={"oauth_access_token": canary},
                operational_env={"VISIBLE": f"prefix-{canary}-suffix"},
            )
        )

    assert runner.calls == []


@pytest.mark.parametrize("unsafe_field", ["label", "adapter"])
def test_provider_credential_verifier_never_returns_a_canary_as_evidence_identity(
    unsafe_field: str,
) -> None:
    canary = "credential-shaped-safe-name-0123456789"
    runner = _RecordingProbeRunner()
    label = canary if unsafe_field == "label" else "openai_api_key"
    adapter = canary if unsafe_field == "adapter" else "test_remote"

    with pytest.raises(ValueError, match="must not contain provider credential canaries"):
        asyncio.run(
            verify_provider_credential_isolation(
                runner,
                adapter=adapter,
                scope="isolated_guest",
                provider_canaries={label: canary},
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )

    assert runner.calls == []


def test_provider_credential_verifier_rejects_undeclared_provider_environment() -> None:
    class UnexpectedProviderEnvRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            environment = {**kwargs["env"], "CAYU_HOME": "/opt/unexpected-cayu-home"}
            return ExecResult(
                stdout=json.dumps(
                    {
                        "environment": environment,
                        "auth_paths": {},
                        "auth_scan_complete": True,
                        "provider_canary_matches": [],
                        "detector_control_match": True,
                    }
                )
            )

    with pytest.raises(RuntimeError, match="undeclared provider environment"):
        asyncio.run(
            verify_provider_credential_isolation(
                UnexpectedProviderEnvRunner(),
                adapter="unexpected_env",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )


def test_explicit_workload_provider_named_environment_remains_allowed() -> None:
    workload_value = "explicit-workload-openai-value-0123456789"

    evidence = asyncio.run(
        verify_provider_credential_isolation(
            _RecordingProbeRunner(),
            adapter="explicit_workload",
            scope="isolated_guest",
            provider_canaries=_canaries(),
            operational_env={"VISIBLE": "positive-control-0123456789"},
            workload_env={"OPENAI_API_KEY": workload_value},
        )
    )

    assert evidence.status == "verified"
    assert "OPENAI_API_KEY" in evidence.positive_controls


def test_provider_credential_verifier_does_not_echo_probe_failure_canary() -> None:
    canary = _canaries()["authorization_header"]

    class FailingRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            raise RuntimeError(f"transport failed with {canary}")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                FailingRunner(),
                adapter="failed",
                scope="isolated_guest",
                provider_canaries={"authorization_header": canary},
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )

    assert str(exc_info.value) == "provider credential isolation probe execution failed"
    assert canary not in str(exc_info.value)
    assert canary not in repr(exc_info.value)
    assert exc_info.value.__context__ is None
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])


def test_provider_credential_verifier_inspects_timeout_cleanup_artifacts_first() -> None:
    canary = _canaries()["oauth_access_token"]

    class TimedOutRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            return ExecResult(
                timed_out=True,
                exit_code=-9,
                artifacts=[{"cleanup_error": f"failed near {canary}"}],
            )

    with pytest.raises(ProviderCredentialIsolationViolation) as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                TimedOutRunner(),
                adapter="timed_out",
                scope="isolated_guest",
                provider_canaries={"oauth_access_token": canary},
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )

    assert exc_info.value.projection == "artifacts"
    assert canary not in str(exc_info.value)


def test_provider_credential_verifier_preserves_sanitized_cancellation() -> None:
    canary = _canaries()["oauth_refresh_token"]

    class CancelledRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            error = asyncio.CancelledError(f"cancelled with {canary}")
            error.artifacts = [{"cleanup": canary}]
            error.headers = {"Authorization": f"Bearer {canary}"}
            error.add_note(f"probe cancelled near {canary}")
            raise error

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                CancelledRunner(),
                adapter="cancelled",
                scope="isolated_guest",
                provider_canaries={"oauth_refresh_token": canary},
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )

    rendered = repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in rendered
    assert exc_info.value.artifacts == []
    assert not hasattr(exc_info.value, "headers")
    assert not hasattr(exc_info.value, "__notes__")
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])
    assert all(frame.name != "exec" for frame in captured.stack)


def test_local_default_proves_environment_minimization_without_claiming_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canaries = _canaries()
    monkeypatch.setenv("OPENAI_API_KEY", canaries["openai_api_key"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "cayu-provider-anthropic-canary-0123456789")
    monkeypatch.setenv("GEMINI_API_KEY", "cayu-provider-gemini-canary-0123456789")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "cayu-provider-aws-access-canary-0123456789")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cayu-provider-aws-secret-canary-0123456789")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "cayu-provider-aws-session-canary-0123456789")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(tmp_path / "provider-google-adc-canary-0123456789.json"),
    )
    monkeypatch.setenv("CAYU_HOME", str(tmp_path / "provider-auth-canary"))
    runner = LocalRunner(tmp_path)

    evidence = asyncio.run(
        verify_provider_credential_isolation(
            runner,
            adapter="local",
            scope="local_environment",
            provider_canaries={
                **canaries,
                "anthropic_api_key": "cayu-provider-anthropic-canary-0123456789",
                "gemini_api_key": "cayu-provider-gemini-canary-0123456789",
                "aws_access_key_id": "cayu-provider-aws-access-canary-0123456789",
                "aws_secret_access_key": "cayu-provider-aws-secret-canary-0123456789",
                "aws_session_token": "cayu-provider-aws-session-canary-0123456789",
                "google_application_credentials_path": str(
                    tmp_path / "provider-google-adc-canary-0123456789.json"
                ),
                "auth_store_path": str(tmp_path / "provider-auth-canary"),
            },
            operational_env={"VISIBLE": "positive-control-0123456789"},
        )
    )

    assert evidence.status == "environment_minimized"
    claim = runner.execution_capability_evidence().claim_for("real_credential_non_possession")
    assert claim is not None
    assert claim.state == "unsupported"


def test_local_environment_probe_never_reads_or_serializes_host_auth_files() -> None:
    runner = _RecordingProbeRunner()

    evidence = asyncio.run(
        verify_provider_credential_isolation(
            runner,
            adapter="local_recording",
            scope="local_environment",
            provider_canaries=_canaries(),
            operational_env={"VISIBLE": "positive-control-0123456789"},
        )
    )

    script = runner.calls[0]["command"].argv[-1]
    assert evidence.status == "environment_minimized"
    assert "auth.json" not in script
    assert ".cayu" not in script
    assert "read_bytes" not in script
    assert "open(" not in script


def test_isolated_probe_fails_on_guest_auth_file_presence_without_reading_content() -> None:
    class AuthPathRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            return ExecResult(
                stdout=json.dumps(
                    {
                        "environment": kwargs["env"],
                        "auth_paths": {"/root/.cayu/auth.json": "present"},
                        "auth_scan_complete": True,
                        "provider_canary_matches": [],
                        "detector_control_match": True,
                    }
                )
            )

    with pytest.raises(RuntimeError, match="unexpected guest auth path"):
        asyncio.run(
            verify_provider_credential_isolation(
                AuthPathRunner(),
                adapter="isolated_auth_path",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )


def test_isolated_probe_checks_the_actual_guest_working_directory_auth_path() -> None:
    runner = _RecordingProbeRunner()

    asyncio.run(
        verify_provider_credential_isolation(
            runner,
            adapter="isolated_cwd",
            scope="isolated_guest",
            provider_canaries=_canaries(),
            operational_env={"VISIBLE": "positive-control-0123456789"},
        )
    )

    script = runner.calls[0]["command"].argv[-1]
    assert "pathlib.Path.cwd()/'.cayu'/'auth.json'" in script
    assert "os.walk(root" in script
    assert "read_bytes" not in script
    assert "read_text" not in script
    assert "open(" not in script


def test_isolated_probe_executes_nested_mounted_workspace_auth_search(
    tmp_path: Path,
) -> None:
    guest = tmp_path / "guest"
    mounted_workspace = tmp_path / "mounted-workspace"
    auth_store = mounted_workspace / "project" / ".cayu" / "auth.json"
    guest.mkdir()
    auth_store.parent.mkdir(parents=True)
    canary = "mounted-auth-content-canary-0123456789"
    auth_store.write_text(canary)

    with pytest.raises(RuntimeError, match="unexpected guest auth path") as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                _ExecutingIsolatedProbeRunner(tmp_path),
                adapter="executing_isolated_probe",
                scope="isolated_guest",
                provider_canaries={"auth_store": canary},
                operational_env={
                    "HOME": str(guest),
                    "VISIBLE": "positive-control-0123456789",
                },
                guest_cwd="guest",
                guest_auth_search_paths={
                    "mounted_workspace": str(mounted_workspace),
                },
            )
        )

    assert canary not in str(exc_info.value)


def test_isolated_probe_executes_arbitrary_mounted_cayu_home_auth_search(
    tmp_path: Path,
) -> None:
    guest = tmp_path / "guest"
    mounted_workspace = tmp_path / "mounted-workspace"
    auth_store = mounted_workspace / "provider-auth-home" / "auth.json"
    guest.mkdir()
    auth_store.parent.mkdir(parents=True)
    canary = "mounted-cayu-home-content-canary-0123456789"
    auth_store.write_text(canary)

    with pytest.raises(RuntimeError, match="unexpected guest auth path") as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                _ExecutingIsolatedProbeRunner(tmp_path),
                adapter="executing_arbitrary_cayu_home_probe",
                scope="isolated_guest",
                provider_canaries={"auth_store": canary},
                operational_env={
                    "HOME": str(guest),
                    "VISIBLE": "positive-control-0123456789",
                },
                guest_cwd="guest",
                guest_auth_search_paths={
                    "mounted_workspace": str(mounted_workspace),
                },
            )
        )

    assert canary not in str(exc_info.value)
    captured = traceback.TracebackException.from_exception(
        exc_info.value,
        capture_locals=True,
    )
    cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
    assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])


@pytest.mark.parametrize(
    "relative_path",
    [
        Path(".aws/credentials"),
        Path(".aws/config"),
        Path(".config/gcloud/application_default_credentials.json"),
    ],
)
def test_isolated_probe_executes_common_cloud_auth_store_search(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    guest = tmp_path / "guest"
    mounted_workspace = tmp_path / "mounted-workspace"
    auth_store = mounted_workspace / relative_path
    guest.mkdir()
    auth_store.parent.mkdir(parents=True)
    canary = "mounted-cloud-auth-content-canary-0123456789"
    auth_store.write_text(canary)

    with pytest.raises(RuntimeError, match="unexpected guest auth path") as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                _ExecutingIsolatedProbeRunner(tmp_path),
                adapter="executing_cloud_auth_probe",
                scope="isolated_guest",
                provider_canaries={"cloud_auth_store": canary},
                operational_env={
                    "HOME": str(guest),
                    "VISIBLE": "positive-control-0123456789",
                },
                guest_cwd="guest",
                guest_auth_search_paths={
                    "mounted_workspace": str(mounted_workspace),
                },
            )
        )

    assert canary not in str(exc_info.value)


def test_isolated_probe_fails_closed_for_a_missing_configured_search_root(
    tmp_path: Path,
) -> None:
    guest = tmp_path / "guest"
    guest.mkdir()

    with pytest.raises(RuntimeError, match="auth path scan was incomplete"):
        asyncio.run(
            verify_provider_credential_isolation(
                _ExecutingIsolatedProbeRunner(tmp_path),
                adapter="executing_missing_auth_root_probe",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={
                    "HOME": str(guest),
                    "VISIBLE": "positive-control-0123456789",
                },
                guest_cwd="guest",
                guest_auth_search_paths={
                    "missing_mounted_workspace": str(tmp_path / "does-not-exist"),
                },
            )
        )


def test_isolated_probe_traverses_a_nested_directory_symlink_for_auth_stores(
    tmp_path: Path,
) -> None:
    guest = tmp_path / "guest"
    mounted_workspace = tmp_path / "mounted-workspace"
    mounted_project = mounted_workspace / "project"
    external_auth_store = tmp_path / "external-provider-auth" / "auth.json"
    guest.mkdir()
    mounted_project.mkdir(parents=True)
    external_auth_store.parent.mkdir()
    canary = "symlinked-auth-content-canary-0123456789"
    external_auth_store.write_text(canary)
    (mounted_project / ".cayu").symlink_to(external_auth_store.parent, target_is_directory=True)

    with pytest.raises(RuntimeError, match="unexpected guest auth path") as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                _ExecutingIsolatedProbeRunner(tmp_path),
                adapter="executing_symlinked_auth_probe",
                scope="isolated_guest",
                provider_canaries={"auth_store": canary},
                operational_env={
                    "HOME": str(guest),
                    "VISIBLE": "positive-control-0123456789",
                },
                guest_cwd="guest",
                guest_auth_search_paths={
                    "mounted_workspace": str(mounted_workspace),
                },
            )
        )

    assert canary not in str(exc_info.value)


def test_isolated_probe_rejects_incomplete_auth_search() -> None:
    class IncompleteSearchRunner(_RecordingProbeRunner):
        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            return ExecResult(
                stdout=json.dumps(
                    {
                        "environment": kwargs["env"],
                        "auth_paths": {},
                        "auth_scan_complete": False,
                        "provider_canary_matches": [],
                        "detector_control_match": True,
                    }
                )
            )

    with pytest.raises(RuntimeError, match="auth path scan was incomplete"):
        asyncio.run(
            verify_provider_credential_isolation(
                IncompleteSearchRunner(),
                adapter="incomplete_search",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )


def test_local_runner_cannot_be_reported_as_an_isolated_guest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LocalRunner cannot prove isolated_guest"):
        asyncio.run(
            verify_provider_credential_isolation(
                LocalRunner(tmp_path),
                adapter="local",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )


def test_relabelled_local_runner_cannot_be_reported_as_an_isolated_guest(
    tmp_path: Path,
) -> None:
    class RelabelledLocalRunner(LocalRunner):
        isolation = "fake-remote"

    with pytest.raises(ValueError, match="LocalRunner cannot prove isolated_guest"):
        asyncio.run(
            verify_provider_credential_isolation(
                RelabelledLocalRunner(tmp_path),
                adapter="relabelled_local",
                scope="isolated_guest",
                provider_canaries=_canaries(),
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )


def test_local_inherit_env_is_an_explicit_untrusted_code_escape_hatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = _canaries()["openai_api_key"]
    monkeypatch.setenv("OPENAI_API_KEY", canary)
    runner = LocalRunner(tmp_path, inherit_env=True)

    with pytest.raises(ProviderCredentialIsolationViolation) as exc_info:
        asyncio.run(
            verify_provider_credential_isolation(
                runner,
                adapter="local",
                scope="local_environment",
                provider_canaries={"openai_api_key": canary},
                operational_env={"VISIBLE": "positive-control-0123456789"},
            )
        )

    assert exc_info.value.canary_label == "openai_api_key"
    assert canary not in str(exc_info.value)


def test_provider_registration_does_not_delegate_authority_or_publish_it(
    provider_credential_canaries,
) -> None:
    values = provider_credential_canaries.values
    workload_secret = "declared-trusted-tool-secret-0123456789"
    assert all(value not in repr(provider_credential_canaries) for value in values.values())

    class CanaryAuth:
        async def credentials(self) -> OpenAISubscriptionCredentials:
            return OpenAISubscriptionCredentials(
                access_token=values["oauth_access_token"],
                refresh_token=values["oauth_refresh_token"],
                expires_at=2_000_000_000,
                account_id=values["account_id"],
            )

    app = CayuApp(enable_logging=False)
    app.register_provider(OpenAISubscriptionProvider(auth=CanaryAuth()), default=True)
    workload_vault = StaticVault({"trusted_tool_secret": workload_secret})
    workload_proxy = PassthroughProxy(workload_vault)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="trusted-tool"),
            vault=workload_vault,
            proxy=workload_proxy,
        ),
        default=True,
    )

    manifest = app.describe().model_dump_json()
    resolved_workload = asyncio.run(workload_proxy.resolve(SecretRef(name="trusted_tool_secret")))

    assert app.list_providers() == ("openai_subscription",)
    assert all(value not in manifest for value in values.values())
    assert resolved_workload.value.get_secret_value() == workload_secret
    assert workload_secret not in manifest
    # Provider registration does not silently turn model authority into a
    # workload registration. The separately declared trusted-tool credential
    # remains resolvable only through its explicit Vault/proxy boundary.
    assert app.redact_json({"candidate": values["oauth_access_token"]}) == {
        "candidate": values["oauth_access_token"]
    }


def test_concurrent_provider_refresh_never_enters_runner_configuration(
    tmp_path: Path,
    provider_credential_canaries,
) -> None:
    values = provider_credential_canaries.values
    store = OpenAISubscriptionAuthStore(provider_credential_canaries.auth_home / "auth.json")
    store.save(
        OpenAISubscriptionCredentials(
            access_token=values["oauth_access_token"],
            refresh_token=values["oauth_refresh_token"],
            expires_at=1,
            account_id=values["account_id"],
        )
    )
    refresh_started = threading.Event()
    release_refresh = threading.Event()
    new_access = "cayu-refreshed-access-canary-0123456789"
    new_refresh = "cayu-refreshed-refresh-canary-0123456789"

    class BlockingRefreshTransport:
        def refresh(self, refresh_token: str) -> dict[str, Any]:
            assert refresh_token == values["oauth_refresh_token"]
            refresh_started.set()
            if not release_refresh.wait(timeout=2):
                raise AssertionError("test did not release provider refresh")
            return {
                "access_token": new_access,
                "refresh_token": new_refresh,
                "expires_in": 3600,
            }

    auth = OpenAISubscriptionAuth(
        store=store,
        oauth_transport=BlockingRefreshTransport(),
        now=lambda: 1_000,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def exercise():
        refresh_task = asyncio.create_task(auth.credentials())
        assert await asyncio.to_thread(refresh_started.wait, 1)
        try:
            evidence = await verify_provider_credential_isolation(
                LocalRunner(workspace),
                adapter="local_during_provider_refresh",
                scope="local_environment",
                provider_canaries={
                    **values,
                    "refreshed_access_token": new_access,
                    "refreshed_refresh_token": new_refresh,
                    "auth_store_path": str(provider_credential_canaries.auth_home),
                },
                operational_env={
                    "CAYU_PROBE_VISIBLE": provider_credential_canaries.positive_env[
                        "CAYU_PROBE_VISIBLE"
                    ]
                },
            )
        finally:
            release_refresh.set()
        credentials = await refresh_task
        return evidence, credentials

    evidence, credentials = asyncio.run(exercise())

    assert evidence.status == "environment_minimized"
    assert credentials.access_token == new_access
    assert credentials.refresh_token == new_refresh


def test_provider_credential_authority_contract_is_documented() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_contracts = (root / "docs" / "runtime-contracts.md").read_text()
    runner_guide = (root / "docs" / "build-a-runner.md").read_text()
    subscription_guide = (root / "docs" / "openai-subscription.md").read_text()
    glossary = (root / "docs" / "glossary.md").read_text()
    credentials_module = (root / "src" / "cayu" / "credentials.py").read_text()

    assert "model-provider authority != workload authority" in runtime_contracts
    assert "never implies workload delegation" in runtime_contracts
    assert "helper subprocess" in runner_guide
    assert "verify_provider_credential_isolation" in runner_guide
    assert "LocalRunner(inherit_env=True)" in subscription_guide
    assert "filesystem security boundary" in subscription_guide
    assert "**Model-provider credential.**" in glossary
    assert "**Workload credential.**" in glossary
    assert "How a workload credential is delivered" in credentials_module
