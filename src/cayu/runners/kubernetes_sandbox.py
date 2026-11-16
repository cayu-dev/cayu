"""Run commands in a Kubernetes sandbox pod through the Cayu command sidecar.

The sidecar runs in the pod in Kubernetes mode (``CAYU_SIDECAR_MODE=kubernetes``):
it authenticates its host with a bearer token, reports a boot ID, and serves
the same command protocol as in a Lambda MicroVM. The control plane that owns
the pod creates, suspends, resumes and deletes it; this runner only drives the
sidecar.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import ssl
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx

from cayu._validation import (
    canonical_durable_json_bytes,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.runners._cleanup import (
    DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
    DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
    DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
    RunnerCleanupPolicy,
    validate_cancel_timeout,
    validate_runner_cleanup_policy,
)
from cayu.runners._sidecar_runner import (
    SidecarProtocolRunner,
    _nonnegative_float,
    _positive_float,
    _SidecarAdmissionObserver,
    _validate_carry_commands,
    _validate_owner_epoch,
)
from cayu.runners._sidecar_transport import (
    SIDECAR_ERRORS,
    SIDECAR_PROTOCOL_VERSION,
    HttpxSidecarEndpointTransport,
    SidecarEndpointTransport,
)
from cayu.runners._subprocess import copy_runner_env
from cayu.runners.base import RunnerInteractiveProcessCapability, RunnerSystemExecutionMode
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity

DEFAULT_KUBERNETES_SANDBOX_CWD = "/workspace"
DEFAULT_KUBERNETES_SANDBOX_POLL_INTERVAL_SECONDS = 0.1
# Bounds each sidecar endpoint call, including one stdin chunk during copy-in.
DEFAULT_KUBERNETES_SANDBOX_REQUEST_TIMEOUT_SECONDS = 120.0
# Health capabilities of a sidecar in Kubernetes mode. A Lambda-mode sidecar
# advertises neither: it relies on Lambda's endpoint token and is refused here.
_REQUIRED_CAPABILITIES = ("bearer_auth", "boot_id")
_BOOT_ID_PATTERN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_MAX_SANDBOX_IDENTITY_ENTRIES = 32
_LOGGER = logging.getLogger(__name__)

_KUBERNETES_SIDECAR_ERRORS = replace(SIDECAR_ERRORS, label="Kubernetes sandbox")

KubernetesSandboxToken = str | Callable[[], str] | Callable[[], Awaitable[str]]
KubernetesSandboxIdentityObserver = Callable[[], Awaitable[Mapping[str, str]]]


class _VerifiedAsyncClient:
    """One lazily created HTTP client that trusts exactly the configured certificates."""

    def __init__(self, verify: ssl.SSLContext) -> None:
        self._verify = verify
        self._client: httpx.AsyncClient | None = None

    def get(self) -> httpx.AsyncClient:
        client = self._client
        if client is None or client.is_closed:
            client = httpx.AsyncClient(verify=self._verify)
            self._client = client
        return client

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


class HttpxKubernetesSandboxEndpointTransport(HttpxSidecarEndpointTransport):
    """HTTPS transport to a Kubernetes-mode sidecar, authenticated with a bearer token.

    ``verify`` is a CA bundle path or an ``ssl.SSLContext`` the sidecar's
    certificate must chain to; without it the runtime's provider trust is used.
    A context that trusts only the sidecar's own self-signed certificate
    (``ssl.create_default_context(cadata=certificate_pem)``) pins it.
    """

    _errors = _KUBERNETES_SIDECAR_ERRORS

    def __init__(self, *, verify: str | os.PathLike[str] | ssl.SSLContext | None = None) -> None:
        super().__init__()
        if verify is not None:
            self._client = _VerifiedAsyncClient(_ssl_context(verify))

    def _auth_headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {require_clean_nonblank(token, 'endpoint token')}"}

    def _base_url(self, endpoint: str) -> str:
        return _endpoint_base_url(endpoint)


class KubernetesSandboxRunner(SidecarProtocolRunner):
    """Execute commands through the Cayu sidecar in a Kubernetes sandbox pod.

    ``endpoint`` is the sidecar's ``https://`` URL, which a Kubernetes-mode
    sidecar always serves; ``verify`` is what its certificate must chain to (see
    :class:`HttpxKubernetesSandboxEndpointTransport`). ``token`` is the bearer
    token the sidecar reads from its token file, or a callable (sync or async)
    returning it; a callable is called again once when the sidecar rejects the
    token, so a rotated token is picked up. ``sandbox`` names the pod's
    ``(namespace, name)``.

    Before its first claim the runner checks the sidecar's health and refuses a
    sidecar that does not advertise ``bearer_auth`` and ``boot_id``, such as one
    in Lambda mode. :attr:`boot_id` is the boot ID that check last saw; a new
    value means the sidecar or its command executor restarted and every command,
    forward and owner claim is gone, so the next fenced call reports this runner
    superseded and the host attaches a new runner.

    The sandbox has no separate trusted lane: ``exec_system`` runs as the agent
    (``system_execution_mode`` is ``"shared"``). The control plane owns the
    pod's lifecycle; ``close`` only stops serving and releases the transport.

    ``identity_observer`` is an optional async callable returning a small
    string mapping that identifies the running sandbox (for example the pod UID
    and image digest read from the Kubernetes API). Admission evidence is bound
    to it, together with the sidecar's boot ID.
    """

    isolation = "kubernetes-sandbox"
    system_execution_mode: RunnerSystemExecutionMode = "shared"
    default_cwd = DEFAULT_KUBERNETES_SANDBOX_CWD
    _errors = _KUBERNETES_SIDECAR_ERRORS
    _adapter = "kubernetes-sandbox"
    _runner_name = "KubernetesSandboxRunner"
    _logger = _LOGGER
    # Where the sandbox cannot nest PID namespaces, the sidecar tears an
    # interactive command down by its process group; a descendant that leaves
    # the group is not stopped with it.
    _interactive_process_trees = frozenset({"pid_namespace", "process_group"})

    def __init__(
        self,
        *,
        endpoint: str,
        token: KubernetesSandboxToken,
        sandbox: tuple[str, str],
        endpoint_transport: SidecarEndpointTransport | None = None,
        verify: str | os.PathLike[str] | ssl.SSLContext | None = None,
        identity_observer: KubernetesSandboxIdentityObserver | None = None,
        default_cwd: str = DEFAULT_KUBERNETES_SANDBOX_CWD,
        env_overlay: Mapping[str, str] | None = None,
        owner_epoch: int | None = None,
        carry_commands: Sequence[str] = (),
        poll_interval_s: float = DEFAULT_KUBERNETES_SANDBOX_POLL_INTERVAL_SECONDS,
        request_timeout_s: float = DEFAULT_KUBERNETES_SANDBOX_REQUEST_TIMEOUT_SECONDS,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
    ) -> None:
        self.endpoint = _endpoint_base_url(endpoint)
        self.namespace, self.name = _validate_sandbox(sandbox)
        self._token = _validate_token_source(token)
        if identity_observer is not None and not callable(identity_observer):
            raise TypeError("identity_observer must be an async callable or None.")
        self._identity_observer = identity_observer
        self._owner_epoch = _validate_owner_epoch(owner_epoch)
        self._carry_requested = _validate_carry_commands(carry_commands, self._owner_epoch)
        self.default_cwd = _validate_sandbox_root(default_cwd)
        self.poll_interval_s = _nonnegative_float(poll_interval_s, "poll_interval_s")
        self.request_timeout_s = _positive_float(request_timeout_s, "request_timeout_s")
        self.cancel_timeout_s = validate_cancel_timeout(cancel_timeout_s)
        self.cancellation_cleanup = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        self.timeout_cleanup = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        self.env_overlay = dict(env_overlay) if env_overlay else {}
        if endpoint_transport is not None and verify is not None:
            raise ValueError("verify configures only the built-in endpoint transport.")
        self._endpoint_transport = (
            endpoint_transport
            if endpoint_transport is not None
            else HttpxKubernetesSandboxEndpointTransport(verify=verify)
        )
        self._owns_endpoint_transport = endpoint_transport is None
        self._init_sidecar_state()
        self._cached_token: str | None = None
        self._token_lock = asyncio.Lock()
        self._boot_id: str | None = None
        # Fail closed: an overlay may carry secret values unless the adapter that
        # built it declares otherwise after verifying the sandbox boundary.
        self._env_overlay_secret_values_present = bool(self.env_overlay)

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("kubernetes-sandbox", self.namespace, self.name)

    @property
    def _resource_id(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def boot_id(self) -> str | None:
        """The sidecar boot ID the last health check saw; ``None`` before the first.

        It changes whenever the sidecar process starts, so a host that sees a
        new value knows the pod restarted and lost its commands and forwards.
        """

        return self._boot_id

    async def observe_boot_id(self) -> str:
        """Check the sidecar's authenticated health now and return its boot ID."""

        if self._closed or self._closing:
            raise RuntimeError(f"{self._runner_name} is closed.")
        await self._endpoint_health()
        assert self._boot_id is not None
        return self._boot_id

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        """Declare the runner configuration that shapes what commands observe."""

        configuration = canonical_durable_json_bytes(
            {
                "default_cwd": self.default_cwd,
                "env_overlay_keys": sorted(self.env_overlay),
                "protocol_version": SIDECAR_PROTOCOL_VERSION,
            },
            "kubernetes_sandbox_runner_configuration",
        )
        return ExecutionProfileBehaviorIdentity(
            name="cayu.kubernetes_sandbox_runner",
            behavior_version="1",
            implementation_version=sha256(configuration).hexdigest(),
        )

    def interactive_process_capability(self) -> RunnerInteractiveProcessCapability | None:
        """Stream a process's stdio through sidecar protocol 6.

        Where the sandbox allows nested PID namespaces the command gets its
        own, and closing it confirms the whole tree exited. Otherwise the
        sidecar reports a ``process_group`` tree: closing the process stops its
        command and signals the command's process group, and a descendant that
        left that group (``setsid``, a double fork into a new group) is not
        stopped with it and keeps running until the sandbox's agent namespace ends.
        """

        return super().interactive_process_capability()

    def _needs_health_before_claim(self) -> bool:
        # Only a sidecar that authenticates its host may be claimed.
        return not self._owner_epoch_verified

    def _observe_health(self, response: Mapping[str, Any], capabilities: object) -> None:
        if not isinstance(capabilities, list) or not all(
            capability in capabilities for capability in _REQUIRED_CAPABILITIES
        ):
            raise self._errors.version_mismatch(
                "Kubernetes sandbox sidecar does not advertise bearer_auth and boot_id; "
                "run it with CAYU_SIDECAR_MODE=kubernetes."
            )
        boot_id = response.get("boot_id")
        if type(boot_id) is not str or _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
            raise self._errors.protocol("Kubernetes sandbox health returned no valid boot_id.")
        self._boot_id = boot_id

    async def _endpoint_token(self, *, force_refresh: bool = False) -> str:
        source = self._token
        if isinstance(source, str):
            return source
        cached = self._cached_token
        if cached is not None and not force_refresh:
            return cached
        async with self._token_lock:
            if self._cached_token is not None and not force_refresh:
                return self._cached_token
            value = source()
            if inspect.isawaitable(value):
                value = await value
            if type(value) is not str:
                raise TypeError("Kubernetes sandbox token callable must return a string.")
            token = require_clean_nonblank(value, "token")
            self._cached_token = token
            return token

    def _admission_local_identity(self) -> tuple[object, ...]:
        return (
            self.namespace,
            self.name,
            self.endpoint,
            self.default_cwd,
            tuple(sorted(copy_runner_env(self.env_overlay, inherit_env=False))),
        )

    def _admission_observer(self, requirements) -> _KubernetesAdmissionObserver:
        return _KubernetesAdmissionObserver(self, requirements)

    async def _sandbox_identity(self) -> tuple[tuple[str, str], ...]:
        observer = self._identity_observer
        if observer is None:
            return ()
        async with asyncio.timeout(self.request_timeout_s):
            observed = await observer()
        if not isinstance(observed, Mapping) or len(observed) > _MAX_SANDBOX_IDENTITY_ENTRIES:
            raise self._errors.error(
                "Kubernetes sandbox identity_observer must return a mapping of at most "
                f"{_MAX_SANDBOX_IDENTITY_ENTRIES} strings."
            )
        entries: dict[str, str] = {}
        for key, value in observed.items():
            if type(key) is not str or type(value) is not str:
                raise self._errors.error(
                    "Kubernetes sandbox identity_observer must return string keys and values."
                )
            entries[require_durable_clean_nonblank(key, "sandbox identity key")] = (
                require_durable_clean_nonblank(value, "sandbox identity value")
            )
        return tuple(sorted(entries.items()))

    async def close(self) -> None:
        """Stop serving commands and close the endpoint transport; the pod is left as is."""

        await self._settle_terminal_lifecycle(
            self._close_transport,
            action="none",
            timeout_s=self.cancel_timeout_s,
        )

    async def _close_transport(self) -> None:
        self._cached_token = None
        if self._owns_endpoint_transport:
            close = getattr(self._endpoint_transport, "aclose", None)
            if callable(close):
                await close()


@dataclass(frozen=True, slots=True)
class _KubernetesAdmissionIdentity:
    """One complete observation of the exact sandbox an admission probe ran in."""

    local: tuple[object, ...]
    protocol_version: str
    boot_id: str
    sandbox: tuple[tuple[str, str], ...]


@dataclass(frozen=True, repr=False)
class _KubernetesAdmissionObserver(_SidecarAdmissionObserver):
    """Probe executables through one exact, unchanged sandbox.

    Identity is the runner's configuration, the boot ID from the sidecar's
    authenticated health, and the ``identity_observer`` output, read before and
    after the probes; any change fails closed.
    """

    runner: KubernetesSandboxRunner
    _candidate_name: ClassVar[str] = "kubernetes-sandbox"

    def _fingerprint(self, identity: _KubernetesAdmissionIdentity | None) -> str:
        runner = self.runner
        return (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    {
                        "namespace": runner.namespace,
                        "name": runner.name,
                        "endpoint": runner.endpoint,
                        "root": runner.default_cwd,
                        "environment_keys": sorted(
                            copy_runner_env(runner.env_overlay, inherit_env=False)
                        ),
                        "protocol_version": None if identity is None else identity.protocol_version,
                        "boot_id": None if identity is None else identity.boot_id,
                        "sandbox": None if identity is None else dict(identity.sandbox),
                    },
                    "kubernetes_sandbox_execution_identity",
                )
            ).hexdigest()
        )

    async def _observe_identity(self) -> _KubernetesAdmissionIdentity:
        runner = self.runner
        runner._require_admission_open()
        local = runner._admission_local_identity()
        # Proves the running sidecar speaks this protocol and authenticates its host.
        await runner._endpoint_health()
        boot_id = runner.boot_id
        assert boot_id is not None
        return _KubernetesAdmissionIdentity(
            local=local,
            protocol_version=SIDECAR_PROTOCOL_VERSION,
            boot_id=boot_id,
            sandbox=await runner._sandbox_identity(),
        )


def _endpoint_base_url(endpoint: str) -> str:
    value = require_clean_nonblank(endpoint, "endpoint").rstrip("/")
    try:
        parsed = urlsplit(value)
        parsed.port  # noqa: B018 - validates the port
    except ValueError as exc:
        raise ValueError("Kubernetes sandbox endpoint must be an https:// URL.") from exc
    if parsed.scheme != "https":
        # The bearer token never crosses the network in clear.
        raise ValueError("Kubernetes sandbox endpoint must use https://.")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "Kubernetes sandbox endpoint must name a host, without credentials, query or fragment."
        )
    return value


def _validate_sandbox(sandbox: tuple[str, str]) -> tuple[str, str]:
    if not isinstance(sandbox, tuple) or len(sandbox) != 2:
        raise TypeError("sandbox must be a (namespace, name) tuple.")
    namespace, name = sandbox
    return (
        require_durable_clean_nonblank(namespace, "sandbox namespace"),
        require_durable_clean_nonblank(name, "sandbox name"),
    )


def _validate_token_source(token: KubernetesSandboxToken) -> KubernetesSandboxToken:
    if isinstance(token, str):
        return require_clean_nonblank(token, "token")
    if callable(token):
        return token
    raise TypeError("token must be a string or a callable returning one.")


def _validate_sandbox_root(value: str) -> str:
    root = require_durable_clean_nonblank(value, "default_cwd")
    if not root.startswith("/"):
        raise ValueError("KubernetesSandboxRunner default_cwd must be an absolute path.")
    return root.rstrip("/") or "/"


def _ssl_context(verify: str | os.PathLike[str] | ssl.SSLContext) -> ssl.SSLContext:
    if isinstance(verify, ssl.SSLContext):
        return verify
    if isinstance(verify, str | os.PathLike):
        return ssl.create_default_context(cafile=os.fspath(verify))
    raise TypeError("verify must be a CA bundle path, an ssl.SSLContext, or None.")


__all__ = [
    "DEFAULT_KUBERNETES_SANDBOX_CWD",
    "HttpxKubernetesSandboxEndpointTransport",
    "KubernetesSandboxRunner",
]
