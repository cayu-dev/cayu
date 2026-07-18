from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import importlib
import json
import os
import posixpath
import secrets
import stat
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, TypedDict

from cayu.egress._remote_adapter import (
    ProxyServerFactory,
    prepare_exposed_proxy_binding,
    run_enforcement_preflight,
    run_setup_commands,
)
from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    EgressBinding,
    SandboxEgressAdapter,
    VirtualEgressRunnerRequest,
    _await_bounded_cleanup_task,
    _consume_accounted_task_cancellation,
    _raise_primary_with_cleanup_cancellation,
)
from cayu.egress.broker import TransparentEgressBroker
from cayu.egress.errors import (
    EgressReconnectConflictError,
    EgressReconnectError,
    EgressReconnectNotFoundError,
    InvalidEgressReconnectMetadataError,
    UnsupportedEgressError,
    UnsupportedEgressReconnectError,
)
from cayu.egress.grants import VirtualCredentialGrant
from cayu.egress.proxy_exposure import (
    MICROSANDBOX_HOST,
    MicrosandboxHostProxyExposure,
    ProxyExposure,
)
from cayu.egress.proxy_server import DualStackLoopbackEgressProxyServer
from cayu.runners.base import ExecCommand, Runner
from cayu.runners.microsandbox import (
    DEFAULT_MICROSANDBOX_RECONNECT_TIMEOUT_SECONDS,
    MicrosandboxReconnectIdentityError,
    MicrosandboxRunner,
)


class _ReconnectIdentity(TypedDict):
    sandbox_name: str
    sandbox_created_at: float
    proxy_listener_port: int
    proxy_endpoint_port: int
    ownership_id: str
    owner_session_id: str
    owner_environment_name: str


@dataclass
class _ReconnectClaim:
    lock_path: Path
    attestation_path: Path
    handle: BinaryIO
    remove_on_close: bool = False
    closed: bool = False

    def read_identity(self) -> dict[str, Any]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.attestation_path, flags)
        except FileNotFoundError as exc:
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect attestation is missing."
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise InvalidEgressReconnectMetadataError(
                    "Microsandbox reconnect attestation is not a regular file."
                )
            with os.fdopen(descriptor, "rb") as handle:
                payload = handle.read()
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            raise
        if not payload:
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect attestation is missing."
            )
        try:
            value = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect attestation is malformed."
            ) from exc
        if not isinstance(value, dict):
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect attestation must be an object."
            )
        return value

    def write_identity(self, identity: _ReconnectIdentity) -> None:
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.attestation_path, flags, 0o600)
        except OSError as exc:
            raise EgressReconnectError(
                "Microsandbox reconnect attestation could not be written."
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise EgressReconnectError(
                    "Microsandbox reconnect attestation is not a regular file."
                )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", buffering=0) as handle:
                handle.write(payload)
                os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(descriptor)
            raise

    def close(self) -> None:
        if self.closed:
            return
        if self.remove_on_close:
            with contextlib.suppress(FileNotFoundError):
                self.attestation_path.unlink()
        try:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.closed = True


def _default_reconnect_state_dir() -> Path:
    owner = getattr(os, "getuid", lambda: "user")()
    return Path(tempfile.gettempdir()) / f"cayu-microsandbox-egress-{owner}"


class MicrosandboxEgressAdapter(SandboxEgressAdapter):
    """Enforced virtual egress for local Microsandbox microVMs."""

    runner_kind = "microsandbox"
    supports_reconnect = True

    def __init__(
        self,
        *,
        exposure: ProxyExposure | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        microsandbox_module: ModuleType | Any | None = None,
        proxy_server_factory: ProxyServerFactory = DualStackLoopbackEgressProxyServer,
        preflight_timeout_s: int = 15,
        reconnect_timeout_s: float = DEFAULT_MICROSANDBOX_RECONNECT_TIMEOUT_SECONDS,
        reconnect_state_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if type(preflight_timeout_s) is not int or preflight_timeout_s <= 0:
            raise ValueError("preflight_timeout_s must be a positive integer.")
        if (
            type(reconnect_timeout_s) not in {int, float}
            or not isfinite(reconnect_timeout_s)
            or reconnect_timeout_s <= 0
        ):
            raise ValueError("reconnect_timeout_s must be a positive finite number.")
        self._exposure = exposure or MicrosandboxHostProxyExposure()
        self._loop = loop
        self._module = microsandbox_module
        self._proxy_server_factory = proxy_server_factory
        self._preflight_timeout_s = preflight_timeout_s
        self._reconnect_timeout_s = float(reconnect_timeout_s)
        self._reconnect_state_dir = (
            _default_reconnect_state_dir()
            if reconnect_state_dir is None
            else Path(reconnect_state_dir).expanduser()
        )
        if not self._reconnect_state_dir.is_absolute():
            raise ValueError("reconnect_state_dir must be an absolute path.")
        self._claims_by_name: dict[str, _ReconnectClaim] = {}
        self._runner_claims: weakref.WeakKeyDictionary[Runner, _ReconnectClaim] = (
            weakref.WeakKeyDictionary()
        )
        self._runner_identities: weakref.WeakKeyDictionary[Runner, _ReconnectIdentity] = (
            weakref.WeakKeyDictionary()
        )

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        return await prepare_exposed_proxy_binding(
            runner_kind=self.runner_kind,
            session_id=session_id,
            broker=broker,
            grants=grants,
            exposure=self._exposure,
            bind_host="127.0.0.1",
            loop=self._loop,
            proxy_server_factory=self._proxy_server_factory,
        )

    async def prepare_reconnect(
        self,
        *,
        session_id: str,
        environment_name: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
        reconnect_metadata: Mapping[str, Any],
    ) -> EgressBinding:
        identity = _reconnect_identity(reconnect_metadata)
        if identity["owner_session_id"] != session_id:
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect identity belongs to a different session."
            )
        if identity["owner_environment_name"] != environment_name:
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect identity belongs to a different environment."
            )
        claim = self._acquire_claim(identity["sandbox_name"])
        try:
            attested = self.validate_reconnect_metadata(claim.read_identity())
            if attested != identity:
                raise InvalidEgressReconnectMetadataError(
                    "Microsandbox reconnect identity does not match its durable attestation."
                )
            binding = await prepare_exposed_proxy_binding(
                runner_kind=self.runner_kind,
                session_id=session_id,
                broker=broker,
                grants=grants,
                exposure=self._exposure,
                bind_host="127.0.0.1",
                bind_port=identity["proxy_listener_port"],
                loop=self._loop,
                proxy_server_factory=self._proxy_server_factory,
            )
        except OSError as exc:
            claim.close()
            if exc.errno == errno.EADDRINUSE:
                raise EgressReconnectConflictError(
                    "Microsandbox virtual-egress reconnect could not reclaim its original "
                    "host proxy listener port "
                    f"{identity['proxy_listener_port']}; another owner is active."
                ) from exc
            raise EgressReconnectError(
                "Microsandbox virtual-egress reconnect could not re-establish its host "
                f"proxy on port {identity['proxy_listener_port']}."
            ) from exc
        except BaseException:
            claim.close()
            raise
        self._hold_claim(
            sandbox_name=identity["sandbox_name"],
            binding=binding,
            claim=claim,
        )
        return binding

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> Runner:
        if request.runner_kind != self.runner_kind:
            raise UnsupportedEgressError(
                f"Microsandbox adapter cannot create runner kind {request.runner_kind!r}."
            )
        proxy_endpoint = request.binding.proxy_endpoint
        if proxy_endpoint is None:
            raise UnsupportedEgressError("Microsandbox egress binding did not provide proxy_url.")
        if proxy_endpoint.host != MICROSANDBOX_HOST:
            raise UnsupportedEgressError(
                "Microsandbox virtual egress requires host.microsandbox.internal exposure."
            )

        reconnect = _optional_reconnect_identity(request.reconnect_metadata)
        if request.session_id is None or not request.session_id.strip():
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox virtual egress requires a nonblank owner session."
            )
        if request.environment_name is None or not request.environment_name.strip():
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox virtual egress requires a nonblank owner environment."
            )
        if reconnect is not None and (
            reconnect["owner_session_id"] != request.session_id
            or reconnect["owner_environment_name"] != request.environment_name
        ):
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect identity scope does not match the runner request."
            )
        if reconnect is not None and proxy_endpoint.port != reconnect["proxy_endpoint_port"]:
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox reconnect guest proxy endpoint changed from durable metadata."
            )
        module = self._microsandbox_module()
        created_new = reconnect is None
        runner: MicrosandboxRunner | None = None
        unbound_claim: _ReconnectClaim | None = None
        try:
            if reconnect is None:
                network = module.Network(
                    policy=module.NetworkPolicy(
                        default_egress=module.Action.DENY,
                        rules=(
                            *module.Rule.allow_dns(),
                            module.Rule.allow(
                                destination=module.Destination.group(module.DestGroup.HOST),
                                protocol=module.Protocol.TCP,
                                port=proxy_endpoint.port,
                            ),
                        ),
                    )
                )
                guest_ca_dir = posixpath.dirname(request.guest_ca_path)
                patches = [
                    module.Patch.mkdir(guest_ca_dir, mode=0o755),
                    module.Patch.copy_file(
                        request.ca_cert_host_path,
                        request.guest_ca_path,
                        mode=0o644,
                        replace=True,
                    ),
                ]
                runner = await MicrosandboxRunner.create(
                    request.name,
                    image=request.image,
                    close_action="remove",
                    env_overlay=request.env_overlay,
                    network=network,
                    patches=patches,
                    sandbox_module=module,
                )
                proxy_listener_port = request.binding.proxy_port
                if type(proxy_listener_port) is not int or not 1 <= proxy_listener_port <= 65535:
                    raise UnsupportedEgressError(
                        "Microsandbox egress binding did not provide its host proxy listener port."
                    )
                identity: _ReconnectIdentity = {
                    "sandbox_name": runner.name,
                    "sandbox_created_at": await self._sandbox_created_at(module, runner.name),
                    "proxy_listener_port": proxy_listener_port,
                    "proxy_endpoint_port": proxy_endpoint.port,
                    "ownership_id": secrets.token_hex(16),
                    "owner_session_id": request.session_id,
                    "owner_environment_name": request.environment_name,
                }
                unbound_claim = self._acquire_claim(runner.name)
                unbound_claim.write_identity(identity)
                self._hold_claim(
                    sandbox_name=runner.name,
                    binding=request.binding,
                    claim=unbound_claim,
                )
                unbound_claim = None
            else:
                try:
                    runner = await MicrosandboxRunner.from_existing(
                        reconnect["sandbox_name"],
                        close_action="detach",
                        env_overlay=request.env_overlay,
                        sandbox_module=module,
                        reconnect_timeout_s=self._reconnect_timeout_s,
                        expected_created_at=reconnect["sandbox_created_at"],
                    )
                except BaseException as exc:
                    if _is_sandbox_not_found(module, exc):
                        raise EgressReconnectNotFoundError(
                            "Microsandbox reconnect sandbox no longer exists: "
                            f"{reconnect['sandbox_name']}."
                        ) from exc
                    if isinstance(exc, TimeoutError):
                        raise EgressReconnectError(
                            "Microsandbox reconnect timed out while attaching to sandbox "
                            f"{reconnect['sandbox_name']!r}."
                        ) from exc
                    if isinstance(exc, MicrosandboxReconnectIdentityError):
                        raise InvalidEgressReconnectMetadataError(
                            "Microsandbox reconnect sandbox incarnation no longer matches "
                            "durable identity."
                        ) from exc
                    raise
                if runner.name != reconnect["sandbox_name"]:
                    raise InvalidEgressReconnectMetadataError(
                        "Microsandbox reconnect attached a different sandbox identity."
                    )
                await _install_ca(runner, request)
                identity = reconnect
            claim = self._claims_by_name.get(identity["sandbox_name"])
            if claim is None:
                raise EgressReconnectConflictError(
                    "Microsandbox reconnect ownership claim was lost before runner handoff."
                )
            self._runner_claims[runner] = claim
            self._runner_identities[runner] = identity
            await run_enforcement_preflight(
                runner,
                request,
                timeout_s=self._preflight_timeout_s,
            )
            await run_setup_commands(runner, request)
            return runner
        except BaseException as original:
            rollback_complete = False
            rollback_cancelled = False
            rollback_failure: BaseException | None = None
            if runner is not None:
                _consume_accounted_task_cancellation(original)
                runner.close_action = "remove" if created_new else "detach"
                cleanup_task = asyncio.create_task(runner.close())
                try:
                    rollback_cancelled = await _await_bounded_cleanup_task(
                        cleanup_task,
                        timeout_s=DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
                        timeout_message="Microsandbox egress runner rollback timed out.",
                    )
                    rollback_complete = True
                except BaseException as cleanup_error:
                    original.add_note(
                        "Microsandbox egress runner rollback incomplete: "
                        f"{type(cleanup_error).__name__}."
                    )
                    try:
                        _raise_primary_with_cleanup_cancellation(
                            original,
                            cleanup_error,
                            message=(
                                "Microsandbox egress runner rollback failed after cancellation."
                            ),
                        )
                    except BaseException as aggregate:
                        # The reconnect ownership claim below must be released
                        # before propagating the authoritative failure aggregate.
                        rollback_failure = aggregate
            if created_new and rollback_complete and runner is not None:
                claim = unbound_claim or self._claims_by_name.get(runner.name)
                if claim is not None:
                    claim.remove_on_close = True
            if unbound_claim is not None:
                unbound_claim.close()
            if rollback_failure is not None:
                raise rollback_failure from rollback_failure.__cause__
            if rollback_cancelled:
                cancellation = asyncio.CancelledError()
                raise BaseExceptionGroup(
                    "Microsandbox egress runner rollback completed after cancellation.",
                    [original, cancellation],
                ) from cancellation
            raise

    def reconnect_metadata(self, runner: Runner) -> dict[str, Any]:
        if not isinstance(runner, MicrosandboxRunner):
            raise TypeError("Microsandbox adapter received a different runner type.")
        identity = self._runner_identities.get(runner)
        if identity is None:
            raise InvalidEgressReconnectMetadataError(
                "Microsandbox runner is missing its durable reconnect identity."
            )
        return dict(identity)

    def validate_reconnect_metadata(
        self,
        reconnect_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        return dict(_reconnect_identity(reconnect_metadata))

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        if not isinstance(runner, MicrosandboxRunner):
            raise TypeError("Microsandbox adapter received a different runner type.")
        if outcome == "interrupted":
            runner.close_action = "detach"
            await runner.close()
            return

        claim = self._runner_claims.get(runner)
        removal_runner = runner
        if runner.closed:
            identity = self._runner_identities.get(runner)
            if identity is None:
                raise InvalidEgressReconnectMetadataError(
                    "Microsandbox runner is missing identity for terminal cleanup retry."
                )
            if claim is None or claim.closed:
                claim = self._acquire_claim(identity["sandbox_name"])
                try:
                    attested = self.validate_reconnect_metadata(claim.read_identity())
                    if attested != identity:
                        raise InvalidEgressReconnectMetadataError(
                            "Microsandbox terminal cleanup identity does not match attestation."
                        )
                except BaseException:
                    claim.close()
                    raise
                self._claims_by_name[identity["sandbox_name"]] = claim
                self._runner_claims[runner] = claim
            try:
                removal_runner = await MicrosandboxRunner.from_existing(
                    identity["sandbox_name"],
                    close_action="remove",
                    sandbox_module=self._microsandbox_module(),
                    reconnect_timeout_s=self._reconnect_timeout_s,
                    expected_created_at=identity["sandbox_created_at"],
                )
            except BaseException as exc:
                if _is_sandbox_not_found(self._microsandbox_module(), exc):
                    claim.remove_on_close = True
                    claim.close()
                    self._claims_by_name.pop(identity["sandbox_name"], None)
                    self._runner_claims.pop(runner, None)
                    return
                raise
        else:
            removal_runner.close_action = "remove"

        await removal_runner.close()
        if claim is not None:
            claim.remove_on_close = True
            claim.close()
            identity = self._runner_identities.get(runner)
            if identity is not None:
                self._claims_by_name.pop(identity["sandbox_name"], None)
            self._runner_claims.pop(runner, None)

    async def _sandbox_created_at(self, module: Any, sandbox_name: str) -> float:
        try:
            async with asyncio.timeout(self._reconnect_timeout_s):
                handle = await module.Sandbox.get(sandbox_name)
        except TimeoutError as exc:
            raise EgressReconnectError(
                "Microsandbox provider identity lookup timed out after sandbox creation."
            ) from exc
        value = getattr(handle, "created_at", None)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value <= 0
        ):
            raise EgressReconnectError(
                "Microsandbox provider did not expose a valid sandbox incarnation."
            )
        return float(value)

    def _acquire_claim(self, sandbox_name: str) -> _ReconnectClaim:
        try:
            import fcntl
        except ModuleNotFoundError as exc:  # pragma: no cover - non-POSIX host
            raise UnsupportedEgressReconnectError(
                "Microsandbox virtual-egress reconnect requires POSIX file locking."
            ) from exc
        try:
            self._reconnect_state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory_mode = self._reconnect_state_dir.lstat().st_mode
        except OSError as exc:
            raise EgressReconnectError(
                "Microsandbox reconnect state directory is not writable."
            ) from exc
        if not stat.S_ISDIR(directory_mode) or stat.S_IMODE(directory_mode) & 0o077:
            raise EgressReconnectError(
                "Microsandbox reconnect state directory must be a private mode-0700 directory."
            )
        digest = hashlib.sha256(sandbox_name.encode("utf-8")).hexdigest()
        lock_path = self._reconnect_state_dir / f"{digest}.lock"
        attestation_path = self._reconnect_state_dir / f"{digest}.json"
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise EgressReconnectError(
                "Microsandbox reconnect attestation could not be opened."
            ) from exc
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise EgressReconnectError(
                    "Microsandbox reconnect attestation is not a regular file."
                )
            os.fchmod(handle.fileno(), 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise EgressReconnectConflictError(
                f"Microsandbox sandbox {sandbox_name!r} already has an active reconnect owner."
            ) from exc
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise EgressReconnectConflictError(
                    f"Microsandbox sandbox {sandbox_name!r} already has an active reconnect owner."
                ) from exc
            raise EgressReconnectError(
                "Microsandbox reconnect ownership claim could not be acquired."
            ) from exc
        except BaseException:
            handle.close()
            raise
        return _ReconnectClaim(
            lock_path=lock_path,
            attestation_path=attestation_path,
            handle=handle,
        )

    def _hold_claim(
        self,
        *,
        sandbox_name: str,
        binding: EgressBinding,
        claim: _ReconnectClaim,
    ) -> None:
        original_teardown = binding.teardown

        async def teardown() -> None:
            if original_teardown is not None:
                await original_teardown()
            claim.close()
            if self._claims_by_name.get(sandbox_name) is claim:
                self._claims_by_name.pop(sandbox_name, None)

        binding.teardown = teardown
        self._claims_by_name[sandbox_name] = claim

    def _microsandbox_module(self) -> ModuleType | Any:
        if self._module is not None:
            return self._module
        try:
            return importlib.import_module("microsandbox")
        except ModuleNotFoundError as exc:
            if exc.name != "microsandbox":
                raise
            raise UnsupportedEgressError(
                "Microsandbox virtual egress requires the optional microsandbox package."
            ) from exc


def _optional_reconnect_identity(metadata: Mapping[str, Any]) -> _ReconnectIdentity | None:
    if not metadata:
        return None
    return _reconnect_identity(metadata)


def _reconnect_identity(metadata: Mapping[str, Any]) -> _ReconnectIdentity:
    expected = {
        "sandbox_name",
        "sandbox_created_at",
        "proxy_listener_port",
        "proxy_endpoint_port",
        "ownership_id",
        "owner_session_id",
        "owner_environment_name",
    }
    if set(metadata) != expected:
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect identity has an invalid schema."
        )
    sandbox_name = metadata["sandbox_name"]
    if not isinstance(sandbox_name, str) or not sandbox_name.strip():
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect sandbox_name must be nonblank."
        )
    sandbox_created_at = metadata["sandbox_created_at"]
    if (
        type(sandbox_created_at) not in {int, float}
        or not isfinite(sandbox_created_at)
        or sandbox_created_at <= 0
    ):
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect sandbox_created_at must be a positive finite number."
        )
    proxy_listener_port = metadata["proxy_listener_port"]
    if type(proxy_listener_port) is not int or not 1 <= proxy_listener_port <= 65535:
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect proxy_listener_port must be an integer between 1 and 65535."
        )
    proxy_endpoint_port = metadata["proxy_endpoint_port"]
    if type(proxy_endpoint_port) is not int or not 1 <= proxy_endpoint_port <= 65535:
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect proxy_endpoint_port must be an integer between 1 and 65535."
        )
    ownership_id = metadata["ownership_id"]
    if not isinstance(ownership_id, str) or len(ownership_id) != 32:
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect ownership_id must be a 32-character identifier."
        )
    try:
        int(ownership_id, 16)
    except ValueError as exc:
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect ownership_id must be hexadecimal."
        ) from exc
    owner_session_id = metadata["owner_session_id"]
    if not isinstance(owner_session_id, str) or not owner_session_id.strip():
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect owner_session_id must be nonblank."
        )
    owner_environment_name = metadata["owner_environment_name"]
    if not isinstance(owner_environment_name, str) or not owner_environment_name.strip():
        raise InvalidEgressReconnectMetadataError(
            "Microsandbox reconnect owner_environment_name must be nonblank."
        )
    return {
        "sandbox_name": sandbox_name.strip(),
        "sandbox_created_at": float(sandbox_created_at),
        "proxy_listener_port": proxy_listener_port,
        "proxy_endpoint_port": proxy_endpoint_port,
        "ownership_id": ownership_id,
        "owner_session_id": owner_session_id.strip(),
        "owner_environment_name": owner_environment_name.strip(),
    }


def _is_sandbox_not_found(module: Any, exc: BaseException) -> bool:
    error_type = getattr(module, "SandboxNotFoundError", None)
    return isinstance(error_type, type) and isinstance(exc, error_type)


async def _install_ca(
    runner: MicrosandboxRunner,
    request: VirtualEgressRunnerRequest,
) -> None:
    certificate = request.binding.ca_cert_pem
    if not certificate:
        raise UnsupportedEgressError(
            "Microsandbox egress binding did not provide a session CA certificate."
        )
    script = (
        "import os, pathlib, sys; "
        "path = pathlib.Path(sys.argv[1]); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_bytes(sys.stdin.buffer.read()); "
        "os.chmod(path, 0o644)"
    )
    result = await runner.exec(
        ExecCommand.process("python3", "-c", script, request.guest_ca_path),
        stdin=certificate.decode("utf-8"),
        timeout_s=30,
    )
    if result.exit_code != 0 or result.timed_out:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise UnsupportedEgressError(
            f"Microsandbox failed to install its refreshed virtual-egress CA: {detail}"
        )
