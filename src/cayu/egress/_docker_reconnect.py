"""Local Docker reconnect ownership and exact-allocation settlement.

The shared private directory is host control-plane state, never guest storage.
Lock files are permanent: unlinking one would permit two locks for one allocation.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from cayu.credentials import CredentialMode
from cayu.egress._remote_adapter import run_enforcement_preflight
from cayu.egress.adapter import EgressBinding, RunnerFinalizationResult, VirtualEgressRunnerRequest
from cayu.egress.errors import DockerEgressReconnectError, InvalidEgressReconnectMetadataError
from cayu.runners.docker import DockerRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cayu.egress.broker import TransparentEgressBroker
    from cayu.egress.docker_adapter import DockerEgressAdapter
    from cayu.egress.grants import VirtualCredentialGrant
    from cayu.runners.base import ExecCommand, ExecResult

OWNER_LABEL = "cayu.egress.allocation"
PROXY_ALIAS = "cayu-egress-proxy"
_ID = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[0-9a-f]{32}\Z")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "version",
        "backend",
        "allocation_id",
        "container_id",
        "container_name",
        "network_id",
        "session_id",
        "environment_name",
        "image_id",
        "configuration",
        "container_configuration",
        "runner_configuration",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != fields
        or type(value.get("version")) is not int
        or value["version"] != 1
        or value["backend"] != "docker"
    ):
        raise InvalidEgressReconnectMetadataError("Invalid Docker reconnect identity schema.")
    for key in fields - {"version"}:
        item = value[key]
        if type(item) is not str or not item or len(item) > 512 or item != item.strip():
            raise InvalidEgressReconnectMetadataError("Invalid Docker reconnect identity field.")
    if (
        not _TOKEN.fullmatch(value["allocation_id"])
        or any(
            not _ID.fullmatch(value[key])
            for key in (
                "container_id",
                "network_id",
                "configuration",
                "container_configuration",
                "runner_configuration",
            )
        )
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", value["image_id"])
    ):
        raise InvalidEgressReconnectMetadataError("Invalid Docker reconnect allocation identity.")
    return dict(value)


@dataclass(eq=False)
class _Claim:
    root: Path
    token: str
    handle: BinaryIO
    journal: dict[str, Any] = field(default_factory=dict)
    runner: _OwnedDockerRunner | None = None
    closed: bool = False

    @property
    def ca_path(self) -> Path:
        return self.root / f"{self.token}.ca.pem"

    def require_owned(self) -> None:
        if self.closed or self.handle.closed:
            raise DockerEgressReconnectError("ownership_uncertain")
        try:
            actual = os.fstat(self.handle.fileno())
            current = (self.root / f"{self.token}.lock").lstat()
            if (actual.st_dev, actual.st_ino) != (current.st_dev, current.st_ino):
                raise DockerEgressReconnectError("ownership_uncertain")
        except OSError:
            raise DockerEgressReconnectError("ownership_uncertain") from None

    def read(self) -> None:
        self.require_owned()
        path = self.root / f"{self.token}.json"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_size > 16384
                ):
                    raise ValueError
                self.journal = json.load(stream)
            if type(self.journal) is not dict or self.journal.get("state") not in {
                "creating",
                "active",
                "retained",
                "recovering",
                "disposal_pending",
                "ownership_uncertain",
                "disposed",
            }:
                raise ValueError
        except (OSError, ValueError, TypeError):
            raise DockerEgressReconnectError("state_unavailable") from None

    def write(self, **updates: Any) -> None:
        self.require_owned()
        self.journal.update(updates)
        data = json.dumps(self.journal, sort_keys=True, separators=(",", ":")).encode()
        if len(data) > 16384:
            raise DockerEgressReconnectError("state_unavailable")
        temporary = self.root / f"{self.token}.{secrets.token_hex(8)}.tmp"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / f"{self.token}.json")
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            raise DockerEgressReconnectError("state_unavailable") from None
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def install_ca(self, content: bytes) -> None:
        self.require_owned()
        # Preserve the bind-mounted inode. Guests stay frozen throughout this write.
        fd = os.open(self.ca_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "wb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise DockerEgressReconnectError("state_unavailable")
            os.fchmod(stream.fileno(), 0o644)  # Public CA, readable by the unprivileged browser.
            stream.truncate(0)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

    def close(self) -> None:
        if self.closed:
            return
        import fcntl

        self.require_owned()
        if self.journal.get("pending_mutation"):
            raise DockerEgressReconnectError("ownership_uncertain")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.closed = True


# Freeze existing processes without stopping the new docker-exec transport used
# by preflight. PID/start-time pairs prevent PID reuse from resuming another task.
# PID 1 must be the adapter-validated inert sleep process. Re-running after process
# loss converges, including a crash before publication of the guest freeze file.
_FREEZE = r"""
import json, os, signal, time
from pathlib import Path
root=Path('/run/cayu'); root.mkdir(mode=0o700, exist_ok=True)
marker=root/'egress-frozen.json'
def info(pid):
    data=Path('/proc',str(pid),'stat').read_text(); tail=data[data.rfind(')')+2:].split()
    return tail[0], int(tail[1]), tail[19]
ancestors={1,os.getpid()}; parent=os.getppid()
while parent > 1 and parent not in ancestors:
    ancestors.add(parent)
    try: parent=info(parent)[1]
    except (OSError,ValueError): break
frozen={}
for attempt in range(100):
    stable=True
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors: continue
        pid=int(entry.name)
        try:
            state,_,start=info(pid)
            if state in ('Z','X'): continue
            frozen[str(pid)]=start
            if state not in ('T','t'):
                os.kill(pid,signal.SIGSTOP); stable=False
        except ProcessLookupError: pass
        except FileNotFoundError: pass
    if stable: break
    time.sleep(.02)
else: raise RuntimeError('guest freeze did not settle')
tmp=marker.with_suffix('.tmp')
with tmp.open('w') as out:
    json.dump(frozen,out); out.flush(); os.fsync(out.fileno())
os.replace(tmp,marker)
"""
_THAW = r"""
import json, os, signal
from pathlib import Path
marker=Path('/run/cayu/egress-frozen.json')
if marker.exists():
    for pid,start in json.loads(marker.read_text()).items():
        try:
            data=Path('/proc',pid,'stat').read_text(); tail=data[data.rfind(')')+2:].split()
            if tail[19] == start and tail[0] in ('T','t'): os.kill(int(pid),signal.SIGCONT)
        except (ProcessLookupError,FileNotFoundError): pass
    marker.unlink()
"""


class _OwnedDockerRunner(DockerRunner):
    _owner: _Claim | None = None
    _manager: DockerReconnect | None = None
    _preflight = False
    _frozen = False
    _admission_complete = False

    def _ensure_exec_open(self) -> None:
        super()._ensure_exec_open()
        if self._owner is not None:
            self._owner.require_owned()

    async def _remove_container(self) -> None:
        if self._owner is not None:
            self._owner.require_owned()
        await super()._remove_container()

    async def _stop_container(self) -> None:
        if self._owner is not None:
            self._owner.require_owned()
        await super()._stop_container()

    async def _exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        self._ensure_exec_open()
        if self._frozen and not self._preflight and self._admission_complete:
            if self._manager is None or self._owner is None:
                raise DockerEgressReconnectError("ownership_uncertain")
            await self._manager.activate(self._owner)
            self._frozen = False
        return await super()._exec(command, **kwargs)


class DockerReconnect:
    def __init__(self, adapter: DockerEgressAdapter, root: Path, timeout_s: float) -> None:
        if os.name != "posix" or (
            os.environ.get("DOCKER_HOST") and not os.environ["DOCKER_HOST"].startswith("unix://")
        ):
            raise DockerEgressReconnectError("unsupported_host")
        self.adapter = adapter
        self.root = root.absolute()
        self.timeout_s = timeout_s
        self.bindings: dict[int, _Claim] = {}
        self.claims: set[_Claim] = set()
        self.pending_commands: set[asyncio.Task[tuple[int, str]]] = set()
        self.activation_lock = asyncio.Lock()

    @property
    def configuration(self) -> str:
        return _digest(
            [
                str(self.root),
                self.adapter._sidecar_image,
                (
                    None
                    if self.adapter._seccomp_profile is None
                    else hashlib.sha256(
                        Path(self.adapter._seccomp_profile).read_bytes()
                    ).hexdigest()
                ),
                self.adapter._docker_cli_env_allowlist,
                self.adapter._proxy_host,
                *(
                    []
                    if self.adapter._control_server_container_id is None
                    else [self.adapter._control_server_container_id]
                ),
            ]
        )

    def claim(self, token: str) -> _Claim:
        self.claims = {claim for claim in self.claims if not claim.closed}
        if not _TOKEN.fullmatch(token):
            raise InvalidEgressReconnectMetadataError(
                "Invalid Docker reconnect ownership identity."
            )
        try:
            import fcntl
        except ImportError:
            raise DockerEgressReconnectError("unsupported_host") from None
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = self.root.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise DockerEgressReconnectError("state_unavailable")
            fd = os.open(
                self.root / f"{token}.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
            handle = os.fdopen(fd, "r+b", buffering=0)
            try:
                info = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                ):
                    raise DockerEgressReconnectError("state_unavailable")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise DockerEgressReconnectError("ownership_conflict") from None
            except BaseException:
                handle.close()
                raise
        except OSError:
            raise DockerEgressReconnectError("state_unavailable") from None
        claim = _Claim(self.root, token, handle)
        self.claims.add(claim)
        return claim

    async def validate_host(self) -> None:
        # Docker contexts can select SSH/TCP even when DOCKER_HOST is unset.
        endpoint = os.environ.get("DOCKER_HOST") if not os.environ.get("DOCKER_CONTEXT") else None
        if endpoint is None:
            try:
                async with asyncio.timeout(self.timeout_s):
                    code, output = await self.adapter._docker_run(
                        ["context", "inspect", "--format", "{{.Endpoints.docker.Host}}"]
                    )
            except (TimeoutError, OSError):
                raise DockerEgressReconnectError("daemon_unavailable") from None
            if code:
                raise DockerEgressReconnectError("daemon_unavailable")
            endpoint = output.strip()
        if not endpoint.startswith("unix:///"):
            raise DockerEgressReconnectError("unsupported_host")

    async def lookup(self, args: Sequence[str]) -> tuple[int, str]:
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self.adapter._docker_run(args)
        except (TimeoutError, OSError):
            raise DockerEgressReconnectError("daemon_unavailable") from None

    async def inspect(
        self, kind: str, reference: str, *, absent_ok: bool = False
    ) -> dict[str, Any] | None:
        code, output = await self.lookup(["inspect", "--type", kind, reference])
        if code:
            # Separate confirmed absence from daemon/transport errors using a successful list.
            list_code, identifiers = await self.lookup(
                ["network", "ls", "--no-trunc", "--format", "{{.ID}}"]
                if kind == "network"
                else ["ps", "-a", "--no-trunc", "--format", "{{.ID}} {{.Names}}"]
            )
            if list_code or reference in identifiers:
                raise DockerEgressReconnectError("daemon_unavailable")
            if absent_ok:
                return None
            raise DockerEgressReconnectError("allocation_absent")
        try:
            items = json.loads(output)
            if type(items) is not list or len(items) != 1 or type(items[0]) is not dict:
                raise ValueError
            return items[0]
        except (ValueError, TypeError):
            raise DockerEgressReconnectError("identity_mismatch") from None

    async def run(self, args: Sequence[str]) -> None:
        """Persist dispatch uncertainty before any daemon mutation.

        A timeout never releases a claim while its command may still mutate Docker.
        A new process treats that journal as uncertain and fences the exact container.
        """
        claim = next(
            (
                item
                for item in self.claims
                if not item.closed
                and any(
                    value and (value in args or f"{OWNER_LABEL}={value}" in args)
                    for value in (
                        item.token,
                        item.journal.get("sidecar_id"),
                        item.journal.get("sidecar_name"),
                        item.journal.get("network_id"),
                        item.journal.get("network_name"),
                        item.journal.get("identity", {}).get("container_id"),
                    )
                )
            ),
            None,
        )
        if claim is None:
            raise DockerEgressReconnectError("ownership_uncertain")
        claim.require_owned()
        if claim.journal.get("pending_mutation"):
            raise DockerEgressReconnectError("ownership_uncertain")
        claim.write(pending_mutation=True)

        async def dispatch() -> tuple[int, str]:
            return await self.adapter._docker_exec(args)

        task = asyncio.create_task(dispatch())
        self.pending_commands.add(task)

        # Retain the task through cancellation; never silently orphan daemon work.
        def settled(done: asyncio.Task[tuple[int, str]]) -> None:
            self.pending_commands.discard(done)
            with contextlib.suppress(BaseException):
                done.result()

        task.add_done_callback(settled)
        try:
            done, _pending = await asyncio.wait({task}, timeout=self.timeout_s)
            if not done:
                raise DockerEgressReconnectError("ownership_uncertain")
            code, _diagnostic = task.result()
        except asyncio.CancelledError:
            raise
        except BaseException:
            # The persistent marker intentionally survives even a late success.
            raise DockerEgressReconnectError("ownership_uncertain") from None
        claim.write(pending_mutation=False)
        if code:
            raise DockerEgressReconnectError("daemon_unavailable")

    async def freeze(self, claim: _Claim) -> None:
        claim.require_owned()
        identity = claim.journal.get("identity")
        if identity is None:
            return
        try:
            await self.run(
                ["exec", "-u", "root", identity["container_id"], "python3", "-c", _FREEZE]
            )
        except Exception:
            raise DockerEgressReconnectError("fencing_failed") from None
        if claim.runner is not None:
            claim.runner._frozen = True

    async def activate(self, claim: _Claim) -> None:
        async with self.activation_lock:
            claim.require_owned()
            if claim.journal["state"] != "active":
                raise DockerEgressReconnectError("ownership_uncertain")
            await self.run(
                [
                    "exec",
                    "-u",
                    "root",
                    claim.journal["identity"]["container_id"],
                    "python3",
                    "-c",
                    _THAW,
                ]
            )

    def container_configuration(self, inspected: dict[str, Any]) -> str:
        config, host = inspected["Config"], inspected["HostConfig"]
        if config.get("Entrypoint") or config.get("Cmd") != ["sleep", "infinity"]:
            raise DockerEgressReconnectError("configuration_mismatch")
        if host.get("Privileged") or host.get("NetworkMode") in {"host", "bridge", "default"}:
            raise DockerEgressReconnectError("configuration_mismatch")
        return _digest(
            [
                inspected["Image"],
                config.get("User"),
                config.get("Entrypoint"),
                config.get("Cmd"),
                host,
                sorted(inspected.get("Mounts", []), key=lambda mount: mount["Destination"]),
            ]
        )

    async def validate_allocation(self, claim: _Claim, identity: dict[str, Any]) -> None:
        claim.require_owned()
        container = await self.inspect("container", identity["container_id"])
        named = await self.inspect("container", identity["container_name"])
        assert container is not None and named is not None
        if (
            named.get("Id") != identity["container_id"]
            or container.get("Id") != identity["container_id"]
        ):
            raise DockerEgressReconnectError("identity_mismatch")
        if not container.get("State", {}).get("Running") or container.get("State", {}).get(
            "Paused"
        ):
            raise DockerEgressReconnectError("fencing_failed")
        if (
            container.get("Image") != identity["image_id"]
            or self.container_configuration(container) != identity["container_configuration"]
        ):
            raise DockerEgressReconnectError("configuration_mismatch")
        networks = container.get("NetworkSettings", {}).get("Networks", {})
        if (
            len(networks) != 1
            or next(iter(networks.values())).get("NetworkID") != identity["network_id"]
        ):
            raise DockerEgressReconnectError("identity_mismatch")
        network = await self.inspect("network", identity["network_id"])
        assert network is not None
        if (
            network.get("Id") != identity["network_id"]
            or network.get("Internal") is not True
            or network.get("Labels", {}).get(OWNER_LABEL) != claim.token
        ):
            raise DockerEgressReconnectError("identity_mismatch")

    async def control_attachment(self, claim: _Claim, network: dict[str, Any]) -> bool:
        """Read both exact endpoints; never adopt a container by name or alias."""
        claim.require_owned()
        control_id = claim.journal.get("control_server_container_id")
        if control_id != self.adapter._control_server_container_id:
            raise DockerEgressReconnectError("configuration_mismatch")
        if control_id is None:
            return False
        if (
            network.get("Id") != claim.journal.get("network_id")
            or network.get("Labels", {}).get(OWNER_LABEL) != claim.token
            or network.get("Internal") is not True
        ):
            raise DockerEgressReconnectError("identity_mismatch")
        control = await self.inspect("container", control_id, absent_ok=True)
        if (
            control is None
            or control.get("Id") != control_id
            or not control.get("State", {}).get("Running")
            or control.get("State", {}).get("Paused", False)
        ):
            raise DockerEgressReconnectError("control_server_unavailable")
        members = network.get("Containers")
        if type(members) is not dict:
            raise DockerEgressReconnectError("identity_mismatch")
        endpoints = control.get("NetworkSettings", {}).get("Networks", {})
        matches = [value for value in endpoints.values() if value.get("NetworkID") == network["Id"]]
        if control_id not in members:
            if matches:
                raise DockerEgressReconnectError("identity_mismatch")
            return False
        if (
            len(matches) != 1
            or not matches[0].get("EndpointID")
            or matches[0]["EndpointID"] != members[control_id].get("EndpointID")
            or "cayu-control" not in (matches[0].get("Aliases") or [])
        ):
            raise DockerEgressReconnectError("control_server_alias_conflict")
        return True

    async def attach_control_server(self, token: str, reference: str) -> None:
        claim = next(
            (item for item in self.claims if item.token == token and not item.closed), None
        )
        if claim is None:
            raise DockerEgressReconnectError("ownership_uncertain")
        claim.require_owned()
        control_id = self.adapter._control_server_container_id
        if control_id is None:
            raise DockerEgressReconnectError("configuration_mismatch")
        network = await self.inspect("network", reference)
        assert network is not None
        if network.get("Labels", {}).get(OWNER_LABEL) != token or not _ID.fullmatch(
            network.get("Id", "")
        ):
            raise DockerEgressReconnectError("identity_mismatch")
        if claim.journal.get("network_id", network["Id"]) != network["Id"]:
            raise DockerEgressReconnectError("identity_mismatch")
        claim.write(network_id=network["Id"])
        # Only allocation-owned guests/sidecars may share this private network.
        # Checking all other endpoint aliases also rejects a stolen control alias.
        for member in network.get("Containers", {}):
            if member == self.adapter._control_server_container_id:
                continue
            inspected = await self.inspect("container", member)
            assert inspected is not None
            if (
                member != claim.journal.get("identity", {}).get("container_id")
                and inspected.get("Config", {}).get("Labels", {}).get(OWNER_LABEL) != token
            ):
                raise DockerEgressReconnectError("identity_mismatch")
            for endpoint in inspected.get("NetworkSettings", {}).get("Networks", {}).values():
                if endpoint.get("NetworkID") == network["Id"] and "cayu-control" in (
                    endpoint.get("Aliases") or []
                ):
                    raise DockerEgressReconnectError("control_server_alias_conflict")
        if await self.control_attachment(claim, network):
            return
        await self.run(
            [
                "network",
                "connect",
                "--alias",
                "cayu-control",
                network["Id"],
                control_id,
            ]
        )
        network = await self.inspect("network", network["Id"])
        assert network is not None
        if not await self.control_attachment(claim, network):
            raise DockerEgressReconnectError("identity_mismatch")

    async def detach_control_server(self, claim: _Claim) -> None:
        claim.require_owned()
        control_id = claim.journal.get("control_server_container_id")
        if control_id != self.adapter._control_server_container_id:
            raise DockerEgressReconnectError("configuration_mismatch")
        if control_id is None:
            return
        reference = claim.journal.get("network_id") or claim.journal.get("network_name")
        if reference is None:
            return
        network = await self.inspect("network", reference, absent_ok=True)
        if network is None:
            return
        if (
            network.get("Id") != claim.journal.get("network_id")
            or network.get("Labels", {}).get(OWNER_LABEL) != claim.token
        ):
            raise DockerEgressReconnectError("identity_mismatch")
        members = network.get("Containers")
        if type(members) is not dict:
            raise DockerEgressReconnectError("identity_mismatch")
        if control_id not in members:
            return
        await self.run(["network", "disconnect", "--force", network["Id"], control_id])
        network = await self.inspect("network", network["Id"])
        if network is None or control_id in network.get("Containers", {control_id: {}}):
            raise DockerEgressReconnectError("identity_mismatch")

    async def remove_sidecar(self, claim: _Claim) -> None:
        claim.require_owned()
        reference = claim.journal.get("sidecar_id") or claim.journal.get("sidecar_name")
        if reference is None:
            return
        sidecar = await self.inspect("container", reference, absent_ok=True)
        if sidecar is not None:
            if sidecar.get("Config", {}).get("Labels", {}).get(
                OWNER_LABEL
            ) != claim.token or not _ID.fullmatch(sidecar.get("Id", "")):
                raise DockerEgressReconnectError("identity_mismatch")
            await self.run(["rm", "-f", sidecar["Id"]])
        claim.write(sidecar_id=None, sidecar_name=None)

    async def is_allocation_disposed(self, reconnect_metadata: Mapping[str, Any]) -> bool:
        identity = validate_identity(reconnect_metadata)
        await self.validate_host()
        claim = self.claim(identity["allocation_id"])
        try:
            claim.read()
            if claim.journal.get("identity") != identity:
                raise DockerEgressReconnectError("identity_mismatch")
            # A terminal journal is immutable. Do not require the new worker's
            # configuration to equal the retired one: profile admission may
            # have explicitly authorized a new Runtime or browser version.
            # All live-state configuration checks remain in prepare().
            return claim.journal["state"] == "disposed" and not claim.journal.get(
                "pending_mutation"
            )
        finally:
            # This observer dispatched no provider mutation. Release only its
            # file lock, even if a preceding worker left pending_mutation in
            # the journal. Keep that durable uncertainty unchanged; normal
            # mutation owners still use close() and its settlement guard.
            claim.handle.close()
            claim.closed = True

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
        environment_name: str | None = None,
        reconnect_metadata: Mapping[str, Any] | None = None,
    ) -> EgressBinding:
        identity = None if reconnect_metadata is None else validate_identity(reconnect_metadata)
        if identity is not None and (
            identity["session_id"] != session_id or identity["environment_name"] != environment_name
        ):
            raise InvalidEgressReconnectMetadataError("Docker reconnect scope does not match.")
        if identity is not None and identity["configuration"] != self.configuration:
            raise DockerEgressReconnectError("configuration_mismatch")
        await self.validate_host()
        claim = self.claim(secrets.token_hex(16) if identity is None else identity["allocation_id"])
        binding = None
        try:
            if identity is not None:
                claim.read()
                if claim.journal.get("identity") != identity:
                    raise DockerEgressReconnectError("identity_mismatch")
                if claim.journal["state"] == "disposed":
                    raise DockerEgressReconnectError("disposed")
                if claim.journal["state"] == "disposal_pending":
                    await self.dispose(claim)
                    raise DockerEgressReconnectError("disposed")
                if claim.journal["state"] == "ownership_uncertain":
                    raise DockerEgressReconnectError("ownership_uncertain")
                if claim.journal.get("pending_mutation"):
                    container = await self.inspect("container", identity["container_id"])
                    if container is None or container.get("Id") != identity["container_id"]:
                        raise DockerEgressReconnectError("identity_mismatch")
                    if not container.get("State", {}).get("Paused"):
                        async with asyncio.timeout(self.timeout_s):
                            code, _ = await self.adapter._docker_exec(
                                ["pause", identity["container_id"]]
                            )
                        if code:
                            raise DockerEgressReconnectError("ownership_uncertain")
                    claim.write(state="ownership_uncertain", pending_mutation=False)
                    raise DockerEgressReconnectError("ownership_uncertain")
                await self.validate_allocation(claim, identity)
                claim.write(state="recovering")
                await self.freeze(claim)
                await self.remove_sidecar(claim)
            else:
                claim.write(
                    state="creating",
                    session_id=session_id,
                    network_name=f"cayu-egress-net-{claim.token}",
                    control_server_container_id=self.adapter._control_server_container_id,
                )
            sidecar_name = f"cayu-egress-{secrets.token_hex(12)}"
            claim.write(sidecar_name=sidecar_name, sidecar_id=None)
            binding = await self.adapter._prepare(
                session_id=session_id,
                grants=grants,
                broker=broker,
                reconnect_network=None if identity is None else identity["network_id"],
                reconnect_token=claim.token,
                reconnect_sidecar=sidecar_name,
            )
            sidecar = await self.inspect("container", sidecar_name)
            network = await self.inspect("network", binding.network or "")
            assert sidecar is not None and network is not None
            expected_sidecar_image = claim.journal.get("sidecar_image_id")
            claim.write(sidecar_id=sidecar["Id"], network_id=network["Id"])
            if (
                expected_sidecar_image is not None
                and sidecar.get("Image") != expected_sidecar_image
            ):
                raise DockerEgressReconnectError("configuration_mismatch")
            claim.write(sidecar_image_id=sidecar.get("Image"))
            # Teardown is exact-ID scoped even if a familiar name is reused later.
            binding.sidecar = sidecar["Id"]
            binding.network = network["Id"]
            claim.install_ca(binding.ca_cert_pem or b"")
            self.bindings[id(binding)] = claim
            original_close = binding.teardown

            async def close() -> None:
                claim.require_owned()
                if claim.runner is not None and not claim.runner.is_closed:
                    await self.finalize(claim.runner, outcome="interrupted")
                if original_close is not None:
                    await original_close()
                await self.remove_sidecar(claim)
                await self.detach_control_server(claim)
                if claim.journal["state"] in {"disposal_pending", "disposed", "creating"}:
                    await self.remove_network(claim)
                    with contextlib.suppress(FileNotFoundError):
                        claim.ca_path.unlink()
                    claim.write(state="disposed")
                claim.close()
                self.bindings.pop(id(binding), None)

            binding.teardown = close
            return binding
        except BaseException as failure:
            if binding is not None:
                await binding.close()
            if not claim.closed:
                # Only mutate records this attempt owns and has durably opened.
                if claim.journal.get("state") in {"creating", "recovering"}:
                    await self.remove_sidecar(claim)
                    if claim.journal["state"] == "creating":
                        await self.remove_network(claim)
                claim.close()
            if isinstance(failure, OSError):
                raise DockerEgressReconnectError(
                    "listener_conflict"
                    if failure.errno == errno.EADDRINUSE
                    else "state_unavailable"
                ) from None
            raise

    async def create_runner(self, request: VirtualEgressRunnerRequest) -> DockerRunner:
        claim = self.bindings.get(id(request.binding))
        if claim is None or request.session_id is None or request.environment_name is None:
            raise DockerEgressReconnectError("ownership_uncertain")
        claim.require_owned()
        if request.host_workspace_path is not None:
            workspace = Path(request.host_workspace_path).resolve()
            state_root = self.root.resolve()
            if state_root.is_relative_to(workspace) or workspace.is_relative_to(state_root):
                raise DockerEgressReconnectError("configuration_mismatch")
        identity = claim.journal.get("identity")
        runner_configuration = _digest(
            [
                request.runner_kind,
                request.image,
                request.setup_commands,
                request.host_workspace_path,
                request.guest_ca_path,
            ]
        )
        if identity is not None and identity["runner_configuration"] != runner_configuration:
            raise DockerEgressReconnectError("configuration_mismatch")
        if identity is None:
            runner = await _OwnedDockerRunner.create(
                request.name,
                image=request.image,
                close_action="remove",
                mount_path=request.host_workspace_path,
                credential_mode=CredentialMode.VIRTUAL_EGRESS,
                network=request.binding.network,
                env_overlay=dict(request.env_overlay),
                _env_overlay_secret_values_present=request.env_overlay_secret_values_present,
                ca_mount=(str(claim.ca_path), request.guest_ca_path),
                seccomp_profile=self.adapter._seccomp_profile,
                setup_commands=request.setup_commands,
                docker_cli_env_allowlist=self.adapter._docker_cli_env_allowlist,
            )
            runner = cast("_OwnedDockerRunner", runner)
            try:
                inspection = await self.inspect("container", runner.container_reference)
                assert inspection is not None
                identity = validate_identity(
                    {
                        "version": 1,
                        "backend": "docker",
                        "allocation_id": claim.token,
                        "container_id": inspection["Id"],
                        "container_name": runner.name,
                        "network_id": claim.journal["network_id"],
                        "session_id": request.session_id,
                        "environment_name": request.environment_name,
                        "image_id": inspection["Image"],
                        "configuration": self.configuration,
                        "container_configuration": self.container_configuration(inspection),
                        "runner_configuration": runner_configuration,
                    }
                )
                claim.write(identity=identity, state="recovering", image=request.image)
            except BaseException:
                # Creation returned an exact ID, even if validation/publication failed.
                await runner.close()
                raise

        else:
            if request.image != claim.journal.get("image"):
                raise DockerEgressReconnectError("configuration_mismatch")
            await self.validate_allocation(claim, identity)
            runner = _OwnedDockerRunner(
                identity["container_name"],
                _container_id=identity["container_id"],
                image=request.image,
                close_action="remove",
                credential_mode=CredentialMode.VIRTUAL_EGRESS,
                env_overlay=dict(request.env_overlay),
                _env_overlay_secret_values_present=request.env_overlay_secret_values_present,
                docker_cli_env_allowlist=self.adapter._docker_cli_env_allowlist,
            )
        runner._owner, runner._manager = claim, self
        claim.runner = runner
        runner._frozen = True
        runner._preflight = True
        try:
            await self.freeze(claim)
            try:
                await run_enforcement_preflight(
                    runner, request, timeout_s=max(1, int(self.timeout_s)), probe_metadata=False
                )
            except Exception:
                raise DockerEgressReconnectError("preflight_failed") from None
            await self.validate_allocation(claim, identity)
            claim.write(state="active")
        except BaseException:
            # No existing workload is thawed on a failed or cancelled admission.
            claim.write(state="recovering")
            runner.close_action = "none"
            await runner.close()
            raise
        finally:
            runner._preflight = False
        return runner

    def metadata(self, runner: DockerRunner) -> dict[str, Any]:
        if not isinstance(runner, _OwnedDockerRunner) or runner._owner is None:
            raise DockerEgressReconnectError("ownership_uncertain")
        runner._owner.require_owned()
        return validate_identity(runner._owner.journal["identity"])

    async def remove_network(self, claim: _Claim) -> None:
        claim.require_owned()
        await self.detach_control_server(claim)
        reference = claim.journal.get("network_id") or claim.journal.get("network_name")
        if reference is None:
            return
        network = await self.inspect("network", reference, absent_ok=True)
        if network is not None:
            if network.get("Labels", {}).get(OWNER_LABEL) != claim.token:
                raise DockerEgressReconnectError("identity_mismatch")
            await self.run(["network", "rm", network["Id"]])

    async def dispose(self, claim: _Claim) -> None:
        claim.require_owned()
        claim.write(state="disposal_pending")
        identity = claim.journal.get("identity")
        if identity is not None:
            existing = await self.inspect("container", identity["container_id"], absent_ok=True)
            if existing is not None:
                if existing.get("Id") != identity["container_id"]:
                    raise DockerEgressReconnectError("identity_mismatch")
                await self.run(["rm", "-f", identity["container_id"]])
        await self.remove_sidecar(claim)
        await self.remove_network(claim)
        with contextlib.suppress(FileNotFoundError):
            claim.ca_path.unlink()
        claim.write(state="disposed")

    async def finalize(
        self, runner: DockerRunner, *, outcome: str | None
    ) -> RunnerFinalizationResult:
        if not isinstance(runner, _OwnedDockerRunner) or runner._owner is None:
            raise DockerEgressReconnectError("ownership_uncertain")
        claim = runner._owner
        claim.require_owned()
        if outcome != "interrupted":
            await runner._finalize_browser_recordings(normal=outcome == "completed")
        if outcome == "interrupted":
            await self.freeze(claim)
            runner.close_action = "none"
            await runner.close()
            claim.write(state="retained")
            return RunnerFinalizationResult(
                workspace_mutations_quiescent=True, allocation_preserved=True
            )
        claim.write(state="disposal_pending")
        runner.close_action = "remove"
        await runner.close()
        # Binding teardown must settle authority, sidecar and network before
        # publishing disposed. Process loss here resumes exact-ID disposal.
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)
