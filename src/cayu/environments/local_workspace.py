"""Request-local directories with reusable root selection and admission checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from cayu.environments.admission import ExecutionAdmissionCandidate
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import NativeBinding, WorkspaceBinding
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.proxies.base import CredentialProxy
from cayu.runners.base import Runner
from cayu.runners.local import LocalRunner
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.vaults.base import Vault
from cayu.workspaces.local import LocalWorkspace


def workspace_root_from_metadata(
    request: EnvironmentFactoryRequest, *, key: str = "workspace_root"
) -> Path:
    """Read a path value; this helper does not authorize workspace ownership."""
    value = request.metadata.get(key)
    if type(value) is not str or not value.strip():
        raise ValueError(f"Request metadata requires a non-empty {key!r} workspace root.")
    return Path(value)


def require_allowed_workspace_root(
    root: str | Path, *, allowed_roots: Sequence[str | Path]
) -> Path:
    """Resolve symlinks and require an existing directory below an allowed root.

    This is a path selection check for trusted local execution, not a sandbox or
    a defense against concurrent changes by a hostile filesystem owner.
    """
    return _require_allowed_root(root, _allowed_directories(allowed_roots))


def _allowed_directories(allowed_roots: Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(allowed_roots, str | Path) or not allowed_roots:
        raise ValueError("allowed_roots must contain at least one directory.")
    return tuple(Path(value).expanduser().resolve() for value in allowed_roots)


def _require_allowed_root(root: str | Path, allowed: tuple[Path, ...]) -> Path:
    if not isinstance(root, str | Path) or not str(root).strip():
        raise ValueError("Workspace root must be a non-empty absolute path.")
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Workspace root must be absolute: {path}")
    resolved = path.resolve()
    if not any(resolved.is_relative_to(parent) for parent in allowed):
        raise ValueError(f"Workspace root {resolved} is outside allowed directories: {allowed}")
    if not resolved.is_dir():
        raise ValueError(f"Workspace root is missing or is not a directory: {resolved}")
    return resolved


async def _release_external_attachment(action: EnvironmentFactoryReleaseAction) -> None:
    # This factory borrows runners; only their allocator may close them.
    del action


class LocalWorkspaceEnvironmentFactory(EnvironmentFactory):
    """Attach each request to its prepared local directory and runner.

    The root comes from ``metadata_key`` or a side-effect-free ``root_from_request``
    callback resolving trusted session ownership. Metadata selection requires
    ``authorize_root(request, resolved_root)`` to positively authorize the exact
    session/path pair; a parent-directory allowlist is not tenant authorization.
    Prepare directories before ``app.run``: pre-create admission cannot
    allocate them. ``runner_factory`` may attach a DockerRunner to an externally
    owned container already sharing this directory. It must be synchronous,
    side-effect free, and select the same runner identity at admission and create.
    Container allocation and disposal remain the caller's responsibility.
    """

    def __init__(
        self,
        *,
        allowed_roots: Sequence[str | Path],
        metadata_key: str = "workspace_root",
        root_from_request: Callable[[EnvironmentFactoryRequest], str | Path] | None = None,
        authorize_root: Callable[[EnvironmentFactoryRequest, Path], bool] | None = None,
        runner_factory: Callable[[Path, EnvironmentFactoryRequest], Runner] | None = None,
        binding_factory: Callable[[Path, EnvironmentFactoryRequest], WorkspaceBinding]
        | None = None,
        vault: Vault | None = None,
        proxy: CredentialProxy | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
    ) -> None:
        self._allowed_roots = _allowed_directories(allowed_roots)
        if type(metadata_key) is not str or not metadata_key.strip():
            raise ValueError("metadata_key must be a non-empty string.")
        if root_from_request is None and authorize_root is None:
            raise ValueError(
                "Metadata root selection requires authorize_root to verify session ownership; "
                "or supply root_from_request backed by trusted session ownership."
            )
        if root_from_request is not None and not callable(root_from_request):
            raise TypeError("root_from_request must be callable.")
        if authorize_root is not None and not callable(authorize_root):
            raise TypeError("authorize_root must be callable.")
        self._authorize_root = authorize_root
        self._metadata_key = metadata_key
        self._root_from_request = root_from_request
        self._runner_factory = runner_factory
        self._binding_factory = binding_factory
        self._vault = vault
        self._proxy = proxy
        self._execution_profile_identity = execution_profile_identity

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        return self._execution_profile_identity

    def resolve_root(self, request: EnvironmentFactoryRequest) -> Path:
        """Select and validate the root without changing the filesystem."""
        root = (
            workspace_root_from_metadata(request, key=self._metadata_key)
            if self._root_from_request is None
            else self._root_from_request(request)
        )
        resolved = _require_allowed_root(root, self._allowed_roots)
        if self._authorize_root is not None and self._authorize_root(request, resolved) is not True:
            raise ValueError("Workspace root is not authorized for this session.")
        return resolved

    def _runner(self, root: Path, request: EnvironmentFactoryRequest) -> Runner:
        runner = (
            LocalRunner(root)
            if self._runner_factory is None
            else self._runner_factory(root, request)
        )
        if not isinstance(runner, Runner):
            raise TypeError("runner_factory must return a Runner.")
        return runner

    def execution_admission_candidate(
        self, request: EnvironmentFactoryRequest
    ) -> ExecutionAdmissionCandidate | None:
        return self._runner(self.resolve_root(request), request).execution_admission_candidate_for(
            request.execution_requirements
        )

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        root = self.resolve_root(request)
        binding = (
            NativeBinding()
            if self._binding_factory is None
            else self._binding_factory(root, request)
        )
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(root, workspace_id=request.session_id),
                runner=self._runner(root, request),
                binding=binding,
                vault=self._vault,
                proxy=self._proxy,
            ),
            release=_release_external_attachment if self._runner_factory is not None else None,
        )
