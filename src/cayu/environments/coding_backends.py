"""Choose where a coding toolchain runs: Docker locally, Cayu Cloud's broker when deployed.

The parts compose and each works alone:

- :class:`CodingBackend` is one way to give a source workspace a coding sandbox
  that verifies a :class:`~cayu.environments.coding_toolchains.CodingToolchainProfile`:
  a factory per workspace, the admission candidate its runners report, and the
  execution requirements the agents that use it need.
- :class:`CodingBackendRegistry` maps backend names to builders. Docker
  (``docker``) and Cayu Cloud's sandbox broker (``cayu-cloud``) are built in;
  another provider registers its own builder under its own name.
- :func:`select_coding_backend` picks a backend from the process environment:
  the backend a deployment requires by name, otherwise Cayu Cloud when all
  three ``CAYU_CLOUD_*`` sandbox variables are set, Docker otherwise.
- :func:`register_coding_environment` registers one source workspace's
  factory with an app, under any name, as often as needed.

The individual factories stay public; an app that always runs on one backend
can construct it directly instead.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from cayu.environments.admission import ExecutionRequirements
from cayu.environments.base import EnvironmentSpec
from cayu.environments.coding_toolchains import (
    CodingToolchainProfile,
    coding_toolchain_evidence_policy,
    owned_coding_toolchain_profile,
)
from cayu.environments.factory import EnvironmentFactory
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.workspaces.local import LocalWorkspace

if TYPE_CHECKING:
    from cayu.artifacts.base import ArtifactStore
    from cayu.managed_settings import ManagedSetting

#: Names the one backend a deployment requires; startup fails if it can't be used.
CODING_BACKEND_ENV = "CAYU_CODING_BACKEND"
DOCKER_CODING_BACKEND = "docker"
CLOUD_CODING_BACKEND = "cayu-cloud"
#: What Cayu Cloud gives an Agent's processes where the install runs a sandbox broker.
CLOUD_SANDBOX_VARIABLES = (
    "CAYU_CLOUD_API_URL",
    "CAYU_CLOUD_APPLICATION_ID",
    "CAYU_CLOUD_SANDBOX_TOKEN",
)
_CLOUD_SANDBOX_TOKEN_VARIABLE = CLOUD_SANDBOX_VARIABLES[2]
_BACKEND_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class _FactoryBuilder(Protocol):
    def __call__(
        self,
        source_workspace: LocalWorkspace,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> EnvironmentFactory: ...


@dataclass(frozen=True)
class CodingBackend:
    """One way to run a coding toolchain, ready to give each source workspace a factory.

    ``candidate`` is the admission candidate the factory's runners report; its
    :class:`~cayu.environments.coding_toolchains.CodingToolchainEvidencePolicy`
    sets the evidence the agents' execution requirements ask for, so they match
    what the command tools accept. ``metadata`` is non-secret and lands in the
    registered environment's spec.
    """

    name: str
    candidate: str
    toolchain_profile: CodingToolchainProfile
    build_factory: _FactoryBuilder
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.name) is not str or _BACKEND_NAME.fullmatch(self.name) is None:
            raise ValueError("A coding backend name must be a portable lowercase identifier.")
        if coding_toolchain_evidence_policy(self.candidate) is None:
            raise ValueError(
                f"Admission candidate {self.candidate!r} has no coding toolchain evidence "
                "policy; register one with register_coding_toolchain_evidence_policy."
            )
        object.__setattr__(
            self, "toolchain_profile", owned_coding_toolchain_profile(self.toolchain_profile)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def factory_for(
        self,
        source_workspace: LocalWorkspace,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> EnvironmentFactory:
        """Return a new factory whose sessions copy in from ``source_workspace``."""

        if not isinstance(source_workspace, LocalWorkspace):
            raise TypeError("source_workspace must be a LocalWorkspace.")
        factory = self.build_factory(source_workspace, artifact_store=artifact_store)
        if not isinstance(factory, EnvironmentFactory):
            raise TypeError(f"The {self.name!r} coding backend returned no EnvironmentFactory.")
        return factory

    def execution_requirements(self, **overrides: object) -> ExecutionRequirements:
        """Trusted requirements for agents on this backend, at its candidate's evidence bar.

        Network access is denied by default and the profile's executables are
        required; ``overrides`` add or replace any other field. The result is
        always ``ExecutionRequirements.trusted``.
        """

        policy = coding_toolchain_evidence_policy(self.candidate)
        assert policy is not None  # checked at construction; policies are never removed
        values: dict[str, object] = {
            "network_access": "deny_by_default",
            "minimum_evidence": policy.minimum_evidence,
            "required_executables": self.toolchain_profile.required_executables,
        }
        values.update(overrides)
        return ExecutionRequirements.trusted(**values)


@dataclass(frozen=True)
class CodingBackendContext:
    """What a backend builder gets: the profile, the environment and its own options."""

    toolchain_profile: CodingToolchainProfile
    environ: Mapping[str, str]
    options: object | None = None


CodingBackendBuilder = Callable[[CodingBackendContext], CodingBackend]
#: Returns why the backend can't be used in this environment, or ``None`` when it can.
CodingBackendAvailability = Callable[[Mapping[str, str]], str | None]


@dataclass(frozen=True)
class _Registration:
    builder: CodingBackendBuilder
    available: CodingBackendAvailability | None


class CodingBackendRegistry:
    """Named coding backend builders. Each registry is independent; none is global."""

    def __init__(self) -> None:
        self._registrations: dict[str, _Registration] = {}

    def register(
        self,
        name: str,
        builder: CodingBackendBuilder,
        *,
        available: CodingBackendAvailability | None = None,
        replace: bool = False,
    ) -> None:
        """Add a backend. ``replace=True`` swaps one already registered, for example in tests."""

        if type(name) is not str or _BACKEND_NAME.fullmatch(name) is None:
            raise ValueError("A coding backend name must be a portable lowercase identifier.")
        if not callable(builder):
            raise TypeError("builder must be callable.")
        if available is not None and not callable(available):
            raise TypeError("available must be callable or None.")
        if name in self._registrations and not replace:
            raise ValueError(f"A coding backend named {name!r} is already registered.")
        self._registrations[name] = _Registration(builder, available)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def selected_name(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        required: str | None = None,
    ) -> str:
        """Return the backend this environment selects, or raise why it can't select one.

        ``required`` (or ``CAYU_CODING_BACKEND``) names a backend that must be
        registered and usable here, so a missing Cloud credential fails at
        startup instead of falling back to Docker. Otherwise Cayu Cloud is
        selected when all three ``CAYU_CLOUD_*`` sandbox variables are set,
        and Docker otherwise. The sandbox token is what signals that a broker
        is meant, so a token without the other two is an error; the API URL and
        application id alone are not, because Cloud's agent registry and the
        ``cayu cloud`` CLI set them too. Any other backend is selected by name
        only.
        """

        env = os.environ if environ is None else environ
        configured = (env.get(CODING_BACKEND_ENV) or "").strip() or None
        if required is not None and configured is not None and required != configured:
            raise ValueError(
                f"{CODING_BACKEND_ENV}={configured} conflicts with the required coding backend "
                f"{required!r}."
            )
        name = required or configured
        if name is not None:
            source = CODING_BACKEND_ENV if required is None else "This application"
            registration = self._registrations.get(name)
            if registration is None:
                raise ValueError(
                    f"{source} requires the {name!r} coding backend, which isn't registered "
                    f"(registered: {', '.join(self.names()) or 'none'})."
                )
            reason = None if registration.available is None else registration.available(env)
            if reason is not None:
                raise ValueError(f"{source} requires the {name!r} coding backend, but {reason}.")
            return name
        reason = cloud_sandbox_unavailable_reason(env)
        if reason is None:
            return CLOUD_CODING_BACKEND
        if (env.get(_CLOUD_SANDBOX_TOKEN_VARIABLE) or "").strip():
            raise ValueError(f"Cayu Cloud's sandbox variables are incomplete: {reason}.")
        return DOCKER_CODING_BACKEND

    def build(
        self,
        *,
        toolchain_profile: CodingToolchainProfile,
        options: Mapping[str, object] | None = None,
        environ: Mapping[str, str] | None = None,
        required: str | None = None,
    ) -> CodingBackend:
        """Build the selected backend; ``options`` are keyed by backend name."""

        env = os.environ if environ is None else environ
        owned_profile = owned_coding_toolchain_profile(toolchain_profile)
        name = self.selected_name(env, required=required)
        registration = self._registrations.get(name)
        if registration is None:
            raise ValueError(f"No coding backend is registered as {name!r}.")
        backend = registration.builder(
            CodingBackendContext(
                toolchain_profile=owned_profile,
                environ=env,
                options=None if options is None else options.get(name),
            )
        )
        if type(backend) is not CodingBackend or backend.name != name:
            raise TypeError(f"The {name!r} builder must return a CodingBackend named {name!r}.")
        if backend.toolchain_profile != owned_profile:
            raise ValueError(f"The {name!r} backend changed the toolchain profile it was given.")
        return backend


def cloud_sandbox_unavailable_reason(environ: Mapping[str, str]) -> str | None:
    """Why Cayu Cloud's broker can't be used here: the sandbox variables it lacks."""

    missing = [name for name in CLOUD_SANDBOX_VARIABLES if not (environ.get(name) or "").strip()]
    if not missing:
        return None
    return ", ".join(missing) + (" is unset" if len(missing) == 1 else " are unset")


def _docker_backend(context: CodingBackendContext) -> CodingBackend:
    from cayu.environments.docker_coding import docker_coding_backend

    return docker_coding_backend(context)


def _cloud_backend(context: CodingBackendContext) -> CodingBackend:
    # Imported only when selected, so Docker-only processes never load the broker client.
    from cayu.sandbox_broker.coding import brokered_coding_backend

    return brokered_coding_backend(context)


def default_coding_backends() -> CodingBackendRegistry:
    """Return a new registry with the built-in ``docker`` and ``cayu-cloud`` backends."""

    registry = CodingBackendRegistry()
    registry.register(DOCKER_CODING_BACKEND, _docker_backend)
    registry.register(
        CLOUD_CODING_BACKEND, _cloud_backend, available=cloud_sandbox_unavailable_reason
    )
    return registry


def select_coding_backend(
    *,
    toolchain_profile: CodingToolchainProfile,
    options: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
    required: str | None = None,
    registry: CodingBackendRegistry | None = None,
) -> CodingBackend:
    """Build the backend this process selects, from ``registry`` or the built-in ones."""

    selected_registry = default_coding_backends() if registry is None else registry
    if not isinstance(selected_registry, CodingBackendRegistry):
        raise TypeError("registry must be a CodingBackendRegistry.")
    return selected_registry.build(
        toolchain_profile=toolchain_profile,
        options=options,
        environ=environ,
        required=required,
    )


def register_coding_environment(
    app: Any,
    *,
    name: str,
    backend: CodingBackend,
    source_workspace: LocalWorkspace,
    artifact_store: ArtifactStore | None = None,
    default: bool = False,
    execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
    metadata: Mapping[str, object] | None = None,
    settings: Iterable[ManagedSetting] | None = None,
) -> EnvironmentFactory:
    """Register one source workspace's coding environment under ``name`` and return its factory.

    Call it once per workspace, for example once per clone that a parallel
    worker edits; ``RunRequest.environment_name`` then picks the workspace. The
    spec's metadata records the backend, the toolchain profile's evidence and
    the factory's identity, plus any non-secret ``metadata`` given here. ``settings``
    are the environment's managed settings, passed to ``register_environment_factory``.
    """

    if type(backend) is not CodingBackend:
        raise TypeError("backend must be a CodingBackend.")
    factory = backend.factory_for(source_workspace, artifact_store=artifact_store)
    factory_identity = factory.execution_profile_identity
    spec_metadata: dict[str, object] = {
        "coding_backend": backend.name,
        "execution_candidate": backend.candidate,
        "network": "none",
        **backend.metadata,
        **backend.toolchain_profile.evidence(),
        "factory_profile_identity": (
            None if factory_identity is None else factory_identity.model_dump(mode="json")
        ),
    }
    if metadata is not None:
        spec_metadata.update(metadata)
    app.register_environment_factory(
        EnvironmentSpec(
            name=name,
            metadata=spec_metadata,
            execution_profile_identity=execution_profile_identity,
        ),
        factory,
        artifact_store=artifact_store,
        default=default,
        settings=settings,
    )
    return factory


__all__ = [
    "CLOUD_CODING_BACKEND",
    "CLOUD_SANDBOX_VARIABLES",
    "CODING_BACKEND_ENV",
    "DOCKER_CODING_BACKEND",
    "CodingBackend",
    "CodingBackendAvailability",
    "CodingBackendBuilder",
    "CodingBackendContext",
    "CodingBackendRegistry",
    "cloud_sandbox_unavailable_reason",
    "default_coding_backends",
    "register_coding_environment",
    "select_coding_backend",
]
