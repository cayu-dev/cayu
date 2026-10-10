"""Environment declarations and atomic artifact/session-closure registration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from itertools import islice
from types import MappingProxyType

from cayu._application_registration import _validate_environment_spec
from cayu._validation import copy_json_value, require_clean_nonblank, require_unicode_scalar_text
from cayu.artifacts import ArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec, copy_environment
from cayu.environments.bindings import copy_bound_workspace
from cayu.environments.factory import EnvironmentFactory
from cayu.knowledge.base import KnowledgeStore
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime.execution_identity import copy_execution_profile_behavior_identity
from cayu.runtime.session_closure import (
    ArtifactSessionClosureStore,
    KnowledgeSessionClosureStore,
    SessionClosureCoordinator,
    SessionClosureStore,
)
from cayu.sessions.base import SessionStore
from cayu.vaults.redaction import SecretRedactor


@dataclass(frozen=True, slots=True)
class _ArtifactStoreRegistration:
    store_id: str
    store: ArtifactStore
    fingerprint: str


class ApplicationEnvironmentRegistry:
    """Own declarations and their closure inventory without an application.

    Registration validates the prospective closure coordinator before publishing
    an environment, its artifact store or its default selection. Factories stay
    unmaterialized; runtime creation and resource cleanup belong to their callers.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        session_closure_stores: tuple[SessionClosureStore, ...] = (),
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        self._session_store = session_store
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._session_closure_external_stores = tuple(session_closure_stores)
        self._knowledge_store = knowledge_store
        self._environments: dict[str, runtime_records.RegisteredEnvironment] = {}
        self._artifact_store_registrations_by_id: dict[str, _ArtifactStoreRegistration] = {}
        self._default_environment_name: str | None = None
        self._session_closure = self._build_session_closure()

    @property
    def registrations(self) -> Mapping[str, runtime_records.RegisteredEnvironment]:
        """Read current declarations without exposing registry mutation."""
        return MappingProxyType(self._environments)

    @property
    def default_name(self) -> str | None:
        return self._default_environment_name

    @property
    def session_closure(self) -> SessionClosureCoordinator:
        return self._session_closure

    def _publish(
        self,
        registration: runtime_records.RegisteredEnvironment,
        artifact_store_registration: _ArtifactStoreRegistration | None,
        *,
        default: bool,
    ) -> None:
        closure = self._session_closure
        if artifact_store_registration is not None:
            closure = self._build_session_closure(
                artifact_store_registration=artifact_store_registration
            )
        self._environments[registration.spec.name] = registration
        if artifact_store_registration is not None:
            self._artifact_store_registrations_by_id[artifact_store_registration.store_id] = (
                artifact_store_registration
            )
            self._session_closure = closure
        if default:
            self._default_environment_name = registration.spec.name

    def register(
        self,
        environment: Environment,
        *,
        default: bool = False,
        registration_site: tuple[str | None, str | None] = (None, None),
    ) -> Environment:
        if not isinstance(environment, Environment):
            raise TypeError("Environment registration requires an Environment.")
        if not isinstance(default, bool):
            raise TypeError("Environment default flag must be a bool.")
        stored_environment = copy_environment(environment)
        stored_spec = _validate_environment_spec(
            stored_environment.spec,
            redactor=self._secret_redactor,
        )
        if stored_spec.name in self._environments:
            raise ValueError(f"Environment already registered: {stored_spec.name}")
        artifact_store = stored_environment.artifact_store
        artifact_store_registration = self._validate_artifact_store_registration(artifact_store)

        registration_source, registration_symbol = registration_site
        registered_environment = runtime_records.RegisteredEnvironment(
            spec=stored_spec,
            environment=stored_environment,
            runner_execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
                None
                if stored_environment.runner is None
                else stored_environment.runner.execution_profile_identity,
                redactor=self._secret_redactor,
                field_name="environment.runner.execution_profile_identity",
            ),
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        self._publish(registered_environment, artifact_store_registration, default=default)
        return environment

    def register_factory(
        self,
        spec: EnvironmentSpec,
        factory: EnvironmentFactory,
        *,
        artifact_store: ArtifactStore | None = None,
        default: bool = False,
        registration_site: tuple[str | None, str | None] = (None, None),
    ) -> EnvironmentFactory:
        if not isinstance(spec, EnvironmentSpec):
            raise TypeError("Environment factory registration requires an EnvironmentSpec.")
        if not isinstance(factory, EnvironmentFactory):
            raise TypeError("Environment factory registration requires an EnvironmentFactory.")
        if not isinstance(default, bool):
            raise TypeError("Environment factory default flag must be a bool.")
        stored_spec = _validate_environment_spec(
            spec,
            redactor=self._secret_redactor,
        )
        if stored_spec.name in self._environments:
            raise ValueError(f"Environment already registered: {stored_spec.name}")
        factory_secret_resolution_scope = factory.secret_resolution_scope
        if factory_secret_resolution_scope not in ("static", "dynamic"):
            raise ValueError(
                "Environment factory secret_resolution_scope must be static or dynamic."
            )
        stored_environment = Environment(stored_spec, artifact_store=artifact_store)
        artifact_store_registration = self._validate_artifact_store_registration(artifact_store)

        registration_source, registration_symbol = registration_site
        registered_environment = runtime_records.RegisteredEnvironment(
            spec=stored_spec,
            environment=stored_environment,
            factory=factory,
            factory_backed=True,
            factory_secret_resolution_scope=factory_secret_resolution_scope,
            factory_execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
                factory.execution_profile_identity,
                redactor=self._secret_redactor,
                field_name="environment_factory.execution_profile_identity",
            ),
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        self._publish(registered_environment, artifact_store_registration, default=default)
        return factory

    def registered_artifact_stores(self) -> tuple[ArtifactStore, ...]:
        """Every distinct artifact store registered with an environment."""

        return tuple(
            registration.store for registration in self._artifact_store_registrations_by_id.values()
        )

    def _validate_artifact_store_registration(
        self,
        artifact_store: ArtifactStore | None,
    ) -> _ArtifactStoreRegistration | None:
        if artifact_store is None:
            return None
        artifact_store_id = require_clean_nonblank(artifact_store.id, "artifact_store.id")
        artifact_store_id = require_unicode_scalar_text(
            artifact_store_id,
            "artifact_store.id",
        )
        registered = self._artifact_store_registrations_by_id.get(artifact_store_id)
        if registered is not None and registered.store is not artifact_store:
            raise ValueError(
                "Artifact store id already belongs to a different registered store: "
                f"{artifact_store_id}"
            )
        if registered is not None:
            return registered
        return _ArtifactStoreRegistration(
            store_id=artifact_store_id,
            store=artifact_store,
            fingerprint=f"sha256:{sha256(artifact_store_id.encode('utf-8')).hexdigest()}",
        )

    def _build_session_closure(
        self,
        *,
        artifact_store_registration: _ArtifactStoreRegistration | None = None,
    ) -> SessionClosureCoordinator:
        """Validate prospective closure inventory without publishing registration state."""
        registrations = dict(self._artifact_store_registrations_by_id)
        if artifact_store_registration is not None:
            registrations[artifact_store_registration.store_id] = artifact_store_registration
        stores = list(self._session_closure_external_stores)
        # Registrations already deduplicate raw artifact IDs. Closure adapters
        # use a separate, qualified namespace; never omit a registered store by
        # comparing its raw ID with an unrelated adapter's identity. The
        # coordinator rejects genuine duplicate adapter identities below.
        for registration in registrations.values():
            stores.append(ArtifactSessionClosureStore(registration.store))
        if self._knowledge_store is not None:
            # Inventory shared references before session artifacts are erased.
            first_artifact = next(
                (
                    index
                    for index, store in enumerate(stores)
                    if type(store) is ArtifactSessionClosureStore
                ),
                len(stores),
            )
            stores.insert(
                first_artifact,
                KnowledgeSessionClosureStore(
                    self._knowledge_store,
                    self._session_store,
                    tuple(store for store in stores if type(store) is ArtifactSessionClosureStore),
                ),
            )
        return SessionClosureCoordinator(
            self._session_store,
            dependent_stores=tuple(stores),
            clock=self._clock,
            secret_redactor=self._secret_redactor,
        )

    def names(self) -> tuple[str, ...]:
        """Return the names of all registered environments (concrete or factory), sorted."""
        return tuple(sorted(self._environments))

    def has_registered_artifact_store(self) -> bool:
        """Return whether any registered environment exposes artifact storage.

        The registration paths maintain this value, so the check is constant-time
        and does not copy registration metadata or materialize environment factories.
        """

        return bool(self._artifact_store_registrations_by_id)

    def artifact_store_registration_count(self) -> int:
        """Return the exact registration count without projecting store identities."""

        return len(self._artifact_store_registrations_by_id)

    def artifact_store_registration_fingerprints(
        self,
        *,
        limit: int,
    ) -> tuple[tuple[str, ...], int]:
        """Return a bounded snapshot of opaque store identities and the exact count.

        Fingerprints are fixed-size SHA-256 correlations of the store identities
        accepted at registration. They let protected diagnostics correlate shared
        registrations without returning a local path or application-defined id.
        """

        if type(limit) is not int:
            raise TypeError("Artifact store fingerprint limit must be an integer.")
        if limit < 1:
            raise ValueError("Artifact store fingerprint limit must be positive.")
        registrations = self._artifact_store_registrations_by_id
        fingerprints = tuple(
            registration.fingerprint for registration in islice(registrations.values(), limit)
        )
        return fingerprints, len(registrations)

    def list_registrations(self) -> tuple[runtime_records.RegisteredEnvironment, ...]:
        """Return registered environment metadata without materializing factories."""
        registrations: list[runtime_records.RegisteredEnvironment] = []
        for name in sorted(self._environments):
            registered_environment = self._environments[name]
            registrations.append(
                runtime_records.RegisteredEnvironment(
                    spec=registered_environment.spec.model_copy(deep=True),
                    environment=copy_environment(registered_environment.environment),
                    runner_execution_profile_identity=(
                        copy_execution_profile_behavior_identity(
                            registered_environment.runner_execution_profile_identity
                        )
                    ),
                    factory_execution_profile_identity=(
                        copy_execution_profile_behavior_identity(
                            registered_environment.factory_execution_profile_identity
                        )
                    ),
                    factory=registered_environment.factory,
                    factory_backed=registered_environment.factory_backed,
                    factory_secret_resolution_scope=registered_environment.factory_secret_resolution_scope,
                    bound_workspace=(
                        copy_bound_workspace(registered_environment.bound_workspace)
                        if registered_environment.bound_workspace is not None
                        else None
                    ),
                    binding_payload=copy_json_value(
                        registered_environment.binding_payload,
                        "binding_payload",
                    )
                    if registered_environment.binding_payload is not None
                    else None,
                    registration_source=registered_environment.registration_source,
                    registration_symbol=registered_environment.registration_symbol,
                    binding_generation_id=registered_environment.binding_generation_id,
                )
            )
        return tuple(registrations)

    def get_concrete(self, name: str | None = None) -> runtime_records.RegisteredEnvironment:
        registered_environment = self.get(name)
        if registered_environment is None:
            raise RuntimeError("No environment registered.")
        if registered_environment.factory is not None:
            raise RuntimeError(
                "Environment is factory-backed and is only concrete for a session: "
                f"{registered_environment.spec.name}"
            )
        return runtime_records.RegisteredEnvironment(
            spec=registered_environment.spec.model_copy(deep=True),
            environment=copy_environment(registered_environment.environment),
            runner_execution_profile_identity=copy_execution_profile_behavior_identity(
                registered_environment.runner_execution_profile_identity
            ),
            factory_execution_profile_identity=copy_execution_profile_behavior_identity(
                registered_environment.factory_execution_profile_identity
            ),
            binding_generation_id=registered_environment.binding_generation_id,
        )

    def get_factory(self, name: str | None = None) -> EnvironmentFactory:
        registered_environment = self.get(name)
        if registered_environment is None:
            raise RuntimeError("No environment registered.")
        if registered_environment.factory is None:
            raise RuntimeError(
                f"Environment is not factory-backed: {registered_environment.spec.name}"
            )
        return registered_environment.factory

    def get(
        self,
        name: str | None = None,
    ) -> runtime_records.RegisteredEnvironment | None:
        if name is not None:
            environment_name = require_clean_nonblank(name, "environment.name")
        else:
            environment_name = self._default_environment_name
        if environment_name is None:
            return None
        try:
            return self._environments[environment_name]
        except KeyError as exc:
            raise KeyError(f"Environment not registered: {environment_name}") from exc
