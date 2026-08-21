"""Server-owned executable target registry for project Evals."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from cayu._validation import require_durable_clean_nonblank, require_unicode_scalar_text
from cayu.evals.execution import CorpusTarget, evaluation_target_identity
from cayu.evals.store import EvalStore
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import RunRequest
from cayu.server.config import (
    DEFAULT_EVAL_LEASE_SECONDS,
    DEFAULT_EVAL_POLL_INTERVAL_SECONDS,
    DEFAULT_EVAL_SHUTDOWN_GRACE_SECONDS,
    EvalsConfig,
)
from cayu.server.contracts import (
    MAX_EVAL_TARGET_COMPONENT_CHARS,
    MAX_EVAL_TARGETS,
    EvalTargetCatalogEntry,
    EvalTargetCatalogResponse,
)

DEFAULT_EVAL_PROFILE_ID = "default"
_EXPLICIT_EVAL_PROFILE_ID = "explicit"
_TARGET_KEY_DOMAIN = b"cayu-generated-eval-target-v1\0"


def _target_identity_component(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    require_unicode_scalar_text(value, field_name)
    if len(value) > MAX_EVAL_TARGET_COMPONENT_CHARS:
        raise ValueError(
            f"{field_name} cannot exceed {MAX_EVAL_TARGET_COMPONENT_CHARS} characters."
        )
    return value


def derive_eval_target_key(
    *,
    project_id: str,
    agent_name: str,
    profile_id: str = DEFAULT_EVAL_PROFILE_ID,
) -> str:
    """Derive a release-independent key from unambiguous logical identity bytes."""

    components = (
        _target_identity_component(project_id, "project_id"),
        _target_identity_component(agent_name, "agent_name"),
        _target_identity_component(profile_id, "profile_id"),
    )
    digest = hashlib.sha256()
    digest.update(_TARGET_KEY_DOMAIN)
    for component in components:
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return f"eval.{digest.hexdigest()}"


@dataclass(frozen=True, slots=True, repr=False)
class EvalTargetRegistration:
    """One public identity paired with process-local execution authority."""

    catalog_entry: EvalTargetCatalogEntry
    target: CorpusTarget
    manifest_project_root: Path | None = None

    def __post_init__(self) -> None:
        if type(self.catalog_entry) is not EvalTargetCatalogEntry:
            raise TypeError("catalog_entry must be an exact EvalTargetCatalogEntry.")
        if type(self.target) is not CorpusTarget:
            raise TypeError("target must be an exact CorpusTarget.")
        if self.catalog_entry.target_key != self.target.key:
            raise ValueError("Eval target catalog key does not match its runtime target.")
        if self.catalog_entry.agent_name != self.target.request_base.agent_name:
            raise ValueError("Eval target catalog agent does not match its request authority.")
        if self.catalog_entry.application_release_id != self.target.application_release_id:
            raise ValueError("Eval target catalog release does not match its runtime target.")
        if self.manifest_project_root is not None and (
            not isinstance(self.manifest_project_root, Path)
            or not self.manifest_project_root.is_absolute()
        ):
            raise TypeError("manifest_project_root must be an absolute Path or None.")


class EvalTargetRegistry:
    """Immutable bounded map from published keys to local runtime authority."""

    __slots__ = ("_catalog", "_registrations")

    def __init__(self, registrations: Iterable[EvalTargetRegistration]) -> None:
        if isinstance(registrations, str | bytes):
            raise TypeError("registrations must be an iterable of target registrations.")
        items = tuple(registrations)
        if not items:
            raise ValueError("An eval target registry cannot be empty.")
        if len(items) > MAX_EVAL_TARGETS:
            raise ValueError(f"An eval target registry cannot exceed {MAX_EVAL_TARGETS} targets.")
        if any(type(item) is not EvalTargetRegistration for item in items):
            raise TypeError("registrations must contain exact EvalTargetRegistration values.")

        ordered = tuple(
            sorted(
                items,
                key=lambda item: (
                    item.catalog_entry.agent_name,
                    item.catalog_entry.profile_id,
                    item.catalog_entry.target_key,
                ),
            )
        )
        keys = tuple(item.target.key for item in ordered)
        if len(keys) != len(set(keys)):
            raise ValueError("Eval target key collision detected while building the registry.")
        logical_identities = tuple(
            (
                item.catalog_entry.project_id,
                item.catalog_entry.agent_name,
                item.catalog_entry.profile_id,
            )
            for item in ordered
        )
        if len(logical_identities) != len(set(logical_identities)):
            raise ValueError("Eval target logical identities must be unique.")
        app = ordered[0].target.app
        if any(item.target.app is not app for item in ordered):
            raise ValueError("All eval registry targets must belong to one CayuApp instance.")
        manifest_project_root = ordered[0].manifest_project_root
        if any(item.manifest_project_root != manifest_project_root for item in ordered):
            raise ValueError("All eval registry targets must use one application-manifest root.")
        manifest_identity = evaluation_target_identity(
            ordered[0].target,
            project_root=manifest_project_root,
        )
        if any(
            item.catalog_entry.app_manifest_fingerprint
            != manifest_identity.app_manifest_fingerprint
            for item in ordered
        ):
            raise ValueError(
                "Eval target catalog manifest does not match its runtime provenance policy."
            )

        registrations_by_key = {item.target.key: item for item in ordered}
        self._registrations = MappingProxyType(registrations_by_key)
        self._catalog = EvalTargetCatalogResponse(
            items=tuple(item.catalog_entry for item in ordered),
            default_target_key=ordered[0].target.key,
        )

    def __repr__(self) -> str:
        return f"EvalTargetRegistry(target_count={len(self._registrations)})"

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(item.target_key for item in self._catalog.items)

    @property
    def default_target_key(self) -> str:
        return self._catalog.default_target_key

    def catalog(self) -> EvalTargetCatalogResponse:
        return self._catalog.model_copy(deep=True)

    def get(self, target_key: str) -> CorpusTarget | None:
        registration = self._registrations.get(target_key)
        return None if registration is None else registration.target

    def registration(self, target_key: str) -> EvalTargetRegistration | None:
        """Resolve executable authority and its manifest policy as one immutable value."""

        return self._registrations.get(target_key)


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedEvalsRuntime:
    """Validated store, registry, and worker policy used by server routes."""

    registry: EvalTargetRegistry
    store: EvalStore
    lease_seconds: int
    poll_interval_seconds: float
    shutdown_grace_seconds: float

    def __post_init__(self) -> None:
        if type(self.registry) is not EvalTargetRegistry:
            raise TypeError("registry must be an exact EvalTargetRegistry.")
        if not isinstance(self.store, EvalStore) or not self.store.durable:
            raise TypeError("store must be a durable EvalStore.")


def generated_eval_target_registry(
    app: CayuApp,
    *,
    project_id: str,
    application_release_id: str,
    app_manifest_fingerprint: str,
    app_manifest_project_root: Path | None = None,
) -> EvalTargetRegistry | None:
    """Build one normal-authority target per currently registered agent."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    project_id = _target_identity_component(project_id, "project_id")
    application_release_id = _target_identity_component(
        application_release_id,
        "application_release_id",
    )
    app_manifest_fingerprint = require_durable_clean_nonblank(
        app_manifest_fingerprint,
        "app_manifest_fingerprint",
    )
    agent_names = app.list_agents()
    if not agent_names:
        return None
    if len(agent_names) > MAX_EVAL_TARGETS:
        raise ValueError(f"Automatic Evals supports at most {MAX_EVAL_TARGETS} registered agents.")

    registrations: list[EvalTargetRegistration] = []
    for agent_name in agent_names:
        agent_name = _target_identity_component(agent_name, "agent_name")
        target_key = derive_eval_target_key(
            project_id=project_id,
            agent_name=agent_name,
            profile_id=DEFAULT_EVAL_PROFILE_ID,
        )
        target = CorpusTarget(
            key=target_key,
            app=app,
            request_base=RunRequest(agent_name=agent_name, messages=[]),
            application_release_id=application_release_id,
        )
        entry = EvalTargetCatalogEntry(
            target_key=target_key,
            project_id=project_id,
            agent_name=agent_name,
            profile_id=DEFAULT_EVAL_PROFILE_ID,
            label=f"{agent_name} · Default",
            source="generated",
            application_release_id=application_release_id,
            app_manifest_fingerprint=app_manifest_fingerprint,
        )
        _require_public_catalog_entry(app, entry)
        registrations.append(
            EvalTargetRegistration(
                catalog_entry=entry,
                target=target,
                manifest_project_root=app_manifest_project_root,
            )
        )
    return EvalTargetRegistry(registrations)


def explicit_eval_target_registry(target: CorpusTarget) -> EvalTargetRegistry:
    """Adapt the V1 singleton target into the common registry contract."""

    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    identity = evaluation_target_identity(target)
    agent_name = _target_identity_component(target.request_base.agent_name, "agent_name")
    entry = EvalTargetCatalogEntry(
        target_key=target.key,
        project_id=None,
        agent_name=agent_name,
        profile_id=_EXPLICIT_EVAL_PROFILE_ID,
        label=f"{agent_name} · Explicit",
        source="explicit",
        application_release_id=identity.application_release_id,
        app_manifest_fingerprint=identity.app_manifest.fingerprint,
    )
    _require_public_catalog_entry(target.app, entry)
    return EvalTargetRegistry((EvalTargetRegistration(catalog_entry=entry, target=target),))


def resolved_evals_runtime(
    *,
    explicit: EvalsConfig | None,
    registry: EvalTargetRegistry | None,
    automatic_store: EvalStore | None,
) -> ResolvedEvalsRuntime | None:
    """Apply indivisible explicit-V1 precedence to the generated project plan."""

    if explicit is not None:
        if type(explicit) is not EvalsConfig:
            raise TypeError("explicit must be an exact EvalsConfig or None.")
        return ResolvedEvalsRuntime(
            registry=explicit_eval_target_registry(explicit.target),
            store=explicit.store,
            lease_seconds=explicit.lease_seconds,
            poll_interval_seconds=explicit.poll_interval_seconds,
            shutdown_grace_seconds=explicit.shutdown_grace_seconds,
        )
    if registry is None or automatic_store is None:
        return None
    return ResolvedEvalsRuntime(
        registry=registry,
        store=automatic_store,
        lease_seconds=DEFAULT_EVAL_LEASE_SECONDS,
        poll_interval_seconds=DEFAULT_EVAL_POLL_INTERVAL_SECONDS,
        shutdown_grace_seconds=DEFAULT_EVAL_SHUTDOWN_GRACE_SECONDS,
    )


def _require_public_catalog_entry(app: CayuApp, entry: EvalTargetCatalogEntry) -> None:
    public = entry.model_dump(mode="json")
    try:
        redacted = app.redact_json(public)
    except Exception as exc:
        raise ValueError(
            "Eval target identity could not cross the application redaction boundary."
        ) from exc
    if redacted != public:
        raise ValueError("Eval target identity contains a workload secret.")


__all__ = [
    "DEFAULT_EVAL_PROFILE_ID",
    "EvalTargetRegistry",
    "ResolvedEvalsRuntime",
    "derive_eval_target_key",
    "explicit_eval_target_registry",
    "generated_eval_target_registry",
    "resolved_evals_runtime",
]
