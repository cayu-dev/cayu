"""Bounded session access policy, scope and shared authorization rules."""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu._resource_access_errors import ResourceAccessDenied as SessionAccessDenied
from cayu._validation import copy_label_map
from cayu.sessions.queries import LabelSelectorOperator, LabelSelectorRequirement
from cayu.sessions.records import Session
from cayu.sessions.requests import RunRequest

Action = Literal[
    "read", "create", "modify", "delete", "execute", "inspect_state", "update_labels", "relabel"
]


_ACTIONS = (
    "read",
    "create",
    "modify",
    "delete",
    "execute",
    "inspect_state",
    "update_labels",
    "relabel",
)


# Access predicates compile into store queries; the parameter bound keeps one
# action well inside SQLite's (32,766) and PostgreSQL's (65,535) limits while
# fitting tenant, team and project selectors for real organizations.
_MAX_SELECTOR_VALUES = 500


# Recipient provenance is a capability, never caller-shaped request data.
_RECIPIENT_PROVENANCE_CAPABILITY = object()

_MAX_RULE_SELECTORS = 64


_MAX_ACTION_RULES = 64


_MAX_ACTION_PARAMETERS = 2_048


# Matches the session label map ceiling: every label key can be protected.
_MAX_PROTECTED_LABEL_KEYS = 200


@dataclass(frozen=True, slots=True)
class SessionAccessSelector:
    """Immutable label requirement using the existing session-query semantics."""

    key: str
    operator: LabelSelectorOperator = LabelSelectorOperator.IN
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validated = LabelSelectorRequirement(
            key=self.key, operator=self.operator, values=self.values
        )
        if len(validated.values) > _MAX_SELECTOR_VALUES:
            raise ValueError(f"Access selectors accept at most {_MAX_SELECTOR_VALUES} values.")
        object.__setattr__(self, "key", validated.key)
        object.__setattr__(self, "operator", validated.operator)
        object.__setattr__(self, "values", validated.values)

    def matches(self, labels: Mapping[str, str]) -> bool:
        value = labels.get(self.key)
        if self.operator is LabelSelectorOperator.EXISTS:
            return value is not None
        if self.operator is LabelSelectorOperator.NOT_EXISTS:
            return value is None
        if self.operator is LabelSelectorOperator.IN:
            return value in self.values
        return value is None or value not in self.values


@dataclass(frozen=True, slots=True)
class SessionAccessRule:
    """Conjunction of selectors; unrestricted access must be explicit."""

    selectors: tuple[SessionAccessSelector, ...] = ()
    allow_all: bool = False

    def __post_init__(self) -> None:
        selectors = tuple(self.selectors)
        if type(self.allow_all) is not bool or len(selectors) > _MAX_RULE_SELECTORS:
            raise ValueError("Invalid access rule bounds.")
        if any(type(item) is not SessionAccessSelector for item in selectors):
            raise TypeError("Access rules require SessionAccessSelector values.")
        if bool(selectors) == self.allow_all:
            raise ValueError("Provide selectors or explicit allow_all=True, exclusively.")
        object.__setattr__(self, "selectors", selectors)

    def matches(self, labels: Mapping[str, str]) -> bool:
        return self.allow_all or all(item.matches(labels) for item in self.selectors)


@dataclass(frozen=True, slots=True)
class SessionAccessScope:
    """Application policy expressed as bounded alternatives for each action.

    Empty action grants deny. All keys used by any rule are protected from
    ordinary label updates, in addition to explicitly protected keys.
    """

    read: tuple[SessionAccessRule, ...] = ()
    create: tuple[SessionAccessRule, ...] = ()
    modify: tuple[SessionAccessRule, ...] = ()
    delete: tuple[SessionAccessRule, ...] = ()
    execute: tuple[SessionAccessRule, ...] = ()
    inspect_state: tuple[SessionAccessRule, ...] = ()
    update_labels: tuple[SessionAccessRule, ...] = ()
    relabel: tuple[SessionAccessRule, ...] = ()
    protected_label_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in _ACTIONS:
            rules = tuple(getattr(self, name))
            if len(rules) > _MAX_ACTION_RULES or any(
                type(rule) is not SessionAccessRule for rule in rules
            ):
                raise ValueError(
                    f"Each action accepts at most {_MAX_ACTION_RULES} SessionAccessRule alternatives."
                )
            if (
                sum(1 + len(selector.values) for rule in rules for selector in rule.selectors)
                > _MAX_ACTION_PARAMETERS
            ):
                raise ValueError(
                    f"Action predicates exceed the {_MAX_ACTION_PARAMETERS}-parameter bound."
                )
            object.__setattr__(self, name, rules)
        if isinstance(self.protected_label_keys, str | bytes):
            raise TypeError("Protected label keys must be a sequence of keys.")
        keys = tuple(self.protected_label_keys)
        if len(keys) > _MAX_PROTECTED_LABEL_KEYS:
            raise ValueError(
                f"At most {_MAX_PROTECTED_LABEL_KEYS} protected label keys are supported."
            )
        validated = copy_label_map(dict.fromkeys(keys, "_"), "protected_label_keys")
        object.__setattr__(self, "protected_label_keys", tuple(sorted(validated)))

    def matches(self, action: Action, labels: Mapping[str, str]) -> bool:
        return any(rule.matches(labels) for rule in getattr(self, action))

    def protected_keys(self) -> frozenset[str]:
        return frozenset(self.protected_label_keys) | frozenset(
            selector.key
            for action in (getattr(self, name) for name in _ACTIONS)
            for rule in action
            for selector in rule.selectors
        )


@dataclass(frozen=True, slots=True)
class _SessionAccessBounds:
    admitted: SessionAccessScope
    current: SessionAccessScope
    binding: ResourceExecutionBinding | None = None

    def matches(self, labels: Mapping[str, str], action: Action = "read") -> bool:
        return self.admitted.matches(action, labels) and self.current.matches(action, labels)

    def require_read(self, session: Session | None) -> Session:
        if session is None or not self.matches(session.labels):
            raise SessionAccessDenied()
        return session

    def require_action(self, session: Session | None, action: Action) -> Session:
        session = self.require_read(session)
        if not self.matches(session.labels, action):
            raise SessionAccessDenied()
        return session

    def require_creation(self, request: RunRequest, parent: Session | None) -> None:
        if not self.matches(request.labels, "create") or not self.matches(request.labels):
            raise SessionAccessDenied()
        if request.parent_session_id is not None:
            self.require_read(parent)

    def label_audit(self, session, labels, at):
        from hashlib import sha256
        from uuid import uuid4

        protected = self.admitted.protected_keys() | self.current.protected_keys()
        if not any(session.labels.get(key) != labels.get(key) for key in protected):
            return None
        identity = "cayu.resource_access.relabel:" + uuid4().hex
        return identity, {
            "schema_version": 1,
            "action": "relabel",
            "session_id": session.id,
            "session_instance_id": session.instance_id,
            "occurred_at": at.isoformat(),
            "old_labels": dict(session.labels),
            "new_labels": dict(labels),
            "authority": None if self.binding is None else self.binding.authority,
            "subject": None if self.binding is None else self.binding.subject,
            "admitted_sha256": None
            if self.binding is None
            else sha256(self.binding.admitted_json.encode()).hexdigest(),
        }

    def require_label_update(self, session: Session | None, labels: dict[str, str]) -> None:
        session = self.require_read(session)
        if not self.matches(session.labels, "update_labels") or not self.matches(
            labels, "update_labels"
        ):
            raise SessionAccessDenied()
        if not self.matches(labels):
            raise SessionAccessDenied()
        protected = self.admitted.protected_keys() | self.current.protected_keys()
        if any(session.labels.get(key) != labels.get(key) for key in protected) and (
            not self.matches(session.labels, "relabel") or not self.matches(labels, "relabel")
        ):
            raise SessionAccessDenied()


_creation_bounds: ContextVar[_SessionAccessBounds | None] = ContextVar(
    "session_access_creation", default=None
)


def _require_scoped_session_creation(request: RunRequest, parent: Session | None) -> None:
    from cayu.resource_access import current_creation_bounds

    bounds = _creation_bounds.get() or current_creation_bounds()
    if bounds is not None:
        bounds.require_creation(request, parent)
        from cayu.resource_access import current_binding

        binding = current_binding()
        if (
            binding is not None
            and parent is not None
            and parent.invocation.resource_access != binding
        ):
            raise SessionAccessDenied()


_query_bounds: ContextVar[_SessionAccessBounds | None] = ContextVar(
    "session_access_query", default=None
)


def runtime_session_query(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.resource_access import current_data_bounds

        bounds = await current_data_bounds()
        if (bounds is not None or _query_bounds.get() is not None) and (
            kwargs.get("additional_events") or kwargs.get("previous") is not None
        ):
            raise SessionAccessDenied()
        if bounds is None or _query_bounds.get() is not None:
            return await operation(self, *args, **kwargs)
        token = _query_bounds.set(bounds)
        try:
            return await operation(self, *args, **kwargs)
        finally:
            _query_bounds.reset(token)

    return guarded


def require_resource_session(session, action="read"):
    bounds = _query_bounds.get()
    if bounds is not None:
        bounds.require_action(session, action)


def runtime_session_mutation(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.resource_access import current_data_bounds

        bounds = await current_data_bounds()
        if bounds is not None:
            # Explicit private bounds cannot replace model execution authority.
            kwargs["_access_bounds"] = bounds
        return await operation(self, *args, **kwargs)

    return guarded


def scoped_creation_binding():
    bounds = _creation_bounds.get()
    return None if bounds is None else bounds.binding
