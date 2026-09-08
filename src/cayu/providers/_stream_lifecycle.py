"""Provider-neutral admission for native stream transitions.

Adapters decode their wire formats and validate native item assembly. This owner
decides whether the resulting transition may change state or become visible.
It owns no transport, clock, task, or durable transaction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from cayu.providers.deadlines import ProviderProgressKind


class StreamPhase(StrEnum):
    PRE_START = "pre_start"
    ACTIVE = "active"
    TERMINAL = "terminal"


class StreamTransitionKind(StrEnum):
    START = "start"
    IDENTITY = "identity"
    SEMANTIC = "semantic"
    USAGE = "usage"
    METADATA = "metadata"
    TERMINAL = "terminal"
    IGNORE = "ignore"
    FAILURE = "failure"


class StreamViolation(StrEnum):
    INVALID_TRANSITION = "invalid_transition"
    INVALID_IDENTITY = "invalid_identity"
    RESPONSE_CONFLICT = "response_conflict"
    MODEL_CONFLICT = "model_conflict"
    BEFORE_START = "before_start"
    REPEATED_START = "repeated_start"
    AFTER_TERMINAL = "after_terminal"
    REPEATED_TERMINAL = "repeated_terminal"
    TERMINAL_CONFLICT = "terminal_conflict"
    TAIL_LIMIT = "tail_limit"
    METADATA_LIMIT = "metadata_limit"
    MISSING_TERMINAL = "missing_terminal"
    REPEATED_COMPLETION = "repeated_completion"


class StreamLifecycleError(ValueError):
    """Content-free rejection; adapters translate the code to native errors."""

    def __init__(self, violation: StreamViolation) -> None:
        self.violation = violation
        super().__init__(f"Provider stream lifecycle rejected: {violation.value}.")


@dataclass(frozen=True, slots=True)
class StreamPolicy:
    require_start: bool = False
    max_terminal_repeats: int = 0
    max_usage_tails: int = 0
    max_metadata_tails: int = 0
    max_total_tails: int = 0
    max_metadata_events: int | None = None

    def __post_init__(self) -> None:
        if type(self.require_start) is not bool:
            raise TypeError("Stream policy require_start must be a bool.")
        for value in (
            self.max_terminal_repeats,
            self.max_usage_tails,
            self.max_metadata_tails,
            self.max_total_tails,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("Stream policy limits must be nonnegative integers.")
        if self.max_metadata_events is not None and (
            type(self.max_metadata_events) is not int or self.max_metadata_events < 0
        ):
            raise ValueError("Stream metadata limit must be a nonnegative integer or None.")


@dataclass(frozen=True, slots=True)
class StreamTerminal:
    """Native disposition plus immutable, canonical terminal comparison data."""

    outcome: str
    metadata: bytes = b""


@dataclass(frozen=True, slots=True)
class StreamTransition:
    kind: StreamTransitionKind
    response_id: str | None = None
    model: str | None = None
    terminal: StreamTerminal | None = None
    progress: ProviderProgressKind | None = None


@dataclass(frozen=True, slots=True)
class AcceptedTransition:
    phase: StreamPhase
    identity_started: bool = False
    terminal_started: bool = False
    repeated: bool = False
    progress: ProviderProgressKind | None = None


def _validate_identity(value: object) -> None:
    if value is None:
        return
    if type(value) is not str or not value.strip():
        raise StreamLifecycleError(StreamViolation.INVALID_IDENTITY)
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise StreamLifecycleError(StreamViolation.INVALID_IDENTITY) from None


class StreamLifecycle:
    """One response's state, with atomic admission and explicit tail policy."""

    def __init__(
        self,
        policy: StreamPolicy,
        *,
        resumed_response_id: str | None = None,
        resumed_model: str | None = None,
        error_factory: Callable[[StreamViolation], Exception] = StreamLifecycleError,
    ) -> None:
        if type(policy) is not StreamPolicy:
            raise TypeError("Stream lifecycle requires a StreamPolicy.")
        if resumed_model is not None and resumed_response_id is None:
            raise error_factory(StreamViolation.INVALID_IDENTITY)
        try:
            _validate_identity(resumed_response_id)
            _validate_identity(resumed_model)
        except StreamLifecycleError as failure:
            raise error_factory(failure.violation) from None
        self.policy = policy
        self._error_factory = error_factory
        self.phase = StreamPhase.PRE_START if resumed_response_id is None else StreamPhase.ACTIVE
        self.response_id = resumed_response_id
        self.model = resumed_model
        self.terminal: StreamTerminal | None = None
        self._terminal_repeats = 0
        self._usage_tails = 0
        self._metadata_tails = 0
        self._metadata_events = 0
        self._completion_claimed = False

    def accept(self, transition: StreamTransition) -> AcceptedTransition:
        """Validate the complete transition before changing any accepted state."""
        try:
            return self._accept(transition)
        except StreamLifecycleError as failure:
            if self._error_factory is StreamLifecycleError:
                raise
            raise self._error_factory(failure.violation) from None

    def _accept(self, transition: StreamTransition) -> AcceptedTransition:
        self._validate_shape(transition)
        kind = transition.kind
        if kind is StreamTransitionKind.IGNORE:
            return AcceptedTransition(self.phase)
        _validate_identity(transition.response_id)
        _validate_identity(transition.model)
        if (
            self.response_id is not None
            and transition.response_id is not None
            and transition.response_id != self.response_id
        ):
            raise StreamLifecycleError(StreamViolation.RESPONSE_CONFLICT)
        if (
            self.model is not None
            and transition.model is not None
            and self.model != transition.model
        ):
            raise StreamLifecycleError(StreamViolation.MODEL_CONFLICT)
        repeated = self._validate_phase(transition)
        if kind is StreamTransitionKind.METADATA and (
            self.policy.max_metadata_events is not None
            and self._metadata_events >= self.policy.max_metadata_events
        ):
            raise StreamLifecycleError(StreamViolation.METADATA_LIMIT)

        # There are no failing validations after this commit point.
        identity_started = (self.response_id is None and transition.response_id is not None) or (
            self.model is None and transition.model is not None
        )
        terminal_started = kind is StreamTransitionKind.TERMINAL and not repeated
        self.response_id = transition.response_id if self.response_id is None else self.response_id
        self.model = transition.model if self.model is None else self.model
        if kind is StreamTransitionKind.METADATA:
            self._metadata_events += 1
        if self.phase is StreamPhase.TERMINAL:
            self._terminal_repeats += int(repeated)
            self._usage_tails += int(kind is StreamTransitionKind.USAGE)
            self._metadata_tails += int(kind is StreamTransitionKind.METADATA)
        elif terminal_started:
            self.terminal = transition.terminal
            self.phase = StreamPhase.TERMINAL
        elif kind not in {StreamTransitionKind.IDENTITY, StreamTransitionKind.FAILURE}:
            self.phase = StreamPhase.ACTIVE
        return AcceptedTransition(
            self.phase,
            identity_started=identity_started,
            terminal_started=terminal_started,
            repeated=repeated,
            progress=None if repeated else transition.progress,
        )

    def require_terminal(self) -> None:
        if self.phase is not StreamPhase.TERMINAL:
            raise self._error_factory(StreamViolation.MISSING_TERMINAL)

    @property
    def metadata_received(self) -> bool:
        return self._metadata_events > 0

    @property
    def completion_claimed(self) -> bool:
        return self._completion_claimed

    def claim_completion(self) -> None:
        """Reserve the only normalized completion, after native assembly succeeds."""
        self.require_terminal()
        if self._completion_claimed:
            raise self._error_factory(StreamViolation.REPEATED_COMPLETION)
        self._completion_claimed = True

    def _validate_phase(self, transition: StreamTransition) -> bool:
        kind = transition.kind
        if self.phase is StreamPhase.TERMINAL:
            if kind is StreamTransitionKind.TERMINAL:
                if self._terminal_repeats >= self.policy.max_terminal_repeats:
                    raise StreamLifecycleError(StreamViolation.REPEATED_TERMINAL)
                if transition.terminal != self.terminal:
                    raise StreamLifecycleError(StreamViolation.TERMINAL_CONFLICT)
                return True
            if kind in {StreamTransitionKind.USAGE, StreamTransitionKind.METADATA}:
                allowed = (
                    self._usage_tails < self.policy.max_usage_tails
                    if kind is StreamTransitionKind.USAGE
                    else self._metadata_tails < self.policy.max_metadata_tails
                )
                if not allowed or (
                    self._usage_tails + self._metadata_tails >= self.policy.max_total_tails
                ):
                    raise StreamLifecycleError(StreamViolation.TAIL_LIMIT)
                return False
            raise StreamLifecycleError(StreamViolation.AFTER_TERMINAL)
        if kind is StreamTransitionKind.START:
            if self.phase is not StreamPhase.PRE_START:
                raise StreamLifecycleError(StreamViolation.REPEATED_START)
        elif (
            self.phase is StreamPhase.PRE_START
            and self.policy.require_start
            and kind is not StreamTransitionKind.FAILURE
        ):
            raise StreamLifecycleError(StreamViolation.BEFORE_START)
        return False

    @staticmethod
    def _validate_shape(transition: StreamTransition) -> None:
        if (
            type(transition) is not StreamTransition
            or type(transition.kind) is not StreamTransitionKind
        ):
            raise StreamLifecycleError(StreamViolation.INVALID_TRANSITION)
        terminal = transition.terminal
        if transition.kind is StreamTransitionKind.TERMINAL:
            if (
                type(terminal) is not StreamTerminal
                or type(terminal.outcome) is not str
                or not terminal.outcome
                or type(terminal.metadata) is not bytes
            ):
                raise StreamLifecycleError(StreamViolation.INVALID_TRANSITION)
        elif terminal is not None:
            raise StreamLifecycleError(StreamViolation.INVALID_TRANSITION)
        if transition.progress is not None and (
            type(transition.progress) is not ProviderProgressKind
            or transition.progress
            in {
                ProviderProgressKind.RESPONSE_IDENTITY,
                ProviderProgressKind.TERMINAL,
            }
        ):
            raise StreamLifecycleError(StreamViolation.INVALID_TRANSITION)
        if transition.kind is StreamTransitionKind.IGNORE and (
            transition.response_id is not None
            or transition.model is not None
            or transition.progress is not None
        ):
            raise StreamLifecycleError(StreamViolation.INVALID_TRANSITION)
