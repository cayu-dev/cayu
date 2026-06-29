from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.artifacts import FileAttachment, file_attachment_from_payload
from cayu.core.agents import AgentSpec
from cayu.core.events import EventType
from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    copy_message,
    copy_message_part,
)
from cayu.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelStreamEventType,
    copy_model_stream_event,
)
from cayu.runtime.sessions import Session
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeHit,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeVisibility,
)

_COMPACTION_CHECKPOINT_KEY = "context_compaction"
_COMPACTION_CHECKPOINT_VERSION = 1
_DEFAULT_KNOWLEDGE_INJECTION_MAX_HITS = 3
_DEFAULT_KNOWLEDGE_INJECTION_MAX_BYTES = 4000
_DEFAULT_KNOWLEDGE_INJECTION_QUERY_MAX_CHARS = 2000
_MIN_KNOWLEDGE_INJECTION_MAX_BYTES = 200
_MIN_KNOWLEDGE_INJECTION_QUERY_MAX_CHARS = 50


class ContextRequest(BaseModel):
    """Input passed to an agent context policy before each model request."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    agent: AgentSpec
    messages: list[Message]
    step: StrictInt = Field(ge=1)
    environment_name: str | None = None
    knowledge_store: Any = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("environment_name")
    @classmethod
    def validate_optional_environment_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "environment_name")


class ContextPolicy(ABC):
    """Builds the model-facing context for a runtime step.

    Policies may trim, summarize, replace tool results, or inject retrieved
    context. They must not be used as durable transcript storage.
    """

    @abstractmethod
    async def build(self, request: ContextRequest) -> list[Message]:
        """Return provider-neutral messages for one model request."""


class ContextCompactionTelemetry(BaseModel):
    """Compaction telemetry that the runtime converts into events."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_context_event_type(cls, value: EventType) -> EventType:
        if value not in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        }:
            raise ValueError("Context compaction telemetry event_type is not supported.")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "payload")


class ContextKnowledgeTelemetry(BaseModel):
    """Knowledge retrieval telemetry that the runtime converts into events."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_context_event_type(cls, value: EventType) -> EventType:
        if value not in {
            EventType.KNOWLEDGE_SEARCH_STARTED,
            EventType.KNOWLEDGE_SEARCH_COMPLETED,
            EventType.KNOWLEDGE_SEARCH_FAILED,
            EventType.KNOWLEDGE_INJECTED,
        }:
            raise ValueError("Context knowledge telemetry event_type is not supported.")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "payload")


def copy_context_compaction_telemetry(
    telemetry: ContextCompactionTelemetry,
) -> ContextCompactionTelemetry:
    if type(telemetry) is not ContextCompactionTelemetry:
        raise TypeError(
            "Context compaction telemetry must be ContextCompactionTelemetry instances."
        )
    return ContextCompactionTelemetry(
        event_type=telemetry.event_type,
        payload=copy_json_value(telemetry.payload, "payload"),
    )


def copy_context_knowledge_telemetry(
    telemetry: ContextKnowledgeTelemetry,
) -> ContextKnowledgeTelemetry:
    if type(telemetry) is not ContextKnowledgeTelemetry:
        raise TypeError("Context knowledge telemetry must be ContextKnowledgeTelemetry instances.")
    return ContextKnowledgeTelemetry(
        event_type=telemetry.event_type,
        payload=copy_json_value(telemetry.payload, "payload"),
    )


class ContextBuildResult(BaseModel):
    """Runtime-managed context result that may include checkpoint updates."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    checkpoint: dict[str, Any] | None = None
    checkpoint_event_payload: dict[str, Any] | None = None
    compaction_telemetry: list[ContextCompactionTelemetry] = Field(default_factory=list)
    knowledge_telemetry: list[ContextKnowledgeTelemetry] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("compaction_telemetry")
    @classmethod
    def copy_compaction_telemetry(cls, value):
        return [copy_context_compaction_telemetry(item) for item in value]

    @field_validator("knowledge_telemetry")
    @classmethod
    def copy_knowledge_telemetry(cls, value):
        return [copy_context_knowledge_telemetry(item) for item in value]

    @field_validator("checkpoint", "checkpoint_event_payload", mode="before")
    @classmethod
    def copy_optional_json_data(cls, value, info):
        if value is None:
            return None
        return copy_json_value(value, info.field_name)


class ContextBuildError(RuntimeError):
    """Context build failure with compaction telemetry to emit first."""

    def __init__(
        self,
        message: str,
        *,
        compaction_telemetry: list[ContextCompactionTelemetry],
        knowledge_telemetry: list[ContextKnowledgeTelemetry] | None = None,
        checkpoint: dict[str, Any] | None = None,
        checkpoint_event_payload: dict[str, Any] | None = None,
        cause: Exception,
    ) -> None:
        super().__init__(message)
        self.compaction_telemetry = tuple(
            copy_context_compaction_telemetry(item) for item in compaction_telemetry
        )
        self.knowledge_telemetry = tuple(
            copy_context_knowledge_telemetry(item)
            for item in ([] if knowledge_telemetry is None else knowledge_telemetry)
        )
        self.checkpoint = None if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        self.checkpoint_event_payload = (
            None
            if checkpoint_event_payload is None
            else copy_json_value(checkpoint_event_payload, "checkpoint_event_payload")
        )
        self.cause = cause


class RuntimeManagedContextPolicy(ContextPolicy):
    """Context policy whose checkpoint writes are owned by the runtime."""

    @abstractmethod
    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        """Return model-facing context and optional checkpoint updates."""

    async def build(self, request: ContextRequest) -> list[Message]:
        result = await self.build_with_checkpoint(request, checkpoint=None)
        return result.messages


class DefaultContextPolicy(ContextPolicy):
    """Default policy that sends transcript context with bounded file attachments."""

    def __init__(self, *, max_attachment_results: int = 1) -> None:
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build(self, request: ContextRequest) -> list[Message]:
        return strip_old_file_attachments(
            request.messages,
            max_attachment_results=self.max_attachment_results,
        )


class KnowledgeInjectionPolicy(RuntimeManagedContextPolicy):
    """Context policy wrapper that injects bounded retrieved knowledge.

    The injected message is model-facing context only. It is not written to the
    durable transcript.
    """

    def __init__(
        self,
        base_policy: ContextPolicy | None = None,
        *,
        enabled: bool = True,
        namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE,
        labels: dict[str, str] | None = None,
        kinds: list[str] | None = None,
        visibilities: list[KnowledgeVisibility] | None = None,
        aspects: list[str] | None = None,
        impact_targets: list[str] | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        mode: KnowledgeSearchMode = KnowledgeSearchMode.AUTO,
        include_expired: bool = False,
        max_hits: int = _DEFAULT_KNOWLEDGE_INJECTION_MAX_HITS,
        max_bytes: int = _DEFAULT_KNOWLEDGE_INJECTION_MAX_BYTES,
        query_max_chars: int = _DEFAULT_KNOWLEDGE_INJECTION_QUERY_MAX_CHARS,
        prefix: str = "Relevant knowledge retrieved for this request:",
        fail_open: bool = True,
    ) -> None:
        if base_policy is None:
            self.base_policy = DefaultContextPolicy()
        elif isinstance(base_policy, ContextPolicy):
            self.base_policy = base_policy
        else:
            raise TypeError("base_policy must be a ContextPolicy.")
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool.")
        if type(include_expired) is not bool:
            raise TypeError("include_expired must be a bool.")
        if type(fail_open) is not bool:
            raise TypeError("fail_open must be a bool.")
        self.enabled = enabled
        self.namespace = require_clean_nonblank(namespace, "namespace")
        self.labels = _copy_optional_label_map(labels)
        self.kinds = _copy_optional_clean_string_list(kinds, "kinds")
        self.visibilities = None if visibilities is None else list(dict.fromkeys(visibilities))
        self.aspects = _copy_clean_string_list(aspects, "aspects")
        self.impact_targets = _copy_clean_string_list(impact_targets, "impact_targets")
        self.source_type = _optional_clean_nonblank(source_type, "source_type")
        self.source_id = _optional_clean_nonblank(source_id, "source_id")
        self.mode = KnowledgeSearchMode(mode)
        self.include_expired = include_expired
        self.max_hits = _validate_positive_int(max_hits, "max_hits")
        self.max_bytes = _validate_minimum_int(
            max_bytes,
            "max_bytes",
            minimum=_MIN_KNOWLEDGE_INJECTION_MAX_BYTES,
        )
        self.query_max_chars = _validate_minimum_int(
            query_max_chars,
            "query_max_chars",
            minimum=_MIN_KNOWLEDGE_INJECTION_QUERY_MAX_CHARS,
        )
        self.prefix = require_nonblank(prefix, "prefix")
        self.fail_open = fail_open

    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        base_result = await self._build_base_context(request, checkpoint=checkpoint)
        if not self.enabled or request.knowledge_store is None:
            return base_result
        if not callable(getattr(request.knowledge_store, "search", None)):
            raise TypeError("ContextRequest.knowledge_store must implement KnowledgeStore.")

        query_text = _latest_user_text(base_result.messages, max_chars=self.query_max_chars)
        if query_text is None:
            return base_result

        query = KnowledgeQuery(
            text=query_text,
            namespace=self.namespace,
            labels=self.labels,
            kinds=self.kinds,
            visibilities=self.visibilities,
            aspects=self.aspects,
            impact_targets=self.impact_targets,
            source_type=self.source_type,
            source_id=self.source_id,
            mode=self.mode,
            include_expired=self.include_expired,
            limit=self.max_hits,
            max_bytes=self.max_bytes,
        )
        telemetry = list(base_result.knowledge_telemetry)
        telemetry.append(
            _knowledge_search_telemetry(
                event_type=EventType.KNOWLEDGE_SEARCH_STARTED,
                policy=self,
                query=query,
                payload={
                    "query_chars": len(query_text),
                },
            )
        )
        try:
            search_result = await request.knowledge_store.search(query)
        except Exception as exc:
            telemetry.append(
                _knowledge_search_telemetry(
                    event_type=EventType.KNOWLEDGE_SEARCH_FAILED,
                    policy=self,
                    query=query,
                    payload={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
            )
            if not self.fail_open:
                raise ContextBuildError(
                    str(exc),
                    compaction_telemetry=list(base_result.compaction_telemetry),
                    knowledge_telemetry=telemetry,
                    checkpoint=base_result.checkpoint,
                    checkpoint_event_payload=base_result.checkpoint_event_payload,
                    cause=exc,
                ) from exc
            return base_result.model_copy(update={"knowledge_telemetry": telemetry})

        telemetry.append(
            _knowledge_search_telemetry(
                event_type=EventType.KNOWLEDGE_SEARCH_COMPLETED,
                policy=self,
                query=query,
                payload={
                    "hit_count": len(search_result.hits),
                    "total_hits_known": search_result.total_hits_known,
                    "truncated": search_result.truncated,
                },
            )
        )
        if not search_result.hits:
            return base_result.model_copy(update={"knowledge_telemetry": telemetry})

        injection_text, injected_bytes = _format_knowledge_injection(
            search_result.hits,
            prefix=self.prefix,
            max_bytes=self.max_bytes,
        )
        injected_messages = _insert_before_latest_user_message(
            base_result.messages,
            Message.text(MessageRole.USER, injection_text),
        )
        telemetry.append(
            ContextKnowledgeTelemetry(
                event_type=EventType.KNOWLEDGE_INJECTED,
                payload={
                    "policy": type(self).__name__,
                    "hit_count": len(search_result.hits),
                    "injected_bytes": injected_bytes,
                    "sources": [_knowledge_source_payload(hit) for hit in search_result.hits],
                },
            )
        )
        return base_result.model_copy(
            update={
                "messages": injected_messages,
                "knowledge_telemetry": telemetry,
            }
        )

    async def _build_base_context(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        if isinstance(self.base_policy, RuntimeManagedContextPolicy):
            return await self.base_policy.build_with_checkpoint(
                request,
                checkpoint=checkpoint,
            )
        messages = await self.base_policy.build(request)
        return ContextBuildResult(messages=messages)


class MessageWindowContextPolicy(ContextPolicy):
    """Built-in policy that keeps a valid recent message window."""

    def __init__(
        self,
        *,
        max_messages: int,
        preserve_system: bool = True,
        max_attachment_results: int = 1,
    ) -> None:
        if type(max_messages) is not int:
            raise TypeError("max_messages must be an integer.")
        if type(preserve_system) is not bool:
            raise TypeError("preserve_system must be a bool.")
        if max_messages < 1:
            raise ValueError("max_messages must be greater than zero.")
        self.max_messages = max_messages
        self.preserve_system = preserve_system
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build(self, request: ContextRequest) -> list[Message]:
        trimmed = trim_context_messages(
            request.messages,
            max_messages=self.max_messages,
            preserve_system=self.preserve_system,
        )
        return strip_old_file_attachments(
            trimmed,
            max_attachment_results=self.max_attachment_results,
        )


class RecentTurnsContextPolicy(ContextPolicy):
    """Built-in policy that keeps recent user turns and complete tool rounds."""

    def __init__(
        self,
        *,
        max_user_turns: int,
        preserve_system: bool = True,
        max_attachment_results: int = 1,
    ) -> None:
        if type(max_user_turns) is not int:
            raise TypeError("max_user_turns must be an integer.")
        if type(preserve_system) is not bool:
            raise TypeError("preserve_system must be a bool.")
        if max_user_turns < 1:
            raise ValueError("max_user_turns must be greater than zero.")
        self.max_user_turns = max_user_turns
        self.preserve_system = preserve_system
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build(self, request: ContextRequest) -> list[Message]:
        trimmed = trim_context_turns(
            request.messages,
            max_user_turns=self.max_user_turns,
            preserve_system=self.preserve_system,
        )
        return strip_old_file_attachments(
            trimmed,
            max_attachment_results=self.max_attachment_results,
        )


class CompactionRequest(BaseModel):
    """Input passed to a compactor when older context needs summarizing."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    agent: AgentSpec
    messages: list[Message]
    existing_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("existing_summary")
    @classmethod
    def validate_optional_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "existing_summary")


CompactionPromptBuilder = Callable[[CompactionRequest], str]


class CompactionResult(BaseModel):
    """Compacted representation of older model-facing context."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return require_nonblank(value, "summary")

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class ContextCompactor(ABC):
    """Summarizes older context into durable checkpoint data."""

    @abstractmethod
    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Return a compact summary for older transcript messages."""


class TranscriptDigestCompactor(ContextCompactor):
    """Deterministic fallback compactor that stores a clipped text digest."""

    def __init__(self, *, max_summary_chars: int = 8000) -> None:
        if type(max_summary_chars) is not int:
            raise TypeError("max_summary_chars must be an integer.")
        if max_summary_chars < 200:
            raise ValueError("max_summary_chars must be at least 200.")
        self.max_summary_chars = max_summary_chars

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        sections: list[str] = []
        if request.existing_summary is not None:
            sections.append("Previous summary:\n" + request.existing_summary)
        if request.messages:
            sections.append("Newly compacted transcript:\n" + _messages_digest(request.messages))
        summary = "\n\n".join(sections)
        if len(summary) > self.max_summary_chars:
            summary = summary[-self.max_summary_chars :]
            summary = "[summary clipped to latest content]\n" + summary
        return CompactionResult(
            summary=summary,
            metadata={
                "compactor": type(self).__name__,
                "max_summary_chars": self.max_summary_chars,
            },
        )


class ModelCompactor(ContextCompactor):
    """Provider-backed compactor that asks a model to summarize older context."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        system_prompt: str = (
            "You summarize prior agent session context for a future model call. "
            "Return only the compact summary. Do not call tools."
        ),
        options: dict[str, Any] | None = None,
        max_input_chars: int | None = 120_000,
        prompt_builder: CompactionPromptBuilder | None = None,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must be a ModelProvider.")
        if max_input_chars is not None:
            if type(max_input_chars) is not int:
                raise TypeError("max_input_chars must be an integer or None.")
            if max_input_chars < 1000:
                raise ValueError("max_input_chars must be at least 1000.")
        self.provider = provider
        self.model = require_clean_nonblank(model, "model")
        self.system_prompt = require_nonblank(system_prompt, "system_prompt")
        self.options = copy_json_value({} if options is None else options, "options")
        self.max_input_chars = max_input_chars
        if prompt_builder is not None and not callable(prompt_builder):
            raise TypeError("prompt_builder must be callable.")
        self.prompt_builder = prompt_builder

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        if self.prompt_builder is None:
            bounded_prompt, input_truncated = _bounded_default_compaction_prompt(
                request,
                max_chars=self.max_input_chars,
            )
        else:
            user_prompt = require_nonblank(self.prompt_builder(request), "prompt")
            bounded_prompt, input_truncated = _bounded_prompt_text(
                user_prompt,
                max_chars=self.max_input_chars,
            )
        model_request = ModelRequest(
            model=self.model,
            messages=[
                Message.text(MessageRole.SYSTEM, self.system_prompt),
                Message.text(MessageRole.USER, bounded_prompt),
            ],
            tools=[],
            options=self.options,
        )

        text_parts: list[str] = []
        completed_payload: dict[str, Any] | None = None
        async for raw_event in self.provider.stream(model_request):
            event = copy_model_stream_event(raw_event)
            if completed_payload is not None:
                raise RuntimeError(
                    f"Compaction provider emitted event after completed: {event.type}"
                )
            if event.type == ModelStreamEventType.TEXT_DELTA:
                text_parts.append(event.delta)
            elif event.type == ModelStreamEventType.TOOL_CALL:
                raise RuntimeError("Compaction model must not call tools.")
            elif event.type == ModelStreamEventType.ERROR:
                raise RuntimeError(
                    str(event.payload.get("error") or "Compaction model provider error")
                )
            elif event.type == ModelStreamEventType.COMPLETED:
                completed_payload = event.payload
            else:
                raise RuntimeError(f"Compaction provider emitted unsupported event: {event.type}")

        if completed_payload is None:
            raise RuntimeError("Compaction model stream ended without a completed event.")

        summary = require_nonblank("".join(text_parts), "summary")
        return CompactionResult(
            summary=summary,
            metadata={
                "compactor": type(self).__name__,
                "provider": require_clean_nonblank(self.provider.name, "provider.name"),
                "model": self.model,
                "input_truncated": input_truncated,
                "max_input_chars": self.max_input_chars,
                "completed": _provider_completed_metadata(completed_payload),
            },
        )


class CheckpointCompactionContextPolicy(RuntimeManagedContextPolicy):
    """Checkpoint-backed context policy for long-running sessions.

    It keeps the durable transcript intact, stores a compact summary in the
    session checkpoint, and sends system messages + summary + recent turns to
    the model.
    """

    def __init__(
        self,
        *,
        compactor: ContextCompactor | None = None,
        max_user_turns: int = 10,
        compact_after_messages: int = 40,
        summary_prefix: str = "Previous session context summary:",
        max_attachment_results: int = 1,
    ) -> None:
        if compactor is None:
            self.compactor = TranscriptDigestCompactor()
        elif isinstance(compactor, ContextCompactor):
            self.compactor = compactor
        else:
            raise TypeError("compactor must be a ContextCompactor.")
        if type(max_user_turns) is not int:
            raise TypeError("max_user_turns must be an integer.")
        if type(compact_after_messages) is not int:
            raise TypeError("compact_after_messages must be an integer.")
        if max_user_turns < 1:
            raise ValueError("max_user_turns must be greater than zero.")
        if compact_after_messages < 1:
            raise ValueError("compact_after_messages must be greater than zero.")
        self.max_user_turns = max_user_turns
        self.compact_after_messages = compact_after_messages
        self.summary_prefix = require_nonblank(summary_prefix, "summary_prefix")
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        previous = _compaction_checkpoint(checkpoint)
        previous_summary = previous.get("summary") if previous is not None else None
        previous_cursor = (
            previous.get("compacted_transcript_cursor") if previous is not None else None
        )
        if type(previous_summary) is not str:
            previous_summary = None
            previous_cursor = None

        (
            system_prefix,
            compactable_messages,
            recent_messages,
            compactable_cursor,
        ) = _split_recent_turns(
            request.messages,
            max_user_turns=self.max_user_turns,
        )
        first_compactable_cursor = len(system_prefix)
        if (
            previous_summary is None
            or type(previous_cursor) is not int
            or previous_cursor < first_compactable_cursor
            or previous_cursor > compactable_cursor
        ):
            previous_cursor = first_compactable_cursor
            previous_summary = None

        newly_compactable = request.messages[previous_cursor:compactable_cursor]
        should_compact = len(compactable_messages) >= self.compact_after_messages and bool(
            newly_compactable
        )

        checkpoint_update = None
        checkpoint_event_payload = None
        compaction_telemetry: list[ContextCompactionTelemetry] = []
        summary = previous_summary
        if should_compact:
            compaction_started = _compaction_telemetry(
                event_type=EventType.CONTEXT_COMPACTION_STARTED,
                compactor=self.compactor,
                compacted_cursor=compactable_cursor,
                previous_cursor=previous_cursor,
                newly_compacted_message_count=len(newly_compactable),
                recent_message_count=len(recent_messages),
            )
            compaction_telemetry.append(compaction_started)
            try:
                result = await self.compactor.compact(
                    CompactionRequest(
                        session=request.session,
                        agent=request.agent,
                        messages=newly_compactable,
                        existing_summary=previous_summary,
                        metadata=request.metadata,
                    )
                )
            except Exception as exc:
                compaction_telemetry.append(
                    _compaction_telemetry(
                        event_type=EventType.CONTEXT_COMPACTION_FAILED,
                        compactor=self.compactor,
                        compacted_cursor=compactable_cursor,
                        previous_cursor=previous_cursor,
                        newly_compacted_message_count=len(newly_compactable),
                        recent_message_count=len(recent_messages),
                        payload={
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                )
                raise ContextBuildError(
                    str(exc),
                    compaction_telemetry=compaction_telemetry,
                    cause=exc,
                ) from exc
            summary = result.summary
            checkpoint_update = copy_json_value(checkpoint, "checkpoint")
            checkpoint_update[_COMPACTION_CHECKPOINT_KEY] = {
                "version": _COMPACTION_CHECKPOINT_VERSION,
                "summary": summary,
                "compacted_transcript_cursor": compactable_cursor,
                "metadata": result.metadata,
            }
            checkpoint_event_payload = {
                "checkpoint": _COMPACTION_CHECKPOINT_KEY,
                "compacted_transcript_cursor": compactable_cursor,
                "previous_compacted_transcript_cursor": previous_cursor,
                "newly_compacted_message_count": len(newly_compactable),
                "recent_message_count": len(recent_messages),
            }
            compaction_telemetry.append(
                _compaction_telemetry(
                    event_type=EventType.CONTEXT_COMPACTION_COMPLETED,
                    compactor=self.compactor,
                    compacted_cursor=compactable_cursor,
                    previous_cursor=previous_cursor,
                    newly_compacted_message_count=len(newly_compactable),
                    recent_message_count=len(recent_messages),
                    payload={
                        "summary_chars": len(summary),
                        "metadata": result.metadata,
                    },
                )
            )

        messages = [copy_message(message) for message in system_prefix]
        if summary is not None:
            messages.append(Message.text(MessageRole.USER, f"{self.summary_prefix}\n{summary}"))
        messages.extend(copy_message(message) for message in recent_messages)
        messages = strip_old_file_attachments(
            messages,
            max_attachment_results=self.max_attachment_results,
        )
        return ContextBuildResult(
            messages=messages,
            checkpoint=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=compaction_telemetry,
        )


def copy_context_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("ContextPolicy.build() must return a list of Message instances.")
    if not messages:
        raise ValueError("ContextPolicy.build() must return at least one message.")
    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)
    return copied_messages


def trim_context_messages(
    messages: list[Message],
    *,
    max_messages: int,
    preserve_system: bool = True,
) -> list[Message]:
    """Return a recent valid suffix without cutting through a tool round."""

    if type(max_messages) is not int:
        raise TypeError("max_messages must be an integer.")
    if type(preserve_system) is not bool:
        raise TypeError("preserve_system must be a bool.")
    if max_messages < 1:
        raise ValueError("max_messages must be greater than zero.")
    copied_messages = [copy_message(message) for message in messages]
    system_prefix, body = _split_system_prefix(copied_messages, preserve_system)
    candidate = system_prefix + body
    if len(candidate) <= max_messages:
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]

    body_limit = max(1, max_messages - len(system_prefix))
    start = max(0, len(body) - body_limit)
    for index in range(start, len(body)):
        candidate = system_prefix + body[index:]
        try:
            validate_context_messages(candidate)
        except ValueError:
            continue
        return [copy_message(message) for message in candidate]
    raise ValueError("Cannot trim context without cutting through a tool round.")


def trim_context_turns(
    messages: list[Message],
    *,
    max_user_turns: int,
    preserve_system: bool = True,
) -> list[Message]:
    """Return the latest user turns with complete assistant/tool follow-up."""

    if type(max_user_turns) is not int:
        raise TypeError("max_user_turns must be an integer.")
    if type(preserve_system) is not bool:
        raise TypeError("preserve_system must be a bool.")
    if max_user_turns < 1:
        raise ValueError("max_user_turns must be greater than zero.")

    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)

    system_prefix, body = _split_system_prefix(copied_messages, preserve_system)
    turn_starts = [index for index, message in enumerate(body) if message.role == MessageRole.USER]
    if not turn_starts:
        candidate = system_prefix + body
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]
    if len(turn_starts) <= max_user_turns:
        candidate = system_prefix + body
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]

    start = turn_starts[-max_user_turns]
    candidate = system_prefix + body[start:]
    validate_context_messages(candidate)
    return [copy_message(message) for message in candidate]


def strip_old_file_attachments(
    messages: list[Message],
    *,
    max_attachment_results: int = 1,
) -> list[Message]:
    """Remove old native file attachment refs from provider-facing context.

    Durable transcript messages keep their original artifacts. This helper only
    projects model-facing context so providers do not receive the same native
    file bytes on every subsequent model request.
    """

    if type(max_attachment_results) is not int:
        raise TypeError("max_attachment_results must be an integer.")
    if max_attachment_results < 0:
        raise ValueError("max_attachment_results must be non-negative.")

    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)

    attachment_positions: list[tuple[int, int]] = []
    for message_index, message in enumerate(copied_messages):
        if message.role != MessageRole.TOOL:
            continue
        for part_index, part in enumerate(message.content):
            if type(part) is not ToolResultPart:
                continue
            if _file_attachments_in_part(part):
                attachment_positions.append((message_index, part_index))

    if len(attachment_positions) <= max_attachment_results:
        return [copy_message(message) for message in copied_messages]

    keep_positions = (
        set()
        if max_attachment_results == 0
        else set(attachment_positions[-max_attachment_results:])
    )
    projected_messages: list[Message] = []
    for message_index, message in enumerate(copied_messages):
        if message.role != MessageRole.TOOL:
            projected_messages.append(copy_message(message))
            continue

        projected_parts: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart] = []
        for part_index, part in enumerate(message.content):
            if type(part) is not ToolResultPart or (message_index, part_index) in keep_positions:
                projected_parts.append(copy_message_part(part))
                continue
            projected_parts.append(_strip_file_attachments_from_tool_result(part))
        projected_messages.append(Message(role=message.role, content=projected_parts))

    validate_context_messages(projected_messages)
    return [copy_message(message) for message in projected_messages]


def validate_context_messages(messages: list[Message]) -> None:
    if type(messages) is not list:
        raise TypeError("Context messages must be a list of Message instances.")
    if not messages:
        raise ValueError("Context messages cannot be empty.")

    pending_tool_call_ids: set[str] | None = None
    for message in messages:
        if type(message) is not Message:
            raise TypeError("Context messages must be Message instances.")

        if pending_tool_call_ids is not None:
            if message.role != MessageRole.TOOL:
                raise ValueError(
                    "Context messages contain assistant tool calls that are not "
                    "followed by matching tool results."
                )
            result_parts = [part for part in message.content if type(part) is ToolResultPart]
            result_ids = [part.tool_call_id for part in result_parts]
            if len(result_ids) != len(set(result_ids)):
                raise ValueError("Context messages contain duplicate tool result ids.")
            if set(result_ids) != pending_tool_call_ids:
                raise ValueError(
                    "Context messages contain tool results that do not match the "
                    "preceding assistant tool calls."
                )
            pending_tool_call_ids = None
            continue

        if message.role == MessageRole.TOOL:
            raise ValueError(
                "Context messages contain tool results without preceding assistant tool calls."
            )

        if message.role == MessageRole.ASSISTANT:
            tool_call_ids = [
                part.tool_call_id for part in message.content if type(part) is ToolCallPart
            ]
            if len(tool_call_ids) != len(set(tool_call_ids)):
                raise ValueError("Context messages contain duplicate tool call ids.")
            if tool_call_ids:
                pending_tool_call_ids = set(tool_call_ids)

    if pending_tool_call_ids is not None:
        raise ValueError(
            "Context messages end with assistant tool calls that have no matching tool results."
        )


def _strip_file_attachments_from_tool_result(part: ToolResultPart) -> ToolResultPart:
    kept_artifacts: list[dict[str, Any]] = []
    stripped_attachments: list[FileAttachment] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is None:
            kept_artifacts.append(copy_json_value(payload, "artifact"))
        else:
            stripped_attachments.append(attachment)

    if not stripped_attachments:
        return ToolResultPart(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            content=part.content,
            structured=copy_json_value(part.structured, "structured"),
            artifacts=kept_artifacts,
            is_error=part.is_error,
        )

    content = _content_with_stripped_file_attachment_note(part.content, stripped_attachments)
    structured = copy_json_value(part.structured, "structured")
    if structured is None:
        structured = {}
    structured["cayu_file_attachments_stripped"] = [
        {
            "artifact_id": attachment.artifact_id,
            "filename": attachment.filename,
            "content_type": attachment.content_type,
            "size_bytes": attachment.size_bytes,
            "kind": attachment.kind.value,
        }
        for attachment in stripped_attachments
    ]
    return ToolResultPart(
        tool_call_id=part.tool_call_id,
        tool_name=part.tool_name,
        content=content,
        structured=structured,
        artifacts=kept_artifacts,
        is_error=part.is_error,
    )


def _content_with_stripped_file_attachment_note(
    content: str,
    attachments: list[FileAttachment],
) -> str:
    lines = [
        (
            f"- {attachment.filename} ({attachment.content_type}, "
            f"{attachment.size_bytes} bytes, artifact_id={attachment.artifact_id})"
        )
        for attachment in attachments
    ]
    note = "File attachments from this older tool result were omitted from this provider request:\n"
    note += "\n".join(lines)
    if content:
        return f"{content}\n\n{note}"
    return note


def _file_attachments_in_part(part: ToolResultPart) -> tuple[FileAttachment, ...]:
    attachments: list[FileAttachment] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is not None:
            attachments.append(attachment)
    return tuple(attachments)


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _validate_minimum_int(value: int, field_name: str, *, minimum: int) -> int:
    _validate_positive_int(value, field_name)
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return value


def _copy_optional_label_map(value: dict[str, str] | None) -> dict[str, str]:
    if value is None:
        return {}
    return copy_label_map(value, "labels")


def _optional_clean_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


def _copy_clean_string_list(value: list[str] | None, field_name: str) -> list[str]:
    copied = _copy_optional_clean_string_list(value, field_name)
    return [] if copied is None else copied


def _copy_optional_clean_string_list(
    value: list[str] | None,
    field_name: str,
) -> list[str] | None:
    if value is None:
        return None
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list.")
    result = [
        require_clean_nonblank(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    ]
    return list(dict.fromkeys(result))


def _latest_user_text(messages: list[Message], *, max_chars: int) -> str | None:
    for message in reversed(messages):
        if message.role != MessageRole.USER:
            continue
        text = "\n".join(part.text for part in message.content if type(part) is TextPart).strip()
        if not text:
            return None
        if len(text) <= max_chars:
            return text
        marker = "[query clipped to latest content]\n"
        keep_chars = max_chars - len(marker)
        if keep_chars <= 0:
            return text[-max_chars:]
        return marker + text[-keep_chars:]
    return None


def _insert_before_latest_user_message(
    messages: list[Message],
    injection: Message,
) -> list[Message]:
    copied = [copy_message(message) for message in messages]
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].role == MessageRole.USER:
            return [*copied[:index], copy_message(injection), *copied[index:]]
    return [*copied, copy_message(injection)]


def _format_knowledge_injection(
    hits: list[KnowledgeHit],
    *,
    prefix: str,
    max_bytes: int,
) -> tuple[str, int]:
    lines = [
        prefix,
        "Use these snippets as background context. They may be incomplete; cite entry ids when relying on them.",
    ]
    for index, hit in enumerate(hits, start=1):
        entry = hit.entry
        title = f" title={entry.title!r}" if entry.title else ""
        chunk = f" chunk_index={hit.chunk.chunk_index}" if hit.chunk is not None else ""
        score = f" score={hit.score:.4f}" if hit.score is not None else ""
        text = hit.text_preview or (hit.chunk.text if hit.chunk is not None else entry.text)
        lines.append(
            f"{index}. entry_id={entry.id!r} kind={entry.kind!r}{title}{chunk}{score}\n{text}"
        )
    injected = _truncate_text_to_bytes("\n\n".join(lines), max_bytes)
    if not injected:
        injected = prefix
    return injected, len(injected.encode("utf-8"))


def _truncate_text_to_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    marker = "\n[knowledge context truncated]"
    marker_bytes = marker.encode("utf-8")
    if max_bytes <= len(marker_bytes):
        return marker[:max_bytes]
    clipped = encoded[: max_bytes - len(marker_bytes)]
    return clipped.decode("utf-8", errors="ignore").rstrip() + marker


def _knowledge_search_telemetry(
    *,
    event_type: EventType,
    policy: KnowledgeInjectionPolicy,
    query: KnowledgeQuery,
    payload: dict[str, Any] | None = None,
) -> ContextKnowledgeTelemetry:
    event_payload = {
        "policy": type(policy).__name__,
        "query": _knowledge_query_payload(query),
    }
    if payload is not None:
        event_payload.update(copy_json_value(payload, "payload"))
    return ContextKnowledgeTelemetry(event_type=event_type, payload=event_payload)


def _knowledge_query_payload(query: KnowledgeQuery) -> dict[str, Any]:
    return {
        "namespace": query.namespace,
        "labels": dict(query.labels),
        "kinds": None if query.kinds is None else list(query.kinds),
        "visibilities": (
            None
            if query.visibilities is None
            else [visibility.value for visibility in query.visibilities]
        ),
        "aspects": list(query.aspects),
        "impact_targets": list(query.impact_targets),
        "source_type": query.source_type,
        "source_id": query.source_id,
        "mode": query.mode.value,
        "include_expired": query.include_expired,
        "limit": query.limit,
        "max_bytes": query.max_bytes,
    }


def _knowledge_source_payload(hit: KnowledgeHit) -> dict[str, Any]:
    entry = hit.entry
    return {
        "entry_id": entry.id,
        "namespace": entry.namespace,
        "kind": entry.kind,
        "title": entry.title,
        "chunk_id": hit.chunk.id if hit.chunk is not None else None,
        "chunk_index": hit.chunk.chunk_index if hit.chunk is not None else None,
        "score": hit.score,
        "score_kind": hit.score_kind,
        "reason": hit.reason,
    }


def _validate_max_attachment_results(value: int) -> int:
    if type(value) is not int:
        raise TypeError("max_attachment_results must be an integer.")
    if value < 0:
        raise ValueError("max_attachment_results must be non-negative.")
    return value


def _split_system_prefix(
    messages: list[Message],
    preserve_system: bool,
) -> tuple[list[Message], list[Message]]:
    system_prefix: list[Message] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            break
        system_prefix.append(message)
    body = messages[len(system_prefix) :]
    if not preserve_system:
        return [], body
    return system_prefix, body


def _split_recent_turns(
    messages: list[Message],
    *,
    max_user_turns: int,
) -> tuple[list[Message], list[Message], list[Message], int]:
    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)
    system_prefix, body = _split_system_prefix(copied_messages, True)
    turn_starts = [index for index, message in enumerate(body) if message.role == MessageRole.USER]
    if not turn_starts or len(turn_starts) <= max_user_turns:
        return system_prefix, [], body, len(system_prefix)
    recent_start = turn_starts[-max_user_turns]
    compactable_cursor = len(system_prefix) + recent_start
    return system_prefix, body[:recent_start], body[recent_start:], compactable_cursor


def _compaction_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    value = checkpoint.get(_COMPACTION_CHECKPOINT_KEY)
    if type(value) is not dict:
        return None
    if value.get("version") != _COMPACTION_CHECKPOINT_VERSION:
        return None
    return copy_json_value(value, _COMPACTION_CHECKPOINT_KEY)


def _compaction_telemetry(
    *,
    event_type: EventType,
    compactor: ContextCompactor,
    compacted_cursor: int,
    previous_cursor: int,
    newly_compacted_message_count: int,
    recent_message_count: int,
    payload: dict[str, Any] | None = None,
) -> ContextCompactionTelemetry:
    event_payload = {
        "checkpoint": _COMPACTION_CHECKPOINT_KEY,
        "compactor": type(compactor).__name__,
        "compacted_transcript_cursor": compacted_cursor,
        "previous_compacted_transcript_cursor": previous_cursor,
        "newly_compacted_message_count": newly_compacted_message_count,
        "recent_message_count": recent_message_count,
    }
    if payload is not None:
        event_payload.update(copy_json_value(payload, "payload"))
    return ContextCompactionTelemetry(event_type=event_type, payload=event_payload)


def _messages_digest(messages: list[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        parts = [_message_part_digest(part) for part in message.content]
        lines.append(f"{message.role}: " + " ".join(parts))
    return "\n".join(lines)


def _message_part_digest(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
) -> str:
    if type(part) is TextPart:
        return part.text
    if type(part) is ToolCallPart:
        return (
            f"[tool_call id={part.tool_call_id} name={part.tool_name} "
            f"arguments={copy_json_value(part.arguments, 'arguments')}]"
        )
    if type(part) is ToolResultPart:
        return (
            f"[tool_result id={part.tool_call_id} name={part.tool_name} "
            f"error={part.is_error} content={part.content}]"
        )
    if type(part) is ProviderStatePart:
        return f"[provider_state provider={part.provider}]"
    raise TypeError("Unsupported message part.")


def _provider_completed_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    copied = copy_json_value(payload, "completed")
    if type(copied) is not dict:
        raise ValueError("Provider completed payload must be an object.")
    copied.pop("provider_state", None)
    return copied


def default_compaction_prompt(
    request: CompactionRequest,
) -> str:
    """Build the default user prompt for model-backed context compaction."""

    prefix, transcript_prefix, transcript_digest = _default_compaction_prompt_parts(request)
    return f"{prefix}\n\n{transcript_prefix}{transcript_digest}"


def _bounded_default_compaction_prompt(
    request: CompactionRequest,
    *,
    max_chars: int | None,
) -> tuple[str, bool]:
    prefix, transcript_prefix, transcript_digest = _default_compaction_prompt_parts(request)
    prompt = f"{prefix}\n\n{transcript_prefix}{transcript_digest}"
    if max_chars is None or len(prompt) <= max_chars:
        return prompt, False

    marker = "[compaction transcript clipped to latest content]\n"
    available = max_chars - len(prefix) - 2 - len(transcript_prefix) - len(marker)
    if available <= 0:
        raise ValueError(
            "max_input_chars is too small to preserve compaction instructions and existing summary."
        )
    return (
        f"{prefix}\n\n{transcript_prefix}{marker}{transcript_digest[-available:]}",
        True,
    )


def _default_compaction_prompt_parts(
    request: CompactionRequest,
) -> tuple[str, str, str]:
    sections = [
        "Summarize the transcript below so a future agent step can continue with the important context.",
        "Preserve concrete user requests, decisions, files or resources mentioned, tool results, errors, and pending work.",
        "Do not invent facts. Keep the summary concise but specific.",
        f"Session: {request.session.id}",
        f"Agent: {request.agent.name}",
    ]
    if request.existing_summary is not None:
        sections.append("Existing summary:\n" + request.existing_summary)
    prefix = "\n\n".join(sections)
    transcript_prefix = "Transcript to compact:\n"
    transcript_digest = _messages_digest(request.messages)
    return prefix, transcript_prefix, transcript_digest


def _bounded_prompt_text(
    prompt: str,
    *,
    max_chars: int | None,
) -> tuple[str, bool]:
    if max_chars is None or len(prompt) <= max_chars:
        return prompt, False
    marker = "[compaction input clipped to latest content]\n"
    keep_chars = max_chars - len(marker)
    if keep_chars <= 0:
        raise ValueError("max_chars is too small for compaction prompt marker.")
    return marker + prompt[-keep_chars:], True
