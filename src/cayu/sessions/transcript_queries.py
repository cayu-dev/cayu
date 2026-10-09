"""Transcript paging and search contracts with their shared lexical and cursor rules."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import secrets
import unicodedata
from bisect import bisect_left
from collections.abc import Callable, Mapping
from hashlib import sha256
from itertools import islice

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    FrozenJsonDict,
    canonical_durable_json_bytes,
    copy_durable_json_object,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.messages import Message, MessageRole, TextPart, ThinkingPart, detach_message
from cayu.sessions.records import MAX_SESSION_ID_BYTES, TranscriptRecord


class TranscriptPage(BaseModel):
    """A retained-row page, not a logical-history extent.

    ``total_records`` counts all retained rows matching the query's role and
    interaction filters, before offset/limit and thinking-content projection.
    Use ``load_transcript_cursor`` for logical extent, or
    ``load_transcript_snapshot`` for retained records and extent in one snapshot.
    """

    model_config = ConfigDict(extra="forbid")

    records: list[TranscriptRecord] = Field(default_factory=list)
    total_records: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)


TRANSCRIPT_SEARCH_TOKENIZER_VERSION = (
    f"cayu.transcript.tokenizer.v1+unicode-{unicodedata.unidata_version}"
)
TRANSCRIPT_SEARCH_INDEX_VERSION = f"cayu.transcript.text.v1+{TRANSCRIPT_SEARCH_TOKENIZER_VERSION}"
TRANSCRIPT_SEARCH_DEFAULT_LIMIT = 20
TRANSCRIPT_SEARCH_MAX_LIMIT = 100
TRANSCRIPT_SEARCH_DEFAULT_MAX_BYTES = 32_000
TRANSCRIPT_SEARCH_MAX_BYTES = 1_000_000
TRANSCRIPT_SEARCH_DEFAULT_SCAN_LIMIT = 10_000
TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT = 100_000
TRANSCRIPT_SEARCH_MAX_SESSION_IDS = 100
TRANSCRIPT_SEARCH_MAX_QUERY_BYTES = 8_192
TRANSCRIPT_SEARCH_MAX_CURSOR_BYTES = 4_096
TRANSCRIPT_SEARCH_MIN_MAX_BYTES = 4
_TRANSCRIPT_SEARCH_CURSOR_VERSION = 1
_TRANSCRIPT_SEARCH_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Scoring scale for search occurrence counts; independent of message size limits.
_TRANSCRIPT_SEARCH_OCCURRENCE_CAP = 65_536
_TRANSCRIPT_SEARCH_COVERAGE_SCALE = _TRANSCRIPT_SEARCH_OCCURRENCE_CAP + 1
_TRANSCRIPT_SEARCH_PHRASE_SCALE = (
    TRANSCRIPT_SEARCH_MAX_QUERY_BYTES + 1
) * _TRANSCRIPT_SEARCH_COVERAGE_SCALE
_TRANSCRIPT_SEARCH_MAX_INLINE_TOKEN_BYTES = 512


class TranscriptQuery(BaseModel):
    """Paginate retained matching rows in ascending absolute-index order.

    ``offset`` skips retained matches, not absolute indexes. Separate calls are
    live reads: concurrent retention can shift offsets. Use absolute windows
    when continuing by message identity instead of retained-row position.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    interaction_id: str | None = None
    role: MessageRole | str | None = None
    offset: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    limit: StrictInt = Field(default=100, ge=1, le=5000)
    # When False, ThinkingPart content is stripped from the returned messages. This is a
    # content view, not a record filter: `total_records` stays the role-matched total, a
    # page may hold fewer than `limit` records when thinking-only turns drop out, and each
    # record keeps its true transcript `index` (so offset pagination is unaffected).
    include_thinking: StrictBool = True

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "interaction_id")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: MessageRole | str | None) -> MessageRole | None:
        if value is None:
            return None
        return MessageRole(value)


class TranscriptSearchQuery(BaseModel):
    """Bounded lexical search over explicitly selected authoritative transcripts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    text: str
    session_ids: tuple[str, ...]
    roles: tuple[MessageRole, ...] = (
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )
    before_transcript_indexes: Mapping[str, int] = Field(default_factory=dict)
    limit: int = TRANSCRIPT_SEARCH_DEFAULT_LIMIT
    max_bytes: int = TRANSCRIPT_SEARCH_DEFAULT_MAX_BYTES
    max_records_scanned: int = TRANSCRIPT_SEARCH_DEFAULT_SCAN_LIMIT
    cursor: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = require_nonblank(value, "text")
        if len(value.encode("utf-8")) > TRANSCRIPT_SEARCH_MAX_QUERY_BYTES:
            raise ValueError(
                f"`text` must be at most {TRANSCRIPT_SEARCH_MAX_QUERY_BYTES} UTF-8 bytes."
            )
        if not transcript_search_query_tokens(value):
            raise ValueError("`text` must contain at least one searchable word.")
        return value

    @field_validator("session_ids", mode="before")
    @classmethod
    def validate_session_ids(cls, value) -> tuple[str, ...]:
        if isinstance(value, str | bytes):
            raise ValueError("`session_ids` must be a sequence of session identifiers.")
        try:
            values = list(islice(value, TRANSCRIPT_SEARCH_MAX_SESSION_IDS + 1))
        except TypeError as exc:
            raise ValueError("`session_ids` must be a sequence of session identifiers.") from exc
        if not values:
            raise ValueError("Transcript search requires at least one explicit session id.")
        if len(values) > TRANSCRIPT_SEARCH_MAX_SESSION_IDS:
            raise ValueError(
                f"`session_ids` cannot contain more than {TRANSCRIPT_SEARCH_MAX_SESSION_IDS} ids."
            )
        cleaned = [require_clean_nonblank(item, "session_ids") for item in values]
        if any(len(session_id.encode("utf-8")) > MAX_SESSION_ID_BYTES for session_id in cleaned):
            raise ValueError(
                f"`session_ids` values must be at most {MAX_SESSION_ID_BYTES} UTF-8 bytes."
            )
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("`session_ids` cannot contain duplicates.")
        return tuple(sorted(cleaned))

    @field_validator("roles", mode="before")
    @classmethod
    def validate_roles(cls, value) -> tuple[MessageRole, ...]:
        if isinstance(value, str | bytes):
            raise ValueError("`roles` must be a sequence of message roles.")
        try:
            roles = tuple(MessageRole(item) for item in islice(value, 3))
        except (TypeError, ValueError) as exc:
            raise ValueError("`roles` contains an invalid message role.") from exc
        if not roles:
            raise ValueError("`roles` cannot be empty.")
        if len(roles) > 2:
            raise ValueError("`roles` cannot contain more than two narrative roles.")
        if len(roles) != len(set(roles)):
            raise ValueError("`roles` cannot contain duplicates.")
        if MessageRole.SYSTEM in roles or MessageRole.TOOL in roles:
            raise ValueError("Transcript recall indexes only user and assistant narrative text.")
        return tuple(sorted(roles, key=str))

    @field_validator("before_transcript_indexes", mode="before")
    @classmethod
    def validate_before_transcript_indexes(cls, value) -> dict[str, int]:
        copied = copy_durable_json_object(value, "before_transcript_indexes")
        result: dict[str, int] = {}
        for session_id, index in copied.items():
            session_id = require_clean_nonblank(session_id, "before_transcript_indexes key")
            if type(index) is not int or not 0 <= index <= MAX_DURABLE_JSON_INTEGER:
                raise ValueError(
                    "`before_transcript_indexes` values must be non-negative durable integers."
                )
            result[session_id] = index
        return {session_id: result[session_id] for session_id in sorted(result)}

    @field_validator("before_transcript_indexes")
    @classmethod
    def freeze_before_transcript_indexes(
        cls,
        value: Mapping[str, int],
    ) -> Mapping[str, int]:
        return FrozenJsonDict(value)

    @field_serializer("before_transcript_indexes")
    def serialize_before_transcript_indexes(
        self,
        value: Mapping[str, int],
    ) -> dict[str, int]:
        return dict(value)

    @model_validator(mode="after")
    def validate_before_index_scope(self) -> TranscriptSearchQuery:
        if not set(self.before_transcript_indexes).issubset(self.session_ids):
            raise ValueError(
                "`before_transcript_indexes` keys must belong to the selected session scope."
            )
        return self

    @field_validator("limit", "max_bytes", "max_records_scanned", mode="before")
    @classmethod
    def validate_bounds(cls, value, info) -> int:
        if type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        maximum = {
            "limit": TRANSCRIPT_SEARCH_MAX_LIMIT,
            "max_bytes": TRANSCRIPT_SEARCH_MAX_BYTES,
            "max_records_scanned": TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT,
        }[info.field_name]
        minimum = TRANSCRIPT_SEARCH_MIN_MAX_BYTES if info.field_name == "max_bytes" else 1
        if not minimum <= value <= maximum:
            raise ValueError(f"`{info.field_name}` must be between {minimum} and {maximum}.")
        return value

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, "cursor")
        if len(value.encode("ascii", errors="ignore")) != len(value):
            raise ValueError("`cursor` must contain ASCII characters only.")
        if len(value) > TRANSCRIPT_SEARCH_MAX_CURSOR_BYTES:
            raise ValueError(
                f"`cursor` must be at most {TRANSCRIPT_SEARCH_MAX_CURSOR_BYTES} bytes."
            )
        return value


class TranscriptSearchHit(BaseModel):
    """One exact transcript record matched through its non-thinking text parts."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    transcript_index: int
    interaction_id: str | None = None
    role: MessageRole
    text: str
    text_complete: bool
    content_hash: str
    text_part_indexes: tuple[int, ...]
    raw_score: float | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "interaction_id")

    @field_validator("transcript_index", mode="before")
    @classmethod
    def validate_transcript_index(cls, value) -> int:
        if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("`transcript_index` must be a non-negative durable integer.")
        return value

    @field_validator("text")
    @classmethod
    def validate_hit_text(cls, value: str) -> str:
        return require_nonblank(value, "text")

    @field_validator("text_complete", mode="before")
    @classmethod
    def validate_text_complete(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`text_complete` must be a boolean.")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("`content_hash` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("text_part_indexes", mode="before")
    @classmethod
    def validate_text_part_indexes(cls, value) -> tuple[int, ...]:
        if isinstance(value, int | str | bytes):
            raise ValueError("`text_part_indexes` must be a sequence of indexes.")
        try:
            indexes = tuple(value)
        except TypeError as exc:
            raise ValueError("`text_part_indexes` must be a sequence of indexes.") from exc
        if not indexes or any(type(index) is not int or index < 0 for index in indexes):
            raise ValueError("`text_part_indexes` must contain non-negative integers.")
        if indexes != tuple(sorted(set(indexes))):
            raise ValueError("`text_part_indexes` must be unique and ascending.")
        return indexes

    @field_validator("raw_score", mode="before")
    @classmethod
    def validate_raw_score(cls, value) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`raw_score` must be a number.")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("`raw_score` must be finite.")
        return score


class TranscriptSearchResult(BaseModel):
    """One bounded transcript-search page and its exact continuation frontier."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    query: TranscriptSearchQuery
    hits: tuple[TranscriptSearchHit, ...] = ()
    index_version: str = TRANSCRIPT_SEARCH_INDEX_VERSION
    matched_records_examined: int = 0
    truncated: bool = False
    coverage_complete: bool = True
    next_cursor: str | None = None

    @field_validator("query", mode="before")
    @classmethod
    def copy_query(cls, value) -> TranscriptSearchQuery:
        if type(value) is not TranscriptSearchQuery:
            raise TypeError("`query` must be a TranscriptSearchQuery.")
        return copy_transcript_search_query(value)

    @field_validator("hits", mode="before")
    @classmethod
    def copy_hits(cls, value) -> tuple[TranscriptSearchHit, ...]:
        return tuple(copy_transcript_search_hit(item) for item in value)

    @field_validator("index_version")
    @classmethod
    def validate_index_version(cls, value: str) -> str:
        return require_clean_nonblank(value, "index_version")

    @field_validator("matched_records_examined", mode="before")
    @classmethod
    def validate_records_examined(cls, value) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("`matched_records_examined` must be a non-negative integer.")
        return value

    @field_validator("truncated", "coverage_complete", mode="before")
    @classmethod
    def validate_boolean(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return TranscriptSearchQuery.validate_cursor(value)

    @model_validator(mode="after")
    def validate_page(self) -> TranscriptSearchResult:
        if len(self.hits) > self.query.limit:
            raise ValueError("Transcript search hits exceed the query limit.")
        identities = [(hit.session_id, hit.transcript_index) for hit in self.hits]
        if len(identities) != len(set(identities)):
            raise ValueError("Transcript search hits cannot repeat a transcript record.")
        if self.next_cursor is not None and not self.truncated:
            raise ValueError("A transcript continuation requires `truncated=True`.")
        if not self.coverage_complete and not self.truncated:
            raise ValueError("Incomplete transcript coverage must be reported as truncated.")
        if not self.coverage_complete and (self.hits or self.next_cursor is not None):
            raise ValueError(
                "Incomplete transcript coverage cannot expose a partial ranking or cursor."
            )
        if self.matched_records_examined > self.query.max_records_scanned:
            raise ValueError("Transcript search exceeded `max_records_scanned`.")
        if len(self.hits) > self.matched_records_examined:
            raise ValueError("Transcript search cannot return more hits than it examined.")
        if any(hit.session_id not in self.query.session_ids for hit in self.hits):
            raise ValueError("Transcript search returned a session outside the explicit scope.")
        if any(hit.role not in self.query.roles for hit in self.hits):
            raise ValueError("Transcript search returned a role outside the explicit scope.")
        if tuple(self.hits) != tuple(
            sorted(
                self.hits,
                key=lambda hit: (
                    -(hit.raw_score if hit.raw_score is not None else float("-inf")),
                    hit.session_id,
                    -hit.transcript_index,
                ),
            )
        ):
            raise ValueError("Transcript search hits must use deterministic relevance order.")
        if sum(len(hit.text.encode("utf-8")) for hit in self.hits) > self.query.max_bytes:
            raise ValueError("Transcript search hits exceeded `max_bytes`.")
        return self


def _without_thinking_parts(message: Message) -> Message | None:
    kept = [part for part in message.content if type(part) is not ThinkingPart]
    if not kept:
        return None
    return Message(role=message.role, content=tuple(kept))


def filter_transcript_records(
    records: list[TranscriptRecord], *, include_thinking: bool
) -> list[TranscriptRecord]:
    """Apply a `TranscriptQuery.include_thinking` filter to a page of records.

    When ``include_thinking`` is False, ThinkingParts are stripped from each message and
    records whose message is left empty (a thinking-only turn) are dropped. Every
    surviving message on that path is freshly rebuilt through full validation, so it
    shares no payload state with the input records; when True, records pass through
    unchanged and the caller is responsible for any isolation.
    """
    if include_thinking:
        return records
    filtered: list[TranscriptRecord] = []
    for record in records:
        message = _without_thinking_parts(record.message)
        if message is not None:
            filtered.append(
                TranscriptRecord(
                    index=record.index,
                    interaction_id=record.interaction_id,
                    message=message,
                )
            )
    return filtered


def copy_transcript_query(query: TranscriptQuery) -> TranscriptQuery:
    if type(query) is not TranscriptQuery:
        raise TypeError("Transcript queries must be TranscriptQuery instances.")
    return TranscriptQuery(
        session_id=query.session_id,
        interaction_id=query.interaction_id,
        role=query.role,
        offset=query.offset,
        limit=query.limit,
        include_thinking=query.include_thinking,
    )


def transcript_search_query_tokens(text: str) -> tuple[str, ...]:
    """Return stable case-folded lexical terms shared by every backend."""

    if type(text) is not str:
        raise TypeError("Transcript search text must be a string.")
    return tuple(
        dict.fromkeys(token.casefold() for token in _TRANSCRIPT_SEARCH_TOKEN_RE.findall(text))
    )


def transcript_search_document_from_text(text: str) -> str:
    """Encode narrative terms as tokenizer-safe portable ASCII lexemes.

    Database tokenizers must never reinterpret the lexical boundary selected by
    Cayu. Ordinary case-folded UTF-8 terms use a collision-free hexadecimal
    identity; long terms use a separately prefixed SHA-256 identity so they
    cannot exceed database lexeme limits. Repeated terms are retained for
    deterministic scoring.
    """

    if type(text) is not str:
        raise TypeError("Transcript search document text must be a string.")
    return " ".join(
        _transcript_search_token_identity(token.casefold())
        for token in _TRANSCRIPT_SEARCH_TOKEN_RE.findall(text)
    )


def transcript_search_query_document(text: str) -> str:
    """Return the deduplicated portable document used for one query."""

    return " ".join(
        _transcript_search_token_identity(token) for token in transcript_search_query_tokens(text)
    )


def _transcript_search_token_identity(token: str) -> str:
    encoded = token.encode("utf-8")
    if len(encoded) <= _TRANSCRIPT_SEARCH_MAX_INLINE_TOKEN_BYTES:
        return "x" + encoded.hex()
    return "h" + sha256(encoded).hexdigest()


def transcript_search_text(message: Message) -> tuple[str, tuple[int, ...]]:
    """Project only provider-neutral narrative text; reasoning and tool payloads stay out."""

    if type(message) is not Message:
        raise TypeError("Transcript search requires a Message.")
    parts = [
        (index, part.text) for index, part in enumerate(message.content) if type(part) is TextPart
    ]
    return "\n".join(text for _, text in parts), tuple(index for index, _ in parts)


def transcript_search_document(message: Message) -> str:
    """Return the canonical indexed lexical document for one transcript message."""

    return transcript_search_document_from_text(transcript_search_text(message)[0])


def transcript_search_session_token(session_id: str) -> str:
    """Return the FTS-safe opaque term used to constrain SQLite candidate lookup."""

    session_id = require_clean_nonblank(session_id, "session_id")
    return "s" + sha256(session_id.encode("utf-8")).hexdigest()


def transcript_search_query_fingerprint(query: TranscriptSearchQuery) -> str:
    if type(query) is not TranscriptSearchQuery:
        raise TypeError("query must be a TranscriptSearchQuery.")
    material = {
        "text": query.text,
        "session_ids": list(query.session_ids),
        "roles": [str(role) for role in query.roles],
        "before_transcript_indexes": dict(query.before_transcript_indexes),
    }
    return sha256(canonical_durable_json_bytes(material, "transcript search query")).hexdigest()


def encode_transcript_search_cursor(
    query: TranscriptSearchQuery,
    *,
    raw_score: int,
    session_id: str,
    transcript_index: int,
) -> str:
    query = copy_transcript_search_query(query, cursor=None)
    session_id = require_clean_nonblank(session_id, "session_id")
    if session_id not in query.session_ids:
        raise ValueError("Transcript search cursor session is outside the query scope.")
    if type(raw_score) is not int or not 0 <= raw_score <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Transcript search cursor score is invalid.")
    if type(transcript_index) is not int or not 0 <= transcript_index <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Transcript search cursor index is invalid.")
    material = {
        "version": _TRANSCRIPT_SEARCH_CURSOR_VERSION,
        "query_sha256": transcript_search_query_fingerprint(query),
        "raw_score": raw_score,
        "session_id_b64": base64.urlsafe_b64encode(session_id.encode("utf-8")).decode("ascii"),
        "transcript_index": transcript_index,
    }
    encoded = base64.urlsafe_b64encode(
        canonical_durable_json_bytes(material, "transcript search cursor")
    ).decode("ascii")
    if len(encoded) > TRANSCRIPT_SEARCH_MAX_CURSOR_BYTES:
        raise ValueError("Transcript search cursor exceeds its byte limit.")
    return encoded


def decode_transcript_search_cursor(
    query: TranscriptSearchQuery,
) -> tuple[int, str, int] | None:
    query = copy_transcript_search_query(query)
    if query.cursor is None:
        return None
    try:
        encoded = query.cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical transcript search cursor.")
        decoded = json.loads(raw.decode("utf-8"))
        if (
            type(decoded) is not dict
            or set(decoded)
            != {
                "version",
                "query_sha256",
                "raw_score",
                "session_id_b64",
                "transcript_index",
            }
            or decoded["version"] != _TRANSCRIPT_SEARCH_CURSOR_VERSION
            or type(decoded["query_sha256"]) is not str
            or type(decoded["raw_score"]) is not int
            or type(decoded["session_id_b64"]) is not str
            or type(decoded["transcript_index"]) is not int
        ):
            raise ValueError("Invalid transcript search cursor material.")
        encoded_session = decoded["session_id_b64"].encode("ascii")
        session_bytes = base64.b64decode(encoded_session, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(session_bytes) != encoded_session:
            raise ValueError("Non-canonical transcript search cursor session.")
        session_id = session_bytes.decode("utf-8")
        expected = transcript_search_query_fingerprint(
            copy_transcript_search_query(query, cursor=None)
        )
        if not secrets.compare_digest(decoded["query_sha256"], expected):
            raise ValueError("Transcript search cursor belongs to another query.")
        if session_id not in query.session_ids:
            raise ValueError("Transcript search cursor session is outside the query scope.")
        transcript_index = decoded["transcript_index"]
        if not 0 <= transcript_index <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("Transcript search cursor index is invalid.")
        raw_score = decoded["raw_score"]
        if not 0 <= raw_score <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("Transcript search cursor score is invalid.")
        return raw_score, session_id, transcript_index
    except (
        binascii.Error,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError("Invalid transcript search cursor.") from exc


def transcript_search_position_after_cursor(
    *,
    raw_score: int,
    session_id: str,
    transcript_index: int,
    cursor: tuple[int, str, int],
) -> bool:
    """Return whether a ranked identity follows an exact keyset frontier."""

    cursor_score, cursor_session_id, cursor_transcript_index = cursor
    return raw_score < cursor_score or (
        raw_score == cursor_score
        and (
            session_id > cursor_session_id
            or (session_id == cursor_session_id and transcript_index < cursor_transcript_index)
        )
    )


def copy_transcript_search_query(
    query: TranscriptSearchQuery,
    *,
    cursor: str | None | object = ...,
) -> TranscriptSearchQuery:
    if type(query) is not TranscriptSearchQuery:
        raise TypeError("Transcript search queries must be TranscriptSearchQuery instances.")
    resolved_cursor = query.cursor if cursor is ... else cursor
    if resolved_cursor is not None and type(resolved_cursor) is not str:
        raise TypeError("Transcript search cursor must be a string or None.")
    return TranscriptSearchQuery(
        text=query.text,
        session_ids=tuple(query.session_ids),
        roles=tuple(query.roles),
        before_transcript_indexes=dict(query.before_transcript_indexes),
        limit=query.limit,
        max_bytes=query.max_bytes,
        max_records_scanned=query.max_records_scanned,
        cursor=resolved_cursor,
    )


def copy_transcript_search_hit(hit: TranscriptSearchHit) -> TranscriptSearchHit:
    if type(hit) is not TranscriptSearchHit:
        raise TypeError("Transcript search hits must be TranscriptSearchHit instances.")
    return TranscriptSearchHit.model_validate(hit.model_dump(mode="python"))


def transcript_search_hit_from_message(
    *,
    session_id: str,
    transcript_index: int,
    interaction_id: str | None,
    message: Message,
    max_text_bytes: int,
    raw_score: float | None = None,
) -> TranscriptSearchHit | None:
    if type(max_text_bytes) is not int or max_text_bytes < 1:
        return None
    text, part_indexes = transcript_search_text(message)
    if not text or not part_indexes:
        return None
    encoded = text.encode("utf-8")
    preview = encoded[:max_text_bytes].decode("utf-8", errors="ignore")
    if not preview:
        return None
    return TranscriptSearchHit(
        session_id=session_id,
        transcript_index=transcript_index,
        interaction_id=interaction_id,
        role=message.role,
        text=preview,
        text_complete=len(preview.encode("utf-8")) == len(encoded),
        content_hash=sha256(encoded).hexdigest(),
        text_part_indexes=part_indexes,
        raw_score=raw_score,
    )


def transcript_search_score(text: str, query: TranscriptSearchQuery) -> float:
    """Return a deterministic diagnostic score; fusion consumes only the rank."""

    document = transcript_search_document_from_text(text)
    query_document = transcript_search_query_document(query.text)
    return float(transcript_search_document_score(document, query_document))


def transcript_search_document_score(document: str, query_document: str) -> int:
    """Score canonical documents identically after indexed candidate lookup."""

    if type(document) is not str or type(query_document) is not str:
        raise TypeError("Transcript search documents must be strings.")
    document_tokens = tuple(document.split())
    query_tokens = tuple(query_document.split())
    query_set = set(query_tokens)
    document_set = set(document_tokens)
    coverage = sum(token in document_set for token in query_tokens)
    occurrences = min(
        sum(token in query_set for token in document_tokens),
        _TRANSCRIPT_SEARCH_OCCURRENCE_CAP,
    )
    phrase = int(_contains_token_sequence(document_tokens, query_tokens))
    score = (
        phrase * _TRANSCRIPT_SEARCH_PHRASE_SCALE
        + coverage * _TRANSCRIPT_SEARCH_COVERAGE_SCALE
        + occurrences
    )
    if score > MAX_DURABLE_JSON_INTEGER:  # pragma: no cover - durable input bounds are tighter
        raise ValueError("Transcript search score exceeds the durable integer limit.")
    return score


def _contains_token_sequence(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    prefix_lengths = [0] * len(needle)
    matched = 0
    for index in range(1, len(needle)):
        while matched and needle[index] != needle[matched]:
            matched = prefix_lengths[matched - 1]
        if needle[index] == needle[matched]:
            matched += 1
            prefix_lengths[index] = matched
    matched = 0
    for token in haystack:
        while matched and token != needle[matched]:
            matched = prefix_lengths[matched - 1]
        if token == needle[matched]:
            matched += 1
            if matched == len(needle):
                return True
    return False


LATEST_TRANSCRIPT_TEXT_MAX_CHARS = 32_000


LATEST_TRANSCRIPT_TEXT_MAX_PARTS = 4_096


LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES = 2 * 1024 * 1024


class TranscriptTextReadLimitExceeded(RuntimeError):
    """A bounded text projection cannot safely inspect its source message."""


def _bounded_transcript_message_text(
    message: Message,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    """Project text with bounded part visits and at most one look-ahead character."""

    pieces: list[str] = []
    retained_chars = 0
    for part_index, part in enumerate(message.content):
        if part_index >= LATEST_TRANSCRIPT_TEXT_MAX_PARTS:
            raise TranscriptTextReadLimitExceeded(
                "Transcript message exceeds the bounded content-part inspection limit."
            )
        if type(part) is not TextPart:
            continue
        remaining = max_chars + 1 - retained_chars
        if remaining <= 0:
            break
        piece = part.text[:remaining]
        pieces.append(piece)
        retained_chars += len(piece)
    text = "".join(pieces)
    return text[:max_chars], len(text) > max_chars


class TranscriptSnapshot(BaseModel):
    """Retained transcript records plus the permanent append cursor."""

    model_config = ConfigDict(extra="forbid")

    records: list[TranscriptRecord] = Field(default_factory=list)
    cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @model_validator(mode="after")
    def validate_record_order(self) -> TranscriptSnapshot:
        indices = [record.index for record in self.records]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("Transcript snapshot records must have unique ascending indices.")
        if indices and indices[-1] >= self.cursor:
            raise ValueError("Transcript snapshot records must precede its append cursor.")
        return self

    def retained_position(self, cursor: int) -> int:
        """Map an exact absolute cursor to this snapshot's retained positions.

        A removed cursor raises ValueError, even if later records survive.
        The snapshot's append cursor maps to the end, including empty retention.
        Unlike ``load_transcript_window``, this does not skip missing history.
        """

        if type(cursor) is not int:
            raise TypeError("Transcript cursor must be an integer.")
        if not 0 <= cursor <= self.cursor:
            raise ValueError("Transcript cursor exceeds the snapshot append cursor.")
        indices = [record.index for record in self.records]
        position = bisect_left(indices, cursor)
        if position < len(indices):
            if indices[position] != cursor:
                raise ValueError("Transcript cursor is not available in retained history.")
        elif cursor != self.cursor:
            raise ValueError("Transcript cursor is not available in retained history.")
        return position


ForkTranscriptValidator = Callable[[tuple[Message, ...], TranscriptSnapshot], bool]


def fork_source_transcript_sha256(snapshot: TranscriptSnapshot) -> str:
    """Hash the permanent cursor and absolute positions of retained source messages."""

    if type(snapshot) is not TranscriptSnapshot:
        raise TypeError("snapshot must be a TranscriptSnapshot.")
    return sha256(
        canonical_durable_json_bytes(
            {
                "record_type": "cayu.fork-source-transcript",
                "schema_version": 1,
                "cursor": snapshot.cursor,
                "records": [
                    {
                        "index": record.index,
                        "message": record.message.model_dump(mode="json", warnings=False),
                    }
                    for record in snapshot.records
                ],
            },
            "fork_source.transcript",
        )
    ).hexdigest()


def fork_transcript_is_accepted(
    messages: list[Message],
    source_snapshot: TranscriptSnapshot | None,
    validator: ForkTranscriptValidator | None,
) -> bool:
    """Require explicit positive validation for one atomic source/copy snapshot."""

    if validator is None:
        return True
    if type(source_snapshot) is not TranscriptSnapshot:
        raise TypeError("source_snapshot must be a TranscriptSnapshot.")
    # A validator is an external callback. Give it isolated projections so
    # mutation cannot alter the source transcript, its absolute indexes, or the
    # messages that will be committed to the child.
    validation_messages = tuple(detach_message(message) for message in messages)
    validation_source_snapshot: TranscriptSnapshot | None = TranscriptSnapshot.model_validate(
        source_snapshot.model_dump(mode="json", warnings=False)
    )
    try:
        accepted = validator(validation_messages, validation_source_snapshot)
    except Exception:
        return False
    finally:
        validation_messages = ()
        validation_source_snapshot = None
    return type(accepted) is bool and accepted
