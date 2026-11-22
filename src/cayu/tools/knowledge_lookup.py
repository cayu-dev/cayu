"""``find_knowledge``: look for kept entries and pending proposals like a draft, read-only.

A model calls it before ``remember_knowledge`` to see whether the fact it is about to
remember is already kept or waiting for review. It reads the entries in the scope the
caller would remember into (the same ``RememberKnowledgePolicy`` namespace, visibility
and labels, with labels copied from the calling session), scores them against the draft
with a ``KnowledgeSimilarity`` part, and returns the best matches by id, title, status,
kind and number of sources. It never returns the body of an unreviewed proposal, so an
unchecked claim can be recognised but can't steer the model; a kept entry's body is a
``read_knowledge`` call away, as with ``search_knowledge``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from cayu.knowledge._sources import knowledge_source_count
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.search import KnowledgeListQuery, encode_knowledge_list_cursor
from cayu.knowledge.similarity import (
    KnowledgeComparable,
    KnowledgeSimilarity,
    KnowledgeSimilarityMatch,
    LexicalKnowledgeSimilarity,
)
from cayu.tools._errors import structured_invalid_arguments, tool_argument_validation
from cayu.tools._redaction import (
    await_revision_stable_secret_output,
    unstable_secret_redaction_result,
)
from cayu.tools._terminal_bounds import (
    DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES,
    bounded_terminal_result,
    spec_with_terminal_payload_bound,
)
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.display import ToolDisplay
from cayu.tools.knowledge import (
    RememberKnowledgePolicy,
    _copy_remember_knowledge_policy,
    _missing_knowledge_store_result,
    _optional_arg_string,
    _optional_positive_int,
    _require_arg_string,
    _require_knowledge_store,
    _validate_bounded_positive_int,
)
from cayu.vaults.redaction import SecretRedactor

DEFAULT_FIND_KNOWLEDGE_LIMIT = 5
MAX_FIND_KNOWLEDGE_LIMIT = 10
#: How many kept and pending entries one call compares at most, in list order (most
#: important, then most recently updated, first).
DEFAULT_FIND_KNOWLEDGE_MAX_CANDIDATES = 500
MAX_FIND_KNOWLEDGE_MAX_CANDIDATES = 5_000
#: The longest draft text one call accepts, in UTF-8 bytes.
MAX_FIND_KNOWLEDGE_TEXT_BYTES = 64 * 1024
_TITLE_MAX_BYTES = 300
_PAGE_SIZE = 100
#: How many listed entries one call reads at most, as a multiple of ``max_candidates``,
#: counting those outside the similarity scope.
_SCAN_FACTOR = 4
_PAGE_MAX_BYTES = 1_048_576
_FIND_KNOWLEDGE_STORE_METHODS = ("list_entries",)
_STATUS_LABELS = {
    KnowledgeStatus.ACTIVE: "kept",
    KnowledgeStatus.PENDING: "unreviewed proposal",
}


class FindKnowledgeTool(Tool):
    spec = ToolSpec(
        display=ToolDisplay(verb="Looked for", noun="similar knowledge"),
        name="find_knowledge",
        effect=ToolEffect.NONE,
        description=(
            "Before remember_knowledge, check whether the same fact is already kept or "
            "waiting for review. Give the title and text you would remember. Returns the "
            "closest kept entries and unreviewed proposals in the scope you would remember "
            "into, by id, title, status, kind and number of sources. Unreviewed proposals "
            "have no body here: they are unchecked, so don't treat them as true. Read a kept "
            "entry with read_knowledge."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "The short title you would remember the fact under.",
                },
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "The knowledge text you would remember.",
                },
                "kind": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": "\\S",
                    "description": "Optional: compare only with entries of this kind.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_FIND_KNOWLEDGE_LIMIT,
                    "description": "Maximum number of matches to return.",
                },
            },
            "required": ["text"],
        },
    )

    def __init__(
        self,
        spec: ToolSpec | None = None,
        *,
        policy: RememberKnowledgePolicy | dict[str, Any] | None = None,
        similarity: KnowledgeSimilarity | None = None,
        max_candidates: int = DEFAULT_FIND_KNOWLEDGE_MAX_CANDIDATES,
        max_terminal_payload_bytes: int | None = None,
    ) -> None:
        """``policy`` is the ``remember_knowledge`` policy whose scope to look in (pass
        the same one). ``similarity`` defaults to ``LexicalKnowledgeSimilarity`` scoped on
        the policy's copied and required labels."""

        super().__init__(
            spec=spec_with_terminal_payload_bound(
                type(self).spec if spec is None else spec,
                max_terminal_payload_bytes,
                default=DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES,
            )
        )
        self._policy = (
            RememberKnowledgePolicy() if policy is None else _copy_remember_knowledge_policy(policy)
        )
        if similarity is not None and not isinstance(similarity, KnowledgeSimilarity):
            raise TypeError("similarity must be a KnowledgeSimilarity.")
        self._default_similarity = similarity is None
        self._similarity = (
            LexicalKnowledgeSimilarity(
                scope_labels=(*self._policy.copy_session_labels, *self._policy.require_labels)
            )
            if similarity is None
            else similarity
        )
        self._max_candidates = _validate_bounded_positive_int(
            max_candidates, "max_candidates", maximum=MAX_FIND_KNOWLEDGE_MAX_CANDIDATES
        )

    def _execution_profile_material(self) -> dict[str, Any] | None:
        """Material only for Cayu's default policy and similarity part.

        Application policy holds exact storage scope values, and a custom part is
        opaque code, so either keeps the tool process-local, as ``remember_knowledge``.
        """

        if self._policy != RememberKnowledgePolicy() or not self._default_similarity:
            return None
        return {
            "policy": "cayu_default",
            "similarity": "lexical_default",
            "max_candidates": self._max_candidates,
        }

    @property
    def _publish_arguments(self) -> bool:
        """Keep draft knowledge out of terminal audit and transcript projections."""

        return False

    @property
    def retain_arguments_for_model(self) -> bool:
        """Keep the draft for the model's next request, which may remember it."""

        return True

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        store = _require_knowledge_store(
            ctx, methods=_FIND_KNOWLEDGE_STORE_METHODS, tool_name=self.spec.name
        )
        if store is None:
            return _missing_knowledge_store_result()
        with tool_argument_validation():
            text = _require_arg_string(args, "text")
            if len(text.encode("utf-8")) > MAX_FIND_KNOWLEDGE_TEXT_BYTES:
                raise ValueError(f"`text` must be at most {MAX_FIND_KNOWLEDGE_TEXT_BYTES} bytes.")
            title = _optional_arg_string(args, "title")
            kind = _optional_arg_string(args, "kind")
            limit = _optional_positive_int(
                args,
                "limit",
                default=DEFAULT_FIND_KNOWLEDGE_LIMIT,
                maximum=MAX_FIND_KNOWLEDGE_LIMIT,
            )
            policy = self._policy
            labels, _ = policy.labels_for(ctx.session_labels)
            subject = KnowledgeComparable(
                text=text,
                title=title,
                kind=kind or policy.default_kind,
                namespace=policy.default_namespace,
                visibility=policy.default_visibility,
                labels=labels,
            )
        query = KnowledgeListQuery(
            namespace=policy.default_namespace,
            labels=labels,
            kinds=None if kind is None else [kind],
            statuses=list(_STATUS_LABELS),
            visibilities=[policy.default_visibility],
            limit=min(_PAGE_SIZE, self._max_candidates),
            max_bytes=_PAGE_MAX_BYTES,
            count_total=False,
        )

        def in_scope(entry: KnowledgeEntry) -> bool:
            return self._similarity.same_scope(subject, KnowledgeComparable.from_entry(entry))

        async def candidates(_redactor: SecretRedactor) -> tuple[list[KnowledgeEntry], bool]:
            return await _list_candidates(
                store, query, max_candidates=self._max_candidates, in_scope=in_scope
            )

        captured = await await_revision_stable_secret_output(ctx, candidates)
        if captured is None:
            return unstable_secret_redaction_result()
        (entries, more), capture_snapshot = captured
        redactor = capture_snapshot.redactor
        matches = await self._similarity.matches(
            subject,
            [KnowledgeComparable.from_entry(entry) for entry in entries],
            limit=limit,
        )
        rows = [_match_payload(entries[match.index], match, redactor=redactor) for match in matches]
        return ToolResult(
            content=_format_matches(rows, compared=len(entries), more=more),
            structured={
                "matches": rows,
                "compared": len(entries),
                "candidates_truncated": more,
                "limit": limit,
            },
        )


async def _list_candidates(
    store: Any,
    query: KnowledgeListQuery,
    *,
    max_candidates: int,
    in_scope: Callable[[KnowledgeEntry], bool],
) -> tuple[list[KnowledgeEntry], bool]:
    """Kept and pending entries in the query's scope that ``in_scope`` admits, in list
    order, at most ``max_candidates``, and whether more exist.

    Entries the similarity scope would refuse (one with a scope label the session lacks)
    don't count toward the cap, so they can't crowd out ones that could match; reading
    stops after ``_SCAN_FACTOR`` times the cap all the same."""

    entries: list[KnowledgeEntry] = []
    seen: set[str] = set()
    scanned = 0
    page_query = query
    while True:
        result = await store.list_entries(page_query)
        scanned += len(result.entries)
        for item in result.entries:
            if (
                item.entry.id not in seen
                and item.entry.status in _STATUS_LABELS
                and in_scope(item.entry)
            ):
                seen.add(item.entry.id)
                entries.append(item.entry)
        more = result.truncated and bool(result.entries)
        if len(entries) >= max_candidates:
            return entries[:max_candidates], more or len(entries) > max_candidates
        if not more:
            # A page cut short by its byte bound with no entry on it can't advance.
            return entries, result.truncated
        if scanned >= max_candidates * _SCAN_FACTOR:
            return entries, True
        page_query = page_query.model_copy(
            update={"cursor": encode_knowledge_list_cursor(result.entries[-1].entry)}
        )


def _match_payload(
    entry: KnowledgeEntry, match: KnowledgeSimilarityMatch, *, redactor: SecretRedactor
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "entry_id": entry.id,
        "title": (
            None
            if entry.title is None
            else redactor.redact_text_bounded(entry.title, max_bytes=_TITLE_MAX_BYTES)
        ),
        "status": _STATUS_LABELS[entry.status],
        "kind": entry.kind,
        "sources": knowledge_source_count(entry),
        "score": round(match.score.score, 3),
    }
    if entry.status is KnowledgeStatus.ACTIVE:
        row["read_with"] = "read_knowledge"
    return row


def _format_matches(rows: list[dict[str, Any]], *, compared: int, more: bool) -> str:
    if not rows:
        lines = [f"No similar knowledge among {compared} kept entries and proposals."]
    else:
        lines = [f"{len(rows)} similar of {compared} kept entries and proposals:"]
        for row in rows:
            sources = "1 source" if row["sources"] == 1 else f"{row['sources']} sources"
            title = row["title"] if row["title"] is not None else "(no title)"
            lines.append(
                f"- {row['entry_id']} ({row['status']}, {row['kind']}, {sources}, "
                f"score {row['score']}): {title}"
            )
        lines.append("Unreviewed proposals are unchecked; read a kept entry with read_knowledge.")
    if more:
        lines.append(
            f"Only the first {compared} entries in list order were compared; others may match."
        )
    return "\n".join(lines)
