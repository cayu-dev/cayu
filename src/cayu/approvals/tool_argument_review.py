"""A ready-made human review policy: show named arguments of named tools.

Most applications that pause a tool for approval want the approver to read a
few of its arguments (an issue's title and body, a payment's amount and payee)
and nothing else. ``ToolArgumentReview`` is that policy, assembled from the
same parts as any hand-written ``HumanReviewPolicy``; write your own when it
doesn't fit.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from cayu.approvals.review import (
    _FIELD_LABEL_MAX_CHARS,
    _FIELD_TEXT_MAX_CHARS,
    _FIELDS_MAX_BYTES,
    HumanReviewContext,
    HumanReviewDisclosure,
    HumanReviewField,
    HumanReviewLongField,
    HumanReviewPolicy,
    HumanReviewSource,
    derive_human_review_binding_key,
)
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime.public_authority import PublicAuthorityAliasKeyring

RecipientAuthorizer = Callable[..., bool]

_MAX_FIELDS = 32
_NOT_SHOWN_LABEL = "Not shown"
# Room kept for the note that says how much didn't fit.
_NOTE_RESERVE_BYTES = 256
_UNLISTED_TOOL_TEXT = "Arguments are not shown for this tool."
# A long field's key: the call's position in the round, then the argument's in ``fields``.
_LONG_FIELD_KEY = re.compile(r"([1-9][0-9]{0,3})\.([1-9][0-9]{0,3})\Z")
_NO_KEY_TEXT = (
    "ToolArgumentReview has no binding_key: pass one, or give the CayuApp (or its "
    "session store) a public authority alias keyring to derive one from."
)


class ToolArgumentReview(HumanReviewPolicy):
    """Show an approver the named arguments of named tools, and agents' questions.

    ``fields`` maps a tool name to the argument names to show, in order::

        CayuApp(
            human_review_policy=ToolArgumentReview({"file_issue": ("title", "body")}),
            public_authority_alias_keyring=keyring,
            ...,
        )

    Without ``binding_key`` the app derives one from its public authority alias
    keyring (``derive_human_review_binding_key``), so it is the same for every
    worker and restart that shares the keyring. An app without a keyring needs
    ``binding_key``: 32+ private random bytes, shared by all workers.

    Calls to tools ``fields`` doesn't name stay approvable as they are without a
    policy: the view names the tool and says its arguments are not shown. A
    question from ``ask_user`` shows its question and options only in static
    secret scope, unless ``attest_questions=True`` explicitly attests that text.
    A long argument is split across numbered fields; the
    whole view stays within the review limits (32 fields, 16 KiB), and what
    doesn't fit is replaced by a note saying how many characters are not shown.
    Every listed argument the view doesn't show in one field is also a long
    field the approver can read whole. The decision still covers the complete
    arguments.

    Any recipient the server authenticated may inspect and decide with
    ``purpose``; ``cayu serve`` points the dashboard at it. Pass
    ``authorize_recipient(context, *, session_id, session_metadata, action)`` to
    narrow who may. SDK callers build ``HumanReviewContext`` themselves and must
    pass only verified identities.

    Listed arguments are shown when the paused round resolved no secrets after
    it was proposed (static secret scope). With ``attest_listed_arguments=True``
    the application asserts the listed arguments are safe to show its operators
    in any scope; without it, a round in a dynamic or unknown scope can only be
    denied. ``version`` changes on its own with every setting here; pass a new
    ``revision`` when ``authorize_recipient`` changes what it allows.
    """

    def __init__(
        self,
        fields: Mapping[str, Sequence[str]],
        *,
        binding_key: bytes | None = None,
        purpose: str = "approvals",
        questions: bool = True,
        attest_listed_arguments: bool = False,
        attest_questions: bool = False,
        authorize_recipient: RecipientAuthorizer | None = None,
        revision: str = "1",
    ) -> None:
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("fields must name at least one tool.")
        selected: dict[str, tuple[str, ...]] = {}
        for tool_name, argument_names in fields.items():
            if type(tool_name) is not str or not tool_name:
                raise ValueError("fields keys must be tool names.")
            if isinstance(argument_names, str) or not isinstance(argument_names, Sequence):
                raise ValueError(f"fields[{tool_name!r}] must be a sequence of argument names.")
            names = tuple(argument_names)
            if not names or any(type(name) is not str or not name for name in names):
                raise ValueError(f"fields[{tool_name!r}] must name at least one argument.")
            selected[tool_name] = names
        if binding_key is not None and (type(binding_key) is not bytes or len(binding_key) < 32):
            raise ValueError("binding_key must be at least 32 private random bytes.")
        if type(purpose) is not str or not 1 <= len(purpose) <= 128 or purpose != purpose.strip():
            raise ValueError("purpose must be 1 to 128 characters without surrounding spaces.")
        if authorize_recipient is not None and not callable(authorize_recipient):
            raise TypeError("authorize_recipient must be callable.")
        if type(revision) is not str or not 1 <= len(revision) <= 32:
            raise ValueError("revision must be 1 to 32 characters.")
        self._fields = selected
        self._binding_key = binding_key
        self._explicit_key = binding_key is not None
        self._purpose = purpose
        self._questions = questions is True
        self._attest = attest_listed_arguments is True
        self._attest_questions = attest_questions is True
        self._authorize_recipient = authorize_recipient
        settings = json.dumps(
            {
                "fields": selected,
                "purpose": purpose,
                "questions": self._questions,
                "attest": self._attest,
                "attest_questions": self._attest_questions,
                "revision": revision,
            },
            sort_keys=True,
        )
        self._version = (
            "tool-argument-review-v1:" + hashlib.sha256(settings.encode()).hexdigest()[:32]
        )

    @property
    def version(self) -> str:
        return self._version

    @property
    def binding_key(self) -> bytes:
        if self._binding_key is None:
            raise ValueError(_NO_KEY_TEXT)
        return self._binding_key

    def _bind_application_keyring(self, keyring: PublicAuthorityAliasKeyring | None) -> None:
        if self._explicit_key:
            return
        if keyring is None:
            raise ValueError(_NO_KEY_TEXT)
        derived = derive_human_review_binding_key(keyring, purpose=self._purpose)
        if self._binding_key is not None and not hmac.compare_digest(self._binding_key, derived):
            raise ValueError(
                "This ToolArgumentReview already uses another app's alias keyring; "
                "give each app its own policy or pass binding_key."
            )
        self._binding_key = derived

    @property
    def dashboard_purpose(self) -> str:
        return self._purpose

    def authorize(
        self,
        context: HumanReviewContext,
        *,
        session_id: str,
        session_metadata: Mapping[str, Any],
        action: Literal["inspect", "decide"],
    ) -> bool:
        if context.purpose != self._purpose or action not in ("inspect", "decide"):
            return False
        if self._authorize_recipient is None:
            return True
        return (
            self._authorize_recipient(
                context, session_id=session_id, session_metadata=session_metadata, action=action
            )
            is True
        )

    def _project_with_redactor(
        self, context: HumanReviewContext, source: HumanReviewSource, redactor: SecretRedactor
    ) -> HumanReviewDisclosure:
        # Scan complete selected values before JSON formatting, splitting or truncation.
        if source.kind == "user_input":
            selected = [source.question, list(source.options)] if self._questions else []
        else:
            selected = [
                [
                    call.tool_name,
                    {
                        name: arguments[name]
                        for name in self._fields.get(call.tool_name, ())
                        if name in arguments
                    },
                ]
                for call in source.calls
                for arguments in (source.arguments_by_call.get(call.tool_call_id) or {},)
            ]
        if redactor.redact_json(selected) != selected:
            return HumanReviewDisclosure(status="redacted")
        return self.project(context, source)

    def project(
        self, context: HumanReviewContext, source: HumanReviewSource
    ) -> HumanReviewDisclosure:
        if source.kind == "user_input":
            if not self._questions or not source.question:
                return HumanReviewDisclosure(status="redacted")
            entries: list[tuple[str, str, str | None]] = [("Question", source.question, None)]
            if source.options:
                entries.append(("Options", "\n".join(source.options), None))
            # Dynamic/unknown question scope needs its own explicit attestation.
            attest = self._attest_questions
        else:
            entries, shows_arguments = self._tool_entries(source)
            attest = self._attest or not shows_arguments
        fields, whole = _fit([(label, text) for label, text, _ in entries])
        # The disclosure holds at most 32; the note still counts what's past them.
        long_fields = tuple(
            HumanReviewLongField(key=key, label=_label(label, 1, 1), characters=len(text))
            for index, (label, text, key) in enumerate(entries)
            if key is not None and index not in whole
        )[:_MAX_FIELDS]
        return HumanReviewDisclosure(
            status="permitted",
            fields=fields,
            long_fields=long_fields,
            sensitive_content="application_attested" if attest else "reject_unknown_scope",
        )

    def project_long_field(
        self, context: HumanReviewContext, source: HumanReviewSource, key: str
    ) -> str | None:
        match = _LONG_FIELD_KEY.match(key) if type(key) is str else None
        if source.kind != "tool_approval" or match is None:
            return None
        call_number, name_number = (int(group) for group in match.groups())
        if call_number > len(source.calls):
            return None
        call = source.calls[call_number - 1]
        names = self._fields.get(call.tool_name, ())
        if name_number > len(names):
            return None
        arguments = source.arguments_by_call.get(call.tool_call_id) or {}
        name = names[name_number - 1]
        return _text(arguments[name]) if name in arguments else None

    def _tool_entries(
        self, source: HumanReviewSource
    ) -> tuple[list[tuple[str, str, str | None]], bool]:
        """(label, text, long field key) for each entry, and whether any shows an argument."""
        entries: list[tuple[str, str, str | None]] = []
        shows_arguments = False
        several = len(source.calls) > 1
        for number, call in enumerate(source.calls, start=1):
            prefix = f"{call.tool_name} (call {number})" if several else call.tool_name
            names = self._fields.get(call.tool_name)
            if names is None:
                entries.append((prefix, _UNLISTED_TOOL_TEXT, None))
                continue
            arguments = source.arguments_by_call.get(call.tool_call_id) or {}
            given = [(index, name) for index, name in enumerate(names, 1) if name in arguments]
            if not given:
                entries.append((prefix, "None of the listed arguments were given.", None))
                continue
            for index, name in given:
                label = f"{prefix}: {name}" if several else name
                entries.append((label, _text(arguments[name]), f"{number}.{index}"))
                shows_arguments = True
        return entries, shows_arguments


def _text(value: object) -> str:
    text = value if type(value) is str else json.dumps(value, indent=2, sort_keys=True)
    return text or "(empty)"


def _label(base: str, part: int, parts: int) -> str:
    suffix = f" ({part}/{parts})" if parts > 1 else ""
    room = _FIELD_LABEL_MAX_CHARS - len(suffix)
    while len(html.escape(base)) > room:
        base = base[: max(len(base) - 8, 1)] if len(base) > 1 else ""
    return (base or "Field") + suffix


def _pieces(text: str) -> list[str]:
    pieces: list[str] = []
    start = escaped_size = 0
    for index, character in enumerate(text):
        width = len(html.escape(character))
        if escaped_size + width > _FIELD_TEXT_MAX_CHARS:
            pieces.append(text[start:index])
            start, escaped_size = index, 0
        escaped_size += width
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _size(fields: list[HumanReviewField]) -> int:
    # The same measure build_review applies to the whole view.
    return len(
        json.dumps([item.model_dump() for item in fields], ensure_ascii=False).encode("utf-8")
    )


def _fit(entries: list[tuple[str, str]]) -> tuple[tuple[HumanReviewField, ...], set[int]]:
    """Split long text across fields and stop, with a note, before the view's limits.

    Also returns the indexes of the entries shown whole in a single field.
    """

    fields: list[HumanReviewField] = []
    whole: set[int] = set()
    budget = _FIELDS_MAX_BYTES - _NOTE_RESERVE_BYTES
    left_out = 0
    for index, (label, text) in enumerate(entries):
        pieces = _pieces(text)
        if len(pieces) == 1 and not left_out:
            whole.add(index)
        for part, piece in enumerate(pieces, start=1):
            if left_out:
                left_out += len(piece)
                continue
            field = HumanReviewField(label=_label(label, part, len(pieces)), text=piece)
            if len(fields) < _MAX_FIELDS - 1 and _size([*fields, field]) <= budget:
                fields.append(field)
                continue
            shown = _longest_prefix_that_fits(fields, field, budget)
            if shown:
                fields.append(HumanReviewField(label=field.label, text=piece[:shown]))
            left_out = len(piece) - shown
            whole.discard(index)
    if left_out:
        fields.append(
            HumanReviewField(
                label=_NOT_SHOWN_LABEL,
                text=(
                    f"{left_out} more characters of the arguments above are not shown. "
                    "The decision covers the complete arguments."
                ),
            )
        )
    return tuple(fields), whole


def _longest_prefix_that_fits(
    fields: list[HumanReviewField], field: HumanReviewField, budget: int
) -> int:
    if len(fields) >= _MAX_FIELDS - 1:
        return 0
    low, high = 0, len(field.text)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = HumanReviewField(label=field.label, text=field.text[:middle])
        if _size([*fields, candidate]) <= budget:
            low = middle
        else:
            high = middle - 1
    return low
