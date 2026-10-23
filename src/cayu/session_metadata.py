"""Reserved descriptive metadata shared by SDK authors and read projections.

Title and summary are written through typed session APIs. Description belongs
in AgentSpec.metadata. Eval-trial and run-lineage marks are runtime authority;
attention-handoff is a caller-authored opt-out, inherited by descendants.
"""

from __future__ import annotations

from typing import Any, Literal, cast, get_args

TITLE_METADATA_KEY = "cayu:title"
SUMMARY_METADATA_KEY = "cayu:summary"
SUMMARY_TONE_METADATA_KEY = "cayu:summary_tone"
DESCRIPTION_METADATA_KEY = "cayu:description"
EVAL_TRIAL_METADATA_KEY = "cayu:eval_trial"
ATTENTION_HANDOFF_METADATA_KEY = "cayu:attention_handoff"
RUN_LINEAGE_METADATA_KEY = "cayu:run_lineage"
MAX_SESSION_TITLE_LENGTH = 256
MAX_SESSION_SUMMARY_LENGTH = 4096
MAX_AGENT_DESCRIPTION_LENGTH = 1024
SummaryTone = Literal["neutral", "attention"]
SUMMARY_TONES = frozenset(get_args(SummaryTone))
SESSION_DISPLAY_METADATA_KEYS = (
    TITLE_METADATA_KEY,
    SUMMARY_METADATA_KEY,
    SUMMARY_TONE_METADATA_KEY,
)


def validate_display_text(value: object, *, limit: int) -> str:
    """Bound before scanning; never render rejected caller values."""
    if (
        type(value) is not str
        or not 1 <= len(value) <= limit
        or not value.strip()
        or any(
            ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF for char in value
        )
    ):
        raise ValueError("Display metadata must be bounded, nonblank, single-line text.")
    return value


def validate_title(value: object) -> str | None:
    return None if value is None else validate_display_text(value, limit=MAX_SESSION_TITLE_LENGTH)


def summary_metadata(summary: object, tone: object) -> dict[str, Any]:
    text = validate_display_text(summary, limit=MAX_SESSION_SUMMARY_LENGTH)
    if type(tone) is not str or tone not in SUMMARY_TONES:
        raise ValueError("Summary tone must be neutral or attention.")
    return {SUMMARY_METADATA_KEY: text, SUMMARY_TONE_METADATA_KEY: cast("SummaryTone", tone)}


def validate_description(metadata: dict[str, Any]) -> dict[str, Any]:
    if DESCRIPTION_METADATA_KEY in metadata:
        validate_display_text(
            metadata[DESCRIPTION_METADATA_KEY], limit=MAX_AGENT_DESCRIPTION_LENGTH
        )
    return metadata


def reject_session_display_metadata(metadata: dict[str, Any]) -> None:
    if any(key in metadata for key in SESSION_DISPLAY_METADATA_KEYS):
        raise ValueError("Session display metadata requires the title or set_summary API.")


def inherited_display_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata[key] for key in SESSION_DISPLAY_METADATA_KEYS if key in metadata}
