"""Public knowledge previews shared by Inbox and kept-entry reads."""

from collections.abc import Callable

_KNOWLEDGE_PREVIEW_CHARS = 500


def knowledge_preview(text: str, redact: Callable[[str], str]) -> str:
    """Redact complete text before a truncation can split a secret."""
    preview = redact(text).strip()
    if len(preview) > _KNOWLEDGE_PREVIEW_CHARS:
        return preview[: _KNOWLEDGE_PREVIEW_CHARS - 1].rstrip() + "…"
    return preview
