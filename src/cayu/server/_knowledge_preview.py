"""Public knowledge previews shared by Inbox and knowledge listings."""

from collections.abc import Callable

from cayu.knowledge._retrieval_results import _knowledge_text_excerpt as knowledge_excerpt

__all__ = ["INBOX_DETAIL_CHARS", "knowledge_excerpt", "knowledge_preview"]

# What an Inbox knowledge proposal shows of its text.
INBOX_DETAIL_CHARS = 500


def knowledge_preview(text: str, redact: Callable[[str], str]) -> str:
    """An excerpt of ``text``, redacted whole before the cut so it can't split a secret."""
    return knowledge_excerpt(redact(text))
