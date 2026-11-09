"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_PAGE_SIZE": ("cayu.connectors.slack.toolkit", "DEFAULT_PAGE_SIZE"),
    "DEFAULT_SLOT": ("cayu.connectors.slack.toolkit", "DEFAULT_SLOT"),
    "MAX_MESSAGE_CHARACTERS": ("cayu.connectors.slack.toolkit", "MAX_MESSAGE_CHARACTERS"),
    "MAX_PAGE_SIZE": ("cayu.connectors.slack.toolkit", "MAX_PAGE_SIZE"),
    "approval_policy": ("cayu.connectors.slack.toolkit", "approval_policy"),
    "review_fields": ("cayu.connectors.slack.toolkit", "review_fields"),
    "tools": ("cayu.connectors.slack.toolkit", "tools"),
}

PUBLIC_NAMES = sorted(EXPORTS)
