"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_SLOT": ("cayu.connectors.sap.toolkit", "DEFAULT_SLOT"),
    "tools": ("cayu.connectors.sap.toolkit", "tools"),
}

PUBLIC_NAMES = sorted(EXPORTS)
