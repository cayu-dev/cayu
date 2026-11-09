"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_MAX_ROWS": ("cayu.connectors.snowflake.toolkit", "DEFAULT_MAX_ROWS"),
    "DEFAULT_SLOT": ("cayu.connectors.snowflake.toolkit", "DEFAULT_SLOT"),
    "DEFAULT_WRITE_SLOT": ("cayu.connectors.snowflake.toolkit", "DEFAULT_WRITE_SLOT"),
    "DEFAULT_TIMEOUT_SECONDS": ("cayu.connectors.snowflake.toolkit", "DEFAULT_TIMEOUT_SECONDS"),
    "MAX_ROWS": ("cayu.connectors.snowflake.toolkit", "MAX_ROWS"),
    "MAX_STATEMENT_CHARACTERS": ("cayu.connectors.snowflake.toolkit", "MAX_STATEMENT_CHARACTERS"),
    "READ_KEYWORDS": ("cayu.connectors.snowflake.sql", "READ_KEYWORDS"),
    "StatementShapeError": ("cayu.connectors.snowflake.sql", "StatementShapeError"),
    "approval_policy": ("cayu.connectors.snowflake.toolkit", "approval_policy"),
    "is_read_statement": ("cayu.connectors.snowflake.sql", "is_read_statement"),
    "review_fields": ("cayu.connectors.snowflake.toolkit", "review_fields"),
    "tools": ("cayu.connectors.snowflake.toolkit", "tools"),
}

PUBLIC_NAMES = sorted(EXPORTS)
