"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "API_URL": ("cayu.connectors.github.client", "API_URL"),
    "DEFAULT_MAX_PAGES": ("cayu.connectors.github.client", "DEFAULT_MAX_PAGES"),
    "DEFAULT_SLOT": ("cayu.connectors.github.client", "DEFAULT_SLOT"),
    "ErrorFilter": ("cayu.connectors.github._graphql", "ErrorFilter"),
    "GitHubAppConnections": ("cayu.connectors.github.credentials", "GitHubAppConnections"),
    "GitHubAppNotInstalled": ("cayu.connectors.github._errors", "GitHubAppNotInstalled"),
    "GitHubClient": ("cayu.connectors.github.client", "GitHubClient"),
    "GitHubConflict": ("cayu.connectors.github._errors", "GitHubConflict"),
    "GitHubConnectionRefused": ("cayu.connectors.github._errors", "GitHubConnectionRefused"),
    "GitHubError": ("cayu.connectors.github._errors", "GitHubError"),
    "GitHubGraphQLFailed": ("cayu.connectors.github._errors", "GitHubGraphQLFailed"),
    "GitHubNotFound": ("cayu.connectors.github._errors", "GitHubNotFound"),
    "GitHubPermissionDenied": ("cayu.connectors.github._errors", "GitHubPermissionDenied"),
    "GitHubRateLimited": ("cayu.connectors.github._errors", "GitHubRateLimited"),
    "GitHubResponse": ("cayu.connectors.github.client", "GitHubResponse"),
    "GitHubUnavailable": ("cayu.connectors.github._errors", "GitHubUnavailable"),
    "GitHubWriteRefused": ("cayu.connectors.github._errors", "GitHubWriteRefused"),
    "GraphQLAnswer": ("cayu.connectors.github._graphql", "GraphQLAnswer"),
    "GraphQLError": ("cayu.connectors.github._graphql", "GraphQLError"),
    "Page": ("cayu.connectors.github.client", "Page"),
    "REPOSITORY_HEADER": ("cayu.connectors.github.client", "REPOSITORY_HEADER"),
    "RefusedWrite": ("cayu.connectors.github.client", "RefusedWrite"),
    "app_connections": ("cayu.connectors.github.credentials", "app_connections"),
    "connections_from_environment": (
        "cayu.connectors.github.credentials",
        "connections_from_environment",
    ),
    "errors_under": ("cayu.connectors.github._graphql", "errors_under"),
    "forbidden_under": ("cayu.connectors.github._graphql", "forbidden_under"),
    "is_mutation": ("cayu.connectors.github._graphql", "is_mutation"),
    "token_connections": ("cayu.connectors.github.credentials", "token_connections"),
}

PUBLIC_NAMES = sorted(EXPORTS)
