"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_PROXY_SERVER_FACTORY": ("cayu.egress._remote_adapter", "DEFAULT_PROXY_SERVER_FACTORY"),
    "DEFAULT_REMOTE_SETUP_COMMAND_TIMEOUT_SECONDS": (
        "cayu.egress._remote_adapter",
        "DEFAULT_REMOTE_SETUP_COMMAND_TIMEOUT_SECONDS",
    ),
    "ExposedProxy": ("cayu.egress.proxy_exposure", "ExposedProxy"),
    "ProxyExposure": ("cayu.egress.proxy_exposure", "ProxyExposure"),
    "ProxyServerFactory": ("cayu.egress._remote_adapter", "ProxyServerFactory"),
    "prepare_exposed_proxy_binding": (
        "cayu.egress._remote_adapter",
        "prepare_exposed_proxy_binding",
    ),
    "run_enforcement_preflight": ("cayu.egress._remote_adapter", "run_enforcement_preflight"),
    "run_setup_commands": ("cayu.egress._remote_adapter", "run_setup_commands"),
    "virtual_egress_execution_capability_evidence": (
        "cayu.egress.adapter",
        "virtual_egress_execution_capability_evidence",
    ),
}

PUBLIC_NAMES = sorted(EXPORTS)
