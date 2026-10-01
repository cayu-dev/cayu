"""Experimental helpers for virtual-egress adapters maintained outside Cayu.

Status: experimental. These are the same objects Cayu's built-in remote
virtual-egress adapters use; names and signatures may change between releases
until the supported extension contract and compatibility tiers are finalized.

- ``prepare_exposed_proxy_binding`` starts a session proxy listener, exposes it
  to the sandbox through a ``ProxyExposure``, and returns an ``EgressBinding``
  whose teardown revokes grants before releasing the exposure and listener.
- ``run_setup_commands`` runs a request's setup commands through
  ``Runner.exec_system``; ``run_enforcement_preflight`` proves proxy
  reachability and direct-egress denial inside the guest and returns the
  observation time.
- ``virtual_egress_execution_capability_evidence`` builds the execution
  admission evidence for an enforced virtual-egress runner.

See ``docs/build-a-runner.md`` and ``docs/virtual-egress.md``.
"""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.extensions.egress._exports import EXPORTS as _EXPORTS
from cayu.extensions.egress._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
