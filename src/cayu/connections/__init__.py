"""Calls to an application's connections: the outside systems it reaches through named slots.

An application names what it needs by slot (``warehouse``, ``crm``) and never holds the
credential behind it when a host serves the slot for it. A ``ConnectionClient`` sends one HTTP
request through a slot and answers the connector's response:

* ``CloudConnectionBroker`` sends it to Cayu Cloud's connection broker, which attaches the
  Organization's credential bound to the slot (``docs/adr/0003-connections-runtime-contract.md``).
  ``from_environment`` configures it from ``CAYU_CONNECTIONS_URL`` and the Agent's own
  ``CAYU_CONNECTIONS_TOKEN``, both of which only Cayu Cloud sets.
* ``LocalConnections`` sends it straight to each slot's own host with headers the application
  gives, for a laptop or a test.

Model slots are the models an application declares by name (``CayuApp.require_model``), which
the Organization binds per Agent in Cayu Cloud: ``ModelSlots`` reads the bindings from the broker
and a new session resolves its agent's slot when it starts.

The connector's status is the answer's, whatever it is; ``ConnectionCallError`` is raised only
when the request couldn't be sent: the slot is unbound, the host refused to send it, or the
connector couldn't be reached.
"""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.connections._exports import EXPORTS as _EXPORTS
from cayu.connections._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
