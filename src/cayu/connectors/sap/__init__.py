"""SAP S/4HANA read tools for an application's ``sap`` connection slot, over OData V2.

``tools(client=None, slot="sap")`` answers ``sap_read``, the rows of an entity set of a granted
service, and the named reads ``sap_supplier_invoices``, ``sap_purchase_order_items``,
``sap_goods_receipts`` and ``sap_suppliers``, which build their filters from typed arguments.
Every tool only reads; the services an Agent may read are its binding's scopes. See
``docs/connector-toolkits.md``.
"""

from typing import Any as _Any

from cayu._api import resolve_export as _resolve_export
from cayu._api import wildcard_names as _wildcard_names
from cayu.connectors.sap._exports import EXPORTS as _EXPORTS
from cayu.connectors.sap._exports import PUBLIC_NAMES as _PUBLIC_NAMES

__all__ = _wildcard_names(_PUBLIC_NAMES, _EXPORTS)


def __getattr__(name: str) -> _Any:
    return _resolve_export(name, globals(), _EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
