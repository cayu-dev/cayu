"""SAP S/4HANA's OData V2 APIs as read tools, through a connection slot."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from cayu.connections import ConnectionCallError, ConnectionClient, ConnectionResponse
from cayu.connectors import _common
from cayu.connectors.sap import _odata
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec

DEFAULT_SLOT = "sap"
_DEFAULT_TOP = 50
_MAX_TOP = 500
#: The most values (PO numbers, supplier IDs) one named read looks up.
_MAX_VALUES = 100
_MAX_COMPANY_CODES = _odata.CHUNK
#: The longest posting-date range one supplier-invoice read covers, both days included.
_MAX_POSTING_DAYS = 93
_MAX_VALUE_CHARACTERS = 60
_MAX_FILTER_CHARACTERS = 4_000
_MAX_SELECT = 100
_MAX_EXPAND = 10
_MAX_ORDERBY = 10
_MESSAGE_CHARACTERS = 500
_ODATA = "/sap/opu/odata/sap"

READ = "sap_read"
SUPPLIER_INVOICES = "sap_supplier_invoices"
PURCHASE_ORDER_ITEMS = "sap_purchase_order_items"
GOODS_RECEIPTS = "sap_goods_receipts"
SUPPLIERS = "sap_suppliers"

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}")
_PROPERTY_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*")
_ORDER = re.compile(rf"{_PROPERTY_PATH.pattern}(?: (?:asc|desc))?")

_INVOICES = "API_SUPPLIERINVOICE_PROCESS_SRV"
_PURCHASE_ORDERS = "API_PURCHASEORDER_PROCESS_SRV"
_MATERIAL_DOCUMENTS = "API_MATERIAL_DOCUMENT_SRV"
_BUSINESS_PARTNERS = "API_BUSINESS_PARTNER"
_INVOICE_ITEMS = "to_SuplrInvcItemPurOrdRef"
_INVOICE_FIELDS = (
    "FiscalYear",
    "SupplierInvoice",
    "CompanyCode",
    "InvoicingParty",
    "SupplierInvoiceIDByInvcgParty",
    "DocumentDate",
    "PostingDate",
    "InvoiceGrossAmount",
    "DocumentCurrency",
    "PaymentTerms",
    "PaymentBlockingReason",
    "SupplierInvoiceStatus",
    "IsReversed",
    "IsReversal",
    "SupplierInvoiceIsCreditMemo",
    "JournalEntry",
    "JournalEntryFiscalYear",
)
_INVOICE_ITEM_FIELDS = (
    "SupplierInvoiceItem",
    "PurchaseOrder",
    "PurchaseOrderItem",
    "SupplierInvoiceItemAmount",
    "QuantityInPurchaseOrderUnit",
    "PurchaseOrderQuantityUnit",
    "QtyInPurchaseOrderPriceUnit",
    "PurchaseOrderPriceUnit",
    "IsSubsequentDebitCredit",
    "ReferenceDocument",
    "ReferenceDocumentFiscalYear",
    "ReferenceDocumentItem",
)
_PURCHASE_ORDER_ITEM_FIELDS = (
    "PurchaseOrder",
    "PurchaseOrderItem",
    "PurchaseOrderItemText",
    "Material",
    "Plant",
    "OrderQuantity",
    "PurchaseOrderQuantityUnit",
    "NetPriceAmount",
    "NetPriceQuantity",
    "DocumentCurrency",
    "InvoiceIsGoodsReceiptBased",
)
_GOODS_RECEIPT_FIELDS = (
    "MaterialDocumentYear",
    "MaterialDocument",
    "MaterialDocumentItem",
    "GoodsMovementType",
    "PurchaseOrder",
    "PurchaseOrderItem",
    "QuantityInEntryUnit",
    "EntryUnit",
    "Supplier",
    "Plant",
)
#: Goods receipts for a purchase order, and their reversals.
_GOODS_RECEIPT_MOVEMENTS = ("101", "102")
_SUPPLIER_FIELDS = (
    "Supplier",
    "SupplierName",
    "CreationDate",
    "PostingIsBlocked",
    "PaymentIsBlockedForSupplier",
)
#: Which bank and since when; never the account number, IBAN, bank key or account holder.
_BANKS = "to_BusinessPartnerBank"
_BANK_FIELDS = (
    "BankIdentification",
    "BankCountryKey",
    "BankName",
    "ValidityStartDate",
    "ValidityEndDate",
)

_CREDENTIAL_REJECTED = (
    "SAP rejected the connection's credential; an admin should check or rotate it. Don't retry."
)
_NOT_AUTHORIZED = (
    "The communication arrangement or role for this service is missing in SAP; an SAP admin "
    "must grant it. Don't retry."
)
_ADVICE = {
    "query_not_allowed": (
        "SAP reads take only $filter, $select, $expand, $orderby, $top, $skip, $skiptoken, "
        "$inlinecount and $format; don't retry unchanged."
    ),
    "method_not_allowed": "SAP connections only read; don't retry.",
    "upstream_response_too_large": (
        "SAP's answer was too large; ask for fewer rows with a smaller top, or for fewer "
        "select fields or expansions."
    ),
}


def tools(client: ConnectionClient | None = None, slot: str = DEFAULT_SLOT) -> list[Tool]:
    """The SAP read tools on ``slot``, calling through ``client`` (default: Cayu Cloud's
    broker). Every tool only reads; the services an Agent may read are its binding's scopes."""

    resolved = _common.resolved_client(client)
    kinds: tuple[type[_SapTool], ...] = (
        _Read,
        _SupplierInvoices,
        _PurchaseOrderItems,
        _GoodsReceipts,
        _Suppliers,
    )
    return [kind(resolved, slot) for kind in kinds]


class _SapError(Exception):
    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.content)
        self.result = result


class _InvalidArguments(ValueError):
    pass


@dataclass(frozen=True)
class _Query:
    """One read of an entity set, its filter split in parts each SAP answers on its own."""

    service: str
    entity_set: str
    select: Sequence[str]
    filters: Sequence[str | None] = (None,)
    orderby: Sequence[str] = ()
    expand: Sequence[str] = ()

    def options(self, filter_: str | None, top: int, skip: int) -> dict[str, str]:
        options = {
            "$select": ",".join(self.select),
            "$top": str(top),
            "$skip": str(skip),
            "$inlinecount": "allpages",
            "$format": "json",
        }
        if filter_ is not None:
            options["$filter"] = filter_
        if self.orderby:
            options["$orderby"] = ",".join(self.orderby)
        if self.expand:
            options["$expand"] = ",".join(self.expand)
        return options


@dataclass
class _Page:
    rows: list[dict[str, Any]]
    #: Each row's place in the list of every part's counted rows, which ``skip`` counts in.
    positions: list[int]
    #: Where the next page starts when every row is answered.
    end: int
    count: int | None
    more: bool
    reasons: list[str] = field(default_factory=list)


class _SapTool(Tool):
    def __init__(self, client: ConnectionClient | None, slot: str) -> None:
        super().__init__()
        self._client = client
        self._slot = slot

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            top, skip = _paging(args)
            query = self._query(args)
            page = await self._page(query, top, skip)
            await self._extend(args, page)
        except _SapError as error:
            return error.result
        except _InvalidArguments as error:
            return _common.invalid_arguments(str(error))
        return _fitted(query, page)

    def _query(self, args: Mapping[str, Any]) -> _Query:
        raise NotImplementedError

    async def _extend(self, args: Mapping[str, Any], page: _Page) -> None:
        """Add to the page's rows what a further read answers; most tools add nothing."""

    async def _page(self, query: _Query, top: int, skip: int) -> _Page:
        """Up to ``top`` rows from ``skip`` on, across the query's filter parts in turn: a
        part's counted rows follow the previous part's, so ``skip`` pages through them all as
        one list. The count is the total only once every part was read.

        A part that answers none of the rows it counts (SAP can count rows its authorisations
        then hide) ends a page that already holds rows, so the next call starts at that part;
        a page with none yet passes it with the reason ``count_mismatch``. Either way the next
        page starts after every position answered or passed, so no row comes twice."""

        rows: list[dict[str, Any]] = []
        positions: list[int] = []
        reasons: list[str] = []
        total = 0
        base = 0
        parts = query.filters
        for index, part in enumerate(parts):
            offset = max(skip - base, 0)
            found, count, incomplete = _rows(
                await self._get(query, query.options(part, top - len(rows), offset))
            )
            total += count
            start = base
            base += count
            if offset >= count:
                continue
            later = index + 1 < len(parts)
            if not found:
                if rows:
                    known = None if later else total
                    return _Page(rows, positions, start + offset, known, True, reasons)
                _reason(reasons, "count_mismatch")
                continue
            if incomplete:
                _reason(reasons, "expansion_incomplete")
            rows.extend(found)
            positions.extend(range(start + offset, start + offset + len(found)))
            reached = offset + len(found)
            # Full, or SAP paged its answer itself: the next call reads on from here.
            if len(rows) >= top or reached < count:
                known = None if later else total
                more = reached < count or later
                return _Page(rows, positions, start + reached, known, more, reasons)
        return _Page(rows, positions, base, total, False, reasons)

    async def _get(self, query: _Query, options: Mapping[str, str]) -> dict[str, Any]:
        if self._client is None:
            raise _SapError(_common.no_client_result())
        path = f"{_ODATA}/{query.service}/{query.entity_set}"
        try:
            response = await self._client.request(self._slot, "GET", path, query=options)
        except ConnectionCallError as error:
            raise _SapError(_refused(error, self._slot, query.service)) from None
        return _answer(response)


def _refused(error: ConnectionCallError, slot: str, service: str) -> ToolResult:
    return _common.call_error_result(
        error,
        slot,
        {
            **_ADVICE,
            "scope_not_allowed": (
                f"This Agent isn't granted the SAP service {service} on the slot {{slot}}; an "
                "admin must add it to the binding's scopes. Don't retry."
            ),
        },
    )


def _answer(response: ConnectionResponse) -> dict[str, Any]:
    """SAP's ``d`` object of a successful read; otherwise a tool error."""

    if response.status != 200:
        raise _SapError(_failure(response))
    answer = _common.json_body(response)
    body = answer.get("d") if answer is not None else None
    if not isinstance(body, dict):
        raise _SapError(
            _common.error_result(
                "upstream_invalid_response",
                "SAP's answer isn't an OData V2 JSON answer.",
                _common.TEMPORARY,
            )
        )
    return body


def _rows(answer: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int, bool]:
    """The answer's plain rows, its count, and whether any row's expanded list is only its
    first part (that row names it in ``incomplete_expansions``)."""

    results = answer.get("results")
    count = answer.get("__count")
    try:
        total = int(count) if isinstance(count, str | int) else -1
    except ValueError:
        total = -1
    if not isinstance(results, list) or total < 0:
        raise _SapError(
            _common.error_result(
                "upstream_invalid_response",
                "SAP's answer has no list of results with their count.",
                _common.TEMPORARY,
            )
        )
    rows = []
    incomplete = False
    for result in results:
        if not isinstance(result, dict):
            continue
        paged: set[str] = set()
        row = _odata.plain(result, paged)
        if paged:
            row["incomplete_expansions"] = sorted(paged)
            incomplete = True
        rows.append(row)
    return rows, total, incomplete


def _failure(response: ConnectionResponse) -> ToolResult:
    """SAP's error answer as a tool error: its own message where it sent JSON, else its
    status only."""

    status = response.status
    if status == 429:
        return _common.connector_rate_limited(response, "SAP")
    sap_code, said, transaction = _error_detail(response)
    message = f"SAP answered HTTP {status}" + (f": {said}" if said else ".")
    details = {"status": status, "sap_code": sap_code, "transaction_id": transaction}
    if status == 401:
        return _common.error_result("sap_not_authorized", message, _CREDENTIAL_REJECTED, **details)
    if status == 403:
        return _common.error_result("sap_not_authorized", message, _NOT_AUTHORIZED, **details)
    if status == 404:
        return _common.error_result(
            "sap_not_found",
            message,
            "Check the service, the entity set and the field names; don't retry unchanged.",
            **details,
        )
    if status >= 500:
        return _common.error_result("upstream_status", message, _common.TEMPORARY, **details)
    return _common.error_result(
        "sap_error",
        message,
        "Fix the read's fields, filter or order; don't retry unchanged.",
        **details,
    )


def _error_detail(response: ConnectionResponse) -> tuple[str | None, str | None, str | None]:
    """SAP's error code, message and transaction ID from an OData V2 or V4 JSON error; none
    from an HTML or XML page."""

    answer = _common.json_body(response)
    error = answer.get("error") if answer is not None else None
    if not isinstance(error, dict):
        return None, None, None
    code = error.get("code")
    message = error.get("message")
    if isinstance(message, dict):
        message = message.get("value")
    inner = error.get("innererror")
    transaction = inner.get("transactionid") if isinstance(inner, dict) else None
    return (
        code if isinstance(code, str) and code else None,
        _common.shortened(message, _MESSAGE_CHARACTERS)[0] if isinstance(message, str) else None,
        transaction if isinstance(transaction, str) and transaction else None,
    )


def _fitted(query: _Query, page: _Page) -> ToolResult:
    """The page with as many of its rows as fit in the tool's budget."""

    def built(kept: int) -> ToolResult:
        reasons = [*page.reasons, *(["answer_size"] if kept < len(page.rows) else [])]
        if kept < len(page.rows):
            next_skip: int | None = page.positions[kept]
        else:
            next_skip = page.end if page.more else None
        return _common.answer_result(
            {
                "service": query.service,
                "entity_set": query.entity_set,
                "rows": page.rows[:kept],
                "count": page.count,
                "returned": kept,
                "next_skip": next_skip,
                "truncated": bool(reasons),
                "truncation_reasons": reasons,
            }
        )

    result = built(len(page.rows))
    if _common.answer_fits(result):
        return result
    low, high = 0, len(page.rows) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if _common.answer_fits(built(middle)):
            low = middle
        else:
            high = middle - 1
    if low == 0:
        return _common.error_result(
            "row_too_large",
            "One row alone is too large to answer.",
            "Select fewer fields or expand less.",
        )
    return built(low)


def _reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _paging(args: Mapping[str, Any]) -> tuple[int, int]:
    top = args.get("top", _DEFAULT_TOP)
    if type(top) is not int or not 1 <= top <= _MAX_TOP:
        raise _InvalidArguments(f"top must be an integer from 1 to {_MAX_TOP}.")
    skip = args.get("skip", 0)
    if type(skip) is not int or skip < 0:
        raise _InvalidArguments("skip must be an integer of at least 0.")
    return top, skip


def _name(args: Mapping[str, Any], argument: str) -> str:
    value = args.get(argument)
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise _InvalidArguments(f"{argument} must be an SAP name such as API_BUSINESS_PARTNER.")
    return value


def _names(
    args: Mapping[str, Any], argument: str, limit: int, pattern: re.Pattern[str]
) -> list[str]:
    value = args.get(argument, [])
    if (
        not isinstance(value, list)
        or len(value) > limit
        or not all(isinstance(item, str) and pattern.fullmatch(item) for item in value)
    ):
        raise _InvalidArguments(
            f"{argument} must be a list of at most {limit} OData property names or paths."
        )
    return value


def _values(args: Mapping[str, Any], argument: str, limit: int) -> list[str]:
    """Distinct, sorted values to look up, so the same values page the same way."""

    value = args.get(argument)
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= limit
        or not all(
            isinstance(item, str)
            and item.strip()
            and len(item) <= _MAX_VALUE_CHARACTERS
            and item.isprintable()
            for item in value
        )
    ):
        raise _InvalidArguments(
            f"{argument} must be a list of 1 to {limit} nonblank values of at most "
            f"{_MAX_VALUE_CHARACTERS} characters."
        )
    return sorted({item for item in value if isinstance(item, str)})


def _day(args: Mapping[str, Any], argument: str) -> date:
    value = args.get(argument)
    if isinstance(value, str) and len(value) == 10:
        try:
            day = date.fromisoformat(value)
        except ValueError:
            pass
        else:
            if day < date.max:
                return day
    raise _InvalidArguments(f"{argument} must be a date, YYYY-MM-DD, before 9999-12-31.")


def _spec(name: str, description: str, properties: dict[str, Any], required: list[str]) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {**properties, **_PAGE},
            "required": required,
            "additionalProperties": False,
        },
        max_terminal_payload_bytes=_common.MAX_TERMINAL_PAYLOAD_BYTES,
        effect=ToolEffect.NONE,
    )


_PAGE = {
    "top": {
        "type": "integer",
        "minimum": 1,
        "maximum": _MAX_TOP,
        "description": f"How many rows to return (default {_DEFAULT_TOP}).",
    },
    "skip": {
        "type": "integer",
        "minimum": 0,
        "description": "How many rows to skip: the next_skip of the previous page.",
    },
}


def _value_list(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "maxLength": _MAX_VALUE_CHARACTERS},
        "minItems": 1,
        "maxItems": _MAX_VALUES,
        "description": description,
    }


class _Read(_SapTool):
    spec = _spec(
        READ,
        "Read rows of an entity set of a granted SAP OData V2 service (such as "
        "API_BUSINESS_PARTNER, A_Supplier). Name the fields with select, and narrow with an "
        "OData $filter, for example \"CompanyCode eq '1710' and PostingDate ge "
        "datetime'2026-10-01T00:00:00'\". Answers the rows, their count and next_skip.",
        {
            "service": {"type": "string", "description": "The OData service's name."},
            "entity_set": {"type": "string", "description": "The entity set, such as A_Supplier."},
            "select": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": _MAX_SELECT,
                "description": "The fields to answer; Navigation/Field for an expanded one.",
            },
            "filter": {
                "type": "string",
                "maxLength": _MAX_FILTER_CHARACTERS,
                "description": "An OData V2 $filter expression.",
            },
            "orderby": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": _MAX_ORDERBY,
                "description": "Fields to order by, each optionally followed by asc or desc.",
            },
            "expand": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": _MAX_EXPAND,
                "description": "Navigation properties to expand.",
            },
        },
        ["service", "entity_set", "select"],
    )

    def _query(self, args: Mapping[str, Any]) -> _Query:
        select = _names(args, "select", _MAX_SELECT, _PROPERTY_PATH)
        if not select:
            raise _InvalidArguments("select must name at least one field.")
        filter_ = args.get("filter")
        if filter_ is not None and (
            not isinstance(filter_, str)
            or not filter_.strip()
            or len(filter_) > _MAX_FILTER_CHARACTERS
        ):
            raise _InvalidArguments(
                f"filter must be a nonblank string of at most {_MAX_FILTER_CHARACTERS} characters."
            )
        return _Query(
            service=_name(args, "service"),
            entity_set=_name(args, "entity_set"),
            select=select,
            filters=(filter_,),
            orderby=_names(args, "orderby", _MAX_ORDERBY, _ORDER),
            expand=_names(args, "expand", _MAX_EXPAND, _PROPERTY_PATH),
        )


class _SupplierInvoices(_SapTool):
    spec = _spec(
        SUPPLIER_INVOICES,
        "Supplier invoices of company codes posted in a date range of at most "
        f"{_MAX_POSTING_DAYS} days, most recent first, each with its purchase-order references "
        f"({_INVOICE_ITEMS}).",
        {
            "company_codes": {
                **_value_list("Company codes, such as 1710."),
                "maxItems": _MAX_COMPANY_CODES,
            },
            "posting_date_from": {"type": "string", "description": "First day, YYYY-MM-DD."},
            "posting_date_to": {"type": "string", "description": "Last day, YYYY-MM-DD."},
        },
        ["company_codes", "posting_date_from", "posting_date_to"],
    )

    def _query(self, args: Mapping[str, Any]) -> _Query:
        codes = _values(args, "company_codes", _MAX_COMPANY_CODES)
        first = _day(args, "posting_date_from")
        last = _day(args, "posting_date_to")
        if last < first:
            raise _InvalidArguments("posting_date_to is before posting_date_from.")
        if (last - first).days + 1 > _MAX_POSTING_DAYS:
            raise _InvalidArguments(
                f"The posting dates span at most {_MAX_POSTING_DAYS} days; read a longer "
                "period in parts."
            )
        filter_ = " and ".join(
            (
                _odata.any_of("CompanyCode", codes),
                _odata.on_or_after("PostingDate", first),
                _odata.before("PostingDate", last + timedelta(days=1)),
            )
        )
        return _Query(
            service=_INVOICES,
            entity_set="A_SupplierInvoice",
            select=(
                *_INVOICE_FIELDS,
                *(f"{_INVOICE_ITEMS}/{name}" for name in _INVOICE_ITEM_FIELDS),
            ),
            filters=(filter_,),
            orderby=("PostingDate desc", "SupplierInvoice desc", "FiscalYear desc"),
            expand=(_INVOICE_ITEMS,),
        )


class _PurchaseOrderItems(_SapTool):
    spec = _spec(
        PURCHASE_ORDER_ITEMS,
        "The items of purchase orders: material, plant, quantity, net price and whether "
        "invoicing follows the goods receipt.",
        {"purchase_orders": _value_list("Purchase order numbers, such as 4500212209.")},
        ["purchase_orders"],
    )

    def _query(self, args: Mapping[str, Any]) -> _Query:
        orders = _values(args, "purchase_orders", _MAX_VALUES)
        return _Query(
            service=_PURCHASE_ORDERS,
            entity_set="A_PurchaseOrderItem",
            select=_PURCHASE_ORDER_ITEM_FIELDS,
            filters=[_odata.any_of("PurchaseOrder", part) for part in _odata.chunked(orders)],
            orderby=("PurchaseOrder", "PurchaseOrderItem"),
        )


class _GoodsReceipts(_SapTool):
    spec = _spec(
        GOODS_RECEIPTS,
        "Goods receipts for purchase orders (movement type 101) and their reversals (102), as "
        "material document items.",
        {"purchase_orders": _value_list("Purchase order numbers, such as 4500212209.")},
        ["purchase_orders"],
    )

    def _query(self, args: Mapping[str, Any]) -> _Query:
        orders = _values(args, "purchase_orders", _MAX_VALUES)
        movements = _odata.any_of("GoodsMovementType", _GOODS_RECEIPT_MOVEMENTS)
        return _Query(
            service=_MATERIAL_DOCUMENTS,
            entity_set="A_MaterialDocumentItem",
            select=_GOODS_RECEIPT_FIELDS,
            filters=[
                f"{movements} and {_odata.any_of('PurchaseOrder', part)}"
                for part in _odata.chunked(orders)
            ],
            orderby=("MaterialDocumentYear", "MaterialDocument", "MaterialDocumentItem"),
        )


class _Suppliers(_SapTool):
    spec = _spec(
        SUPPLIERS,
        "Suppliers by ID: name, creation date and posting and payment blocks. With "
        "include_bank, also each one's bank accounts: which bank and since when, never the "
        "account number.",
        {
            "suppliers": _value_list("Supplier IDs, such as USSU_V8005."),
            "include_bank": {"type": "boolean", "description": "Add bank accounts (default no)."},
        },
        ["suppliers"],
    )

    def _query(self, args: Mapping[str, Any]) -> _Query:
        suppliers = _values(args, "suppliers", _MAX_VALUES)
        if not isinstance(args.get("include_bank", False), bool):
            raise _InvalidArguments("include_bank must be true or false.")
        return _Query(
            service=_BUSINESS_PARTNERS,
            entity_set="A_Supplier",
            select=_SUPPLIER_FIELDS,
            filters=[_odata.any_of("Supplier", part) for part in _odata.chunked(suppliers)],
            orderby=("Supplier",),
        )

    async def _extend(self, args: Mapping[str, Any], page: _Page) -> None:
        """Each supplier's bank accounts, through its business partner: a supplier's number
        and its partner's often differ. A supplier whose partner or banks can't be read in
        full gets ``banks: null`` and the reason ``banks_incomplete``, never an empty list."""

        if args.get("include_bank") is not True or not page.rows:
            return
        ids = sorted({row["Supplier"] for row in page.rows if isinstance(row.get("Supplier"), str)})
        partners = _Query(
            service=_BUSINESS_PARTNERS,
            entity_set="A_BusinessPartner",
            select=("BusinessPartner", "Supplier", *(f"{_BANKS}/{name}" for name in _BANK_FIELDS)),
            orderby=("BusinessPartner",),
            expand=(_BANKS,),
        )
        found: dict[str, tuple[str, list[dict[str, Any]] | None]] = {}
        for part in _odata.chunked(ids):
            answer = await self._get(
                partners, partners.options(_odata.any_of("Supplier", part), len(part), 0)
            )
            for partner in answer.get("results") or []:
                if isinstance(partner, dict) and isinstance(partner.get("Supplier"), str):
                    found[partner["Supplier"]] = (
                        str(partner.get("BusinessPartner")),
                        _bank_accounts(partner.get(_BANKS)),
                    )
        for row in page.rows:
            partner, banks = found.get(str(row.get("Supplier")), (None, None))
            row["BusinessPartner"] = partner
            row["banks"] = banks
            if banks is None:
                _reason(page.reasons, "banks_incomplete")


def _bank_accounts(expanded: Any) -> list[dict[str, Any]] | None:
    """A partner's expanded bank accounts; None when SAP didn't answer them whole."""

    if not isinstance(expanded, dict) or expanded.get("__next"):
        return None
    results = expanded.get("results")
    if not isinstance(results, list):
        return None
    banks = [_odata.plain(bank) for bank in results if isinstance(bank, dict)]
    for bank in banks:
        bank.pop("BusinessPartner", None)
    return banks
