"""The SAP toolkit: each read's request through a slot, its plain and bounded answer, and SAP's
and the broker's errors as tool errors. Answers are SAP's public sandbox demo data."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from tests.connector_broker_support import (
    Answer,
    FakeBroker,
    Refusal,
    Sent,
    assert_within_budgets,
    context,
)

from cayu.connections import LocalConnections
from cayu.connectors import sap
from cayu.connectors._common import ANSWER_BUDGET_BYTES
from cayu.tools import Tool, ToolEffect, ToolResult

FIXTURES = Path(__file__).parents[1] / "fixtures" / "sap"
ODATA = "/sap/opu/odata/sap"
INVOICE_FILTER = (
    "(CompanyCode eq '1710') and PostingDate ge datetime'2026-10-01T00:00:00' and "
    "PostingDate lt datetime'2026-11-01T00:00:00'"
)


def _recorded(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _tools(broker: FakeBroker, slot: str = "erp") -> dict[str, Tool]:
    return {tool.name: tool for tool in sap.tools(broker.client, slot=slot)}


def _run(tool: Tool, **args: Any) -> ToolResult:
    return asyncio.run(tool.run(context(), args))


def _page(rows: list[dict[str, Any]], count: int | None = None, **extra: Any) -> Answer:
    return Answer(
        body={
            "d": {"results": rows, "__count": str(len(rows) if count is None else count), **extra}
        }
    )


def _structured(result: ToolResult) -> dict[str, Any]:
    assert result.structured is not None
    return result.structured


def test_the_toolkit_has_only_read_tools() -> None:
    tools = sap.tools(LocalConnections({}))
    assert {tool.name: tool.spec.effect for tool in tools} == {
        "sap_read": ToolEffect.NONE,
        "sap_supplier_invoices": ToolEffect.NONE,
        "sap_purchase_order_items": ToolEffect.NONE,
        "sap_goods_receipts": ToolEffect.NONE,
        "sap_suppliers": ToolEffect.NONE,
    }
    assert sap.DEFAULT_SLOT == "sap"
    assert tools[0].spec.input_schema["required"] == ["service", "entity_set", "select"]


def test_sap_read_sends_one_get_with_only_odata_options() -> None:
    broker = FakeBroker(lambda _: _page([]))
    result = _run(
        _tools(broker)["sap_read"],
        service="API_BUSINESS_PARTNER",
        entity_set="A_Supplier",
        select=["Supplier", "SupplierName", "to_SupplierCompany/CompanyCode"],
        filter="Supplier eq 'USSU_V8005'",
        orderby=["Supplier desc", "SupplierName"],
        expand=["to_SupplierCompany"],
        top=10,
        skip=20,
    )

    [sent] = broker.sent
    assert (sent.slot, sent.method, sent.path) == (
        "erp",
        "GET",
        f"{ODATA}/API_BUSINESS_PARTNER/A_Supplier",
    )
    assert sent.query == {
        "$select": "Supplier,SupplierName,to_SupplierCompany/CompanyCode",
        "$filter": "Supplier eq 'USSU_V8005'",
        "$orderby": "Supplier desc,SupplierName",
        "$expand": "to_SupplierCompany",
        "$top": "10",
        "$skip": "20",
        "$inlinecount": "allpages",
        "$format": "json",
    }
    assert "sap-client" not in sent.query
    assert (sent.headers, sent.body) == ({}, b"")
    assert _structured(result) == {
        "service": "API_BUSINESS_PARTNER",
        "entity_set": "A_Supplier",
        "rows": [],
        "count": 0,
        "returned": 0,
        "next_skip": None,
        "truncated": False,
        "truncation_reasons": [],
    }


@pytest.mark.parametrize(
    "args",
    [
        {"service": "API/../X", "entity_set": "A_Supplier", "select": ["Supplier"]},
        {"service": "API_BUSINESS_PARTNER", "entity_set": "$batch", "select": ["Supplier"]},
        {"service": "API_BUSINESS_PARTNER", "entity_set": "A_Supplier", "select": []},
        {"service": "API_BUSINESS_PARTNER", "entity_set": "A_Supplier", "select": ["a,b"]},
        {
            "service": "API_BUSINESS_PARTNER",
            "entity_set": "A_Supplier",
            "select": ["S"],
            "top": 501,
        },
        {
            "service": "API_BUSINESS_PARTNER",
            "entity_set": "A_Supplier",
            "select": ["S"],
            "skip": -1,
        },
        {
            "service": "API_BUSINESS_PARTNER",
            "entity_set": "A_Supplier",
            "select": ["S"],
            "filter": " ",
        },
        {
            "service": "API_BUSINESS_PARTNER",
            "entity_set": "A_Supplier",
            "select": ["S"],
            "orderby": ["S; drop"],
        },
    ],
)
def test_sap_read_refuses_bad_arguments_before_any_call(args: dict[str, Any]) -> None:
    broker = FakeBroker(lambda _: _page([]))
    result = _run(_tools(broker)["sap_read"], **args)

    assert _structured(result)["error"] == "invalid_arguments"
    assert broker.sent == []


def test_supplier_invoices_build_the_recorded_filter_and_trim_the_answer() -> None:
    broker = FakeBroker(lambda _: Answer(body=_recorded("supplier_invoices")))
    result = _run(
        _tools(broker)["sap_supplier_invoices"],
        company_codes=["1710"],
        posting_date_from="2026-10-01",
        posting_date_to="2026-10-31",
        top=25,
    )

    [sent] = broker.sent
    assert sent.path == f"{ODATA}/API_SUPPLIERINVOICE_PROCESS_SRV/A_SupplierInvoice"
    assert sent.query["$filter"] == INVOICE_FILTER
    assert sent.query["$expand"] == "to_SuplrInvcItemPurOrdRef"
    assert sent.query["$orderby"] == "PostingDate desc,SupplierInvoice desc,FiscalYear desc"
    assert sent.query["$select"].startswith("FiscalYear,SupplierInvoice,CompanyCode,")
    assert "to_SuplrInvcItemPurOrdRef/PurchaseOrder" in sent.query["$select"]
    assert (sent.query["$top"], sent.query["$skip"]) == ("25", "0")

    answer = _structured(result)
    assert_within_budgets(result)
    assert answer["count"] == 5397
    assert 0 < answer["returned"] < 25
    assert answer["truncated"] is True
    assert answer["truncation_reasons"] == ["answer_size"]
    assert answer["next_skip"] == answer["returned"]
    first = answer["rows"][0]
    assert (first["SupplierInvoice"], first["PostingDate"], first["DocumentDate"]) == (
        "5109898670",
        "2026-10-31",
        "2026-10-31",
    )
    assert first["InvoiceGrossAmount"] == "27279.84"
    assert first["to_SuplrInvcItemPurOrdRef"][0]["SupplierInvoiceItemAmount"] == "14186.40"
    assert first["to_SuplrInvcItemPurOrdRef"][0]["PurchaseOrder"] == "4500212209"


def test_supplier_invoices_double_quotes_and_bound_the_posting_dates() -> None:
    broker = FakeBroker(lambda _: _page([]))
    tool = _tools(broker)["sap_supplier_invoices"]

    _run(
        tool,
        company_codes=["O'1", "1710", "1710"],
        posting_date_from="2026-01-01",
        posting_date_to="2026-04-03",
    )
    [sent] = broker.sent
    assert sent.query["$filter"] == (
        "(CompanyCode eq '1710' or CompanyCode eq 'O''1') and PostingDate ge "
        "datetime'2026-01-01T00:00:00' and PostingDate lt datetime'2026-04-04T00:00:00'"
    )
    for first, last in (
        ("2026-01-01", "2026-04-04"),
        ("2026-02-02", "2026-02-01"),
        ("2026-1-1", "2026-01-02"),
        ("9999-12-30", "9999-12-31"),
    ):
        refused = _run(tool, company_codes=["1710"], posting_date_from=first, posting_date_to=last)
        assert _structured(refused)["error"] == "invalid_arguments"
    too_many = _run(
        tool,
        company_codes=[str(code) for code in range(21)],
        posting_date_from="2026-01-01",
        posting_date_to="2026-01-01",
    )
    assert _structured(too_many)["error"] == "invalid_arguments"
    assert len(broker.sent) == 1

    _run(tool, company_codes=["1710"], posting_date_from="9999-12-30", posting_date_to="9999-12-30")
    assert broker.sent[1].query["$filter"].endswith("PostingDate lt datetime'9999-12-31T00:00:00'")


def test_purchase_order_items_read_the_recorded_answer() -> None:
    broker = FakeBroker(lambda _: Answer(body=_recorded("purchase_order_items")))
    result = _run(_tools(broker)["sap_purchase_order_items"], purchase_orders=["4500212209"])

    [sent] = broker.sent
    assert sent.path == f"{ODATA}/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrderItem"
    assert sent.query["$filter"] == "(PurchaseOrder eq '4500212209')"
    assert sent.query["$orderby"] == "PurchaseOrder,PurchaseOrderItem"
    answer = _structured(result)
    # The recorded answer was its first page of one row.
    assert (answer["count"], answer["returned"], answer["next_skip"]) == (7, 1, 1)
    assert answer["rows"][0]["NetPriceAmount"] == "102.80"
    assert answer["rows"][0]["InvoiceIsGoodsReceiptBased"] is False


def test_goods_receipts_ask_for_receipts_and_reversals_of_the_orders() -> None:
    broker = FakeBroker(lambda _: Answer(body=_recorded("goods_receipts")))
    result = _run(_tools(broker)["sap_goods_receipts"], purchase_orders=["4500212209"], top=1)

    [sent] = broker.sent
    assert sent.path == f"{ODATA}/API_MATERIAL_DOCUMENT_SRV/A_MaterialDocumentItem"
    assert sent.query["$filter"] == (
        "(GoodsMovementType eq '101' or GoodsMovementType eq '102') and "
        "(PurchaseOrder eq '4500212209')"
    )
    answer = _structured(result)
    assert answer["rows"][0]["QuantityInEntryUnit"] == "138"
    assert (answer["count"], answer["next_skip"]) == (7, 1)


def _orders(count: int) -> list[str]:
    return [f"45{number:08d}" for number in range(count)]


def test_many_values_are_read_in_or_chains_of_20_as_one_list() -> None:
    def connector(sent: Sent) -> Answer:
        orders = sent.query["$filter"].count(" eq ")
        skip, top = int(sent.query["$skip"]), int(sent.query["$top"])
        rows = [{"PurchaseOrder": str(index)} for index in range(orders)][skip : skip + top]
        return _page(rows, orders)

    broker = FakeBroker(connector)
    tool = _tools(broker)["sap_purchase_order_items"]
    orders = _orders(45)

    first = _structured(_run(tool, purchase_orders=[*reversed(orders), orders[0]], top=30))
    chains = [sent.query["$filter"] for sent in broker.sent]
    assert [chain.count(" or ") + 1 for chain in chains] == [20, 20]
    assert chains[0].startswith(f"(PurchaseOrder eq '{orders[0]}' or ")
    assert [sent.query["$top"] for sent in broker.sent] == ["30", "10"]
    assert (first["returned"], first["count"], first["next_skip"]) == (30, None, 30)

    broker.sent.clear()
    last = _structured(_run(tool, purchase_orders=orders, top=30, skip=30))
    assert [(sent.query["$skip"], sent.query["$top"]) for sent in broker.sent] == [
        ("30", "30"),
        ("10", "30"),
        ("0", "20"),
    ]
    assert (last["returned"], last["count"], last["next_skip"]) == (15, 45, None)

    refused = _run(tool, purchase_orders=_orders(101))
    assert _structured(refused)["error"] == "invalid_arguments"


def _partners(sent: Sent, numbers: dict[str, str], paged: frozenset[str] = frozenset()) -> Answer:
    """``A_BusinessPartner`` answering the suppliers ``sent`` asks for that have a partner in
    ``numbers``, each with its recorded bank accounts expanded. In the sandbox a supplier's
    partner has the same number, so the recorded banks are keyed by the supplier."""

    recorded = _recorded("supplier_banks")["d"]["results"]
    partners = []
    for supplier, partner in numbers.items():
        if f"'{supplier}'" not in sent.query["$filter"]:
            continue
        banks = [
            {**bank, "BusinessPartner": partner}
            for bank in recorded
            if bank["BusinessPartner"] == supplier
        ]
        expanded: dict[str, Any] = {"results": banks}
        if supplier in paged:
            expanded["__next"] = "https://sap.example.test/next"
        partners.append(
            {"BusinessPartner": partner, "Supplier": supplier, "to_BusinessPartnerBank": expanded}
        )
    return _page(partners)


def _first_chain(sent: Sent) -> bool:
    return "'4500000000'" in sent.query["$filter"]


def test_a_single_chain_answering_none_of_the_rows_it_counts_ends_the_read() -> None:
    broker = FakeBroker(lambda _: _page([], 5))
    answer = _structured(
        _run(_tools(broker)["sap_purchase_order_items"], purchase_orders=["4500000000"])
    )

    assert len(broker.sent) == 1
    assert (answer["returned"], answer["count"], answer["next_skip"]) == (0, 5, None)
    assert (answer["truncated"], answer["truncation_reasons"]) == (True, ["count_mismatch"])


def test_an_empty_first_chain_is_passed_for_the_next_chains_rows() -> None:
    def connector(sent: Sent) -> Answer:
        if _first_chain(sent):
            return _page([], 5)
        return _page([{"PurchaseOrder": "4500000020"}], 1)

    broker = FakeBroker(connector)
    answer = _structured(
        _run(_tools(broker)["sap_purchase_order_items"], purchase_orders=_orders(21))
    )

    assert [sent.query["$skip"] for sent in broker.sent] == ["0", "0"]
    assert (answer["returned"], answer["count"], answer["next_skip"]) == (1, 6, None)
    assert answer["truncation_reasons"] == ["count_mismatch"]


def test_an_empty_chain_after_rows_ends_the_page_and_the_next_page_passes_it() -> None:
    def connector(sent: Sent) -> Answer:
        if _first_chain(sent):
            rows = [{"PurchaseOrder": "4500000000"}, {"PurchaseOrder": "4500000001"}]
            return _page(rows[int(sent.query["$skip"]) :], 2)
        return _page([], 3)

    broker = FakeBroker(connector)
    tool = _tools(broker)["sap_purchase_order_items"]

    first = _structured(_run(tool, purchase_orders=_orders(21)))
    assert (first["returned"], first["count"], first["next_skip"]) == (2, 5, 2)
    assert first["truncated"] is False

    broker.sent.clear()
    second = _structured(_run(tool, purchase_orders=_orders(21), skip=first["next_skip"]))
    assert [sent.query["$skip"] for sent in broker.sent] == ["2", "0"]
    assert (second["returned"], second["next_skip"]) == (0, None)
    assert second["truncation_reasons"] == ["count_mismatch"]


def _hidden_first_chain(rows: int, hidden: int = 5):
    """Chain 1 counts ``hidden`` rows and answers none; chain 2 holds R0, R1, ..."""

    def connector(sent: Sent) -> Answer:
        if _first_chain(sent):
            return _page([], hidden)
        skip, top = int(sent.query["$skip"]), int(sent.query["$top"])
        held = [{"PurchaseOrder": f"R{index}"} for index in range(rows)]
        return _page(held[skip : skip + top], rows)

    return connector


def _all_pages(broker: FakeBroker, top: int) -> list[list[str]]:
    tool = _tools(broker)["sap_purchase_order_items"]
    pages: list[list[str]] = []
    skip: int | None = 0
    while skip is not None:
        answer = _structured(_run(tool, purchase_orders=_orders(21), top=top, skip=skip))
        pages.append([row["PurchaseOrder"] for row in answer["rows"]])
        assert answer["next_skip"] != skip
        skip = answer["next_skip"]
    return pages


@pytest.mark.parametrize("top", [10, 3])
def test_paging_past_a_hidden_chain_answers_each_row_once(top: int) -> None:
    pages = _all_pages(FakeBroker(_hidden_first_chain(30)), top)

    answered = [row for page in pages for row in page]
    assert answered == [f"R{index}" for index in range(30)]
    assert pages[0] == [f"R{index}" for index in range(top)]


def test_a_page_ended_at_a_hidden_chain_and_cut_for_size_resumes_at_the_first_cut_row() -> None:
    long = "x" * 4_000

    def connector(sent: Sent) -> Answer:
        if _first_chain(sent):
            held = [{"PurchaseOrder": f"A{index}", "Text": long} for index in range(20)]
            skip = int(sent.query["$skip"])
            return _page(held[skip : skip + int(sent.query["$top"])], 20)
        return _page([], 4)

    broker = FakeBroker(connector)
    tool = _tools(broker)["sap_purchase_order_items"]

    first = _structured(_run(tool, purchase_orders=_orders(21), top=50))
    cut = first["returned"]
    assert 0 < cut < 20
    assert first["truncation_reasons"] == ["answer_size"]
    assert first["next_skip"] == cut

    answered = [row["PurchaseOrder"] for row in first["rows"]]
    skip = first["next_skip"]
    while skip is not None:
        page = _structured(_run(tool, purchase_orders=_orders(21), top=50, skip=skip))
        answered += [row["PurchaseOrder"] for row in page["rows"]]
        assert page["next_skip"] != skip
        skip = page["next_skip"]
    assert answered == [f"A{index}" for index in range(20)]


def test_a_paged_expansion_is_marked_incomplete() -> None:
    invoice = {
        "SupplierInvoice": "5109898670",
        "to_SuplrInvcItemPurOrdRef": {
            "results": [{"SupplierInvoiceItem": "1", "PurchaseOrder": "4500212209"}],
            "__next": "https://sap.example.test/next-items",
        },
    }
    whole = {"SupplierInvoice": "5109898671", "to_SuplrInvcItemPurOrdRef": {"results": []}}
    broker = FakeBroker(lambda _: _page([invoice, whole]))
    answer = _structured(
        _run(
            _tools(broker)["sap_supplier_invoices"],
            company_codes=["1710"],
            posting_date_from="2026-10-01",
            posting_date_to="2026-10-31",
        )
    )

    assert answer["rows"] == [
        {
            "SupplierInvoice": "5109898670",
            "to_SuplrInvcItemPurOrdRef": [
                {"SupplierInvoiceItem": "1", "PurchaseOrder": "4500212209"}
            ],
            "incomplete_expansions": ["to_SuplrInvcItemPurOrdRef"],
        },
        {"SupplierInvoice": "5109898671", "to_SuplrInvcItemPurOrdRef": []},
    ]
    assert (answer["truncated"], answer["truncation_reasons"]) == (
        True,
        ["expansion_incomplete"],
    )
    assert answer["next_skip"] is None


def test_suppliers_leave_bank_details_out_unless_asked() -> None:
    broker = FakeBroker(lambda _: Answer(body=_recorded("suppliers")))

    plain = _structured(_run(_tools(broker)["sap_suppliers"], suppliers=["USSU_V8005"]))

    [sent] = broker.sent
    assert sent.path == f"{ODATA}/API_BUSINESS_PARTNER/A_Supplier"
    assert sent.query["$filter"] == "(Supplier eq 'USSU_V8005')"
    assert plain["rows"] == [
        {
            "CreationDate": "2016-10-04",
            "PaymentIsBlockedForSupplier": False,
            "PostingIsBlocked": False,
            "Supplier": "USSU_V8005",
            "SupplierName": "Vendor V8005",
        }
    ]


def test_bank_accounts_are_found_through_each_suppliers_business_partner() -> None:
    numbers = {"USSU_V8000": "1000000", "USSU_V8007": "1000007"}

    def connector(sent: Sent) -> Answer:
        if sent.path.endswith("/A_BusinessPartner"):
            return _partners(sent, numbers)
        return _page([{"Supplier": "USSU_V8000"}, {"Supplier": "USSU_V8007"}])

    broker = FakeBroker(connector)
    answer = _structured(
        _run(
            _tools(broker)["sap_suppliers"],
            suppliers=["USSU_V8000", "USSU_V8007"],
            include_bank=True,
        )
    )

    partners = broker.sent[1]
    assert partners.path == f"{ODATA}/API_BUSINESS_PARTNER/A_BusinessPartner"
    assert partners.query["$filter"] == "(Supplier eq 'USSU_V8000' or Supplier eq 'USSU_V8007')"
    assert partners.query["$expand"] == "to_BusinessPartnerBank"
    assert partners.query["$select"] == (
        "BusinessPartner,Supplier,to_BusinessPartnerBank/BankIdentification,"
        "to_BusinessPartnerBank/BankCountryKey,to_BusinessPartnerBank/BankName,"
        "to_BusinessPartnerBank/ValidityStartDate,to_BusinessPartnerBank/ValidityEndDate"
    )
    for secret in ("BankAccount", "IBAN", "BankNumber", "Holder"):
        assert secret not in partners.query["$select"]
    assert [row["BusinessPartner"] for row in answer["rows"]] == ["1000000", "1000007"]
    assert answer["rows"][1]["banks"] == [
        {
            "BankCountryKey": "US",
            "BankIdentification": "0001",
            "BankName": "JP Morgan Chase",
            "ValidityEndDate": "9999-12-31T23:59:59+00:00",
            "ValidityStartDate": "2018-05-29T00:00:00+00:00",
        }
    ]
    assert (answer["truncated"], answer["truncation_reasons"]) == (False, [])


def test_a_supplier_without_a_partner_or_its_full_banks_is_marked_incomplete() -> None:
    numbers = {"USSU_V8000": "1000000", "USSU_V8001": "1000001"}

    def connector(sent: Sent) -> Answer:
        if sent.path.endswith("/A_BusinessPartner"):
            return _partners(sent, numbers, paged=frozenset({"USSU_V8001"}))
        return _page([{"Supplier": supplier} for supplier in ("USSU_V8000", "USSU_V8001", "X")])

    broker = FakeBroker(connector)
    answer = _structured(
        _run(
            _tools(broker)["sap_suppliers"],
            suppliers=["USSU_V8000", "USSU_V8001", "X"],
            include_bank=True,
        )
    )

    assert [(row["BusinessPartner"], row["banks"] is None) for row in answer["rows"]] == [
        ("1000000", False),
        ("1000001", True),
        (None, True),
    ]
    assert len(answer["rows"][0]["banks"]) == 1
    assert (answer["truncated"], answer["truncation_reasons"]) == (True, ["banks_incomplete"])


def test_rows_are_made_plain() -> None:
    row = {
        "__metadata": {"id": "https://sap.example.test/A_Supplier('1')", "type": "A_SupplierType"},
        "Supplier": "1",
        "Amount": "1234567890.123",
        "Day": "/Date(1793404800000)/",
        "Changed": "/Date(1793449815123)/",
        "Created": "/Date(1793404800000+0120)/",
        "Odd": "/Date(1793404800000+9999)/",
        "Note": "Date(1)",
        "to_Company": {"__deferred": {"uri": "https://sap.example.test/A_Supplier('1')/x"}},
        "to_Items": {"results": [{"__metadata": {"type": "Item"}, "Day": "/Date(0)/"}]},
    }
    broker = FakeBroker(lambda _: _page([row]))
    answer = _structured(
        _run(
            _tools(broker)["sap_read"],
            service="API_BUSINESS_PARTNER",
            entity_set="A_Supplier",
            select=["Supplier"],
        )
    )

    assert answer["rows"] == [
        {
            "Supplier": "1",
            "Amount": "1234567890.123",
            "Day": "2026-10-31",
            "Changed": "2026-10-31T12:30:15.123",
            "Created": "2026-10-31T02:00:00+02:00",
            "Odd": "/Date(1793404800000+9999)/",
            "Note": "Date(1)",
            "to_Items": [{"Day": "1970-01-01"}],
        }
    ]


def _texts(rows: int) -> dict[str, Any]:
    broker = FakeBroker(
        lambda _: _page([{"Id": str(index), "Text": "x" * 1_000} for index in range(rows)])
    )
    result = _run(
        _tools(broker)["sap_read"], service="API_X", entity_set="A_X", select=["Id"], top=rows
    )
    assert_within_budgets(result)
    return {"size": len(result.content.encode()), **_structured(result)}


def test_a_page_near_the_answer_budget_is_answered_whole() -> None:
    answer = _texts(54)

    assert 54 * 1024 < answer["size"] <= ANSWER_BUDGET_BYTES
    assert (answer["returned"], answer["truncated"], answer["next_skip"]) == (54, False, None)


def test_a_page_over_the_answer_budget_is_cut_just_below_it() -> None:
    answer = _texts(80)

    assert ANSWER_BUDGET_BYTES - 1_100 < answer["size"] <= ANSWER_BUDGET_BYTES
    assert answer["truncation_reasons"] == ["answer_size"]
    assert answer["next_skip"] == answer["returned"] < 80


def test_a_row_too_large_to_answer_alone_is_an_error() -> None:
    broker = FakeBroker(lambda _: _page([{"Text": "x" * ANSWER_BUDGET_BYTES}]))
    result = _run(_tools(broker)["sap_read"], service="API_X", entity_set="A_X", select=["Text"])

    assert _structured(result)["error"] == "row_too_large"


def _read(answer: Answer | Refusal) -> dict[str, Any]:
    broker = FakeBroker(lambda _: answer)
    result = _run(
        _tools(broker)["sap_read"],
        service="API_SUPPLIERINVOICE_PROCESS_SRV",
        entity_set="A_SupplierInvoice",
        select=["SupplierInvoice"],
    )
    assert result.is_error
    return _structured(result)


def test_an_odata_v2_error_names_sap_s_code_message_and_transaction() -> None:
    error = _read(
        Answer(
            status=400,
            body={
                "error": {
                    "code": "/IWBEP/CM_MGW_RT/022",
                    "message": {"lang": "en", "value": "Property Nope not found in type."},
                    "innererror": {"transactionid": "5A1B2C3D4E5F"},
                }
            },
        )
    )

    assert error["error"] == "sap_error"
    assert error["message"] == "SAP answered HTTP 400: Property Nope not found in type."
    assert (error["sap_code"], error["transaction_id"], error["status"]) == (
        "/IWBEP/CM_MGW_RT/022",
        "5A1B2C3D4E5F",
        400,
    )
    assert "don't retry unchanged" in error["advice"]


def test_an_odata_v4_error_names_its_message() -> None:
    error = _read(Answer(status=400, body={"error": {"code": "400", "message": "Bad filter."}}))

    assert error["message"] == "SAP answered HTTP 400: Bad filter."


@pytest.mark.parametrize(
    ("status", "advice"),
    [
        (401, "the connection's credential; an admin should check or rotate it"),
        (403, "communication arrangement or role for this service is missing"),
    ],
)
def test_an_authorization_failure_says_what_to_fix(status: int, advice: str) -> None:
    error = _read(
        Answer(
            status=status,
            headers={"content-type": "text/html"},
            body=b"<html><body>Forbidden</body></html>",
        )
    )

    assert error["error"] == "sap_not_authorized"
    assert error["message"] == f"SAP answered HTTP {status}."
    assert advice in error["advice"]
    assert "sap_code" not in error


def test_not_found_and_server_failures_carry_their_status_only_from_xml() -> None:
    xml = b'<?xml version="1.0"?><error><message>Resource not found</message></error>'
    missing = _read(Answer(status=404, headers={"content-type": "application/xml"}, body=xml))
    failed = _read(Answer(status=500, headers={"content-type": "application/xml"}, body=xml))

    assert (missing["error"], missing["message"]) == ("sap_not_found", "SAP answered HTTP 404.")
    assert (failed["error"], failed["status"]) == ("upstream_status", 500)
    assert "retry later" in failed["advice"]


def test_sap_rate_limiting_says_when_to_retry() -> None:
    error = _read(Answer(status=429, headers={"Retry-After": "7"}, body=b""))

    assert error["error"] == "too_many_calls"
    assert error["retry_after_seconds"] == 7
    assert "7 seconds" in error["advice"]


@pytest.mark.parametrize(
    "body",
    [b"<feed/>", {"d": {"results": []}}, {"d": {"results": "x", "__count": "1"}}, {"value": []}],
)
def test_an_answer_that_isnt_an_odata_v2_page_is_an_error(body: Any) -> None:
    assert _read(Answer(body=body))["error"] == "upstream_invalid_response"


@pytest.mark.parametrize(
    ("refusal", "advice"),
    [
        (Refusal(403, "scope_not_allowed"), "isn't granted the SAP service API_SUPPLIERINVOICE"),
        (Refusal(422, "query_not_allowed"), "SAP reads take only $filter"),
        (Refusal(405, "method_not_allowed"), "SAP connections only read"),
        (Refusal(422, "path_not_allowed"), "don't retry"),
        (Refusal(502, "upstream_response_too_large"), "smaller top"),
        (Refusal(404, "connection_unbound"), "An admin must bind the slot erp"),
    ],
)
def test_a_broker_refusal_is_a_tool_error_with_sap_advice(refusal: Refusal, advice: str) -> None:
    error = _read(refusal)

    assert error["error"] == refusal.code
    assert advice in error["advice"]


def test_without_a_client_outside_cloud_the_tools_answer_an_error(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_CONNECTIONS_URL", raising=False)
    [tool, *_] = sap.tools()
    result = _run(tool, service="API_X", entity_set="A_X", select=["X"])

    assert _structured(result)["error"] == "no_connection_client"
