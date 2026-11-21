"""A Release declares its connection slots and publishes them in its registration and manifest."""

from __future__ import annotations

import asyncio
import json
import logging
import warnings

import httpx
import pytest
from pydantic import ValidationError
from tests.core.test_cloud_inventory import RELEASE, SECRET, _Cloud, _config, _usage_app

from cayu import AgentSpec, CayuApp, cloud_inventory
from cayu.cloud_inventory import CloudInventoryPublisher
from cayu.connections import ConnectionSlot, LocalConnections
from cayu.connectors import github, sap, slack, snowflake
from cayu.managed_settings import model_setting
from cayu.runtime.manifest import AppManifest
from cayu.vaults.redaction import SecretRedactor

_SAP = {
    "name": "erp",
    "connectors": ["sap", "sap-certificate", "sap-sandbox"],
    "scopes": ["API_BUSINESS_PARTNER.read", "API_SUPPLIERINVOICE_PROCESS_SRV.read"],
    "description": "Supplier invoices to reconcile",
}
_ANY = {"name": "crm", "connectors": [], "scopes": []}
_PARTS = {"connection_slots", "model_slots", "settings_schema"}


async def _declaring_app():
    app = await _usage_app()
    app.require_model("primary", default="bedrock/claude-sonnet-4-5")
    app.require_connection(
        "erp",
        connectors=("sap-sandbox", "sap", "sap-certificate"),
        scopes=["API_SUPPLIERINVOICE_PROCESS_SRV.read", "API_BUSINESS_PARTNER.read"],
        description="Supplier invoices to reconcile",
    )
    app.require_connection("crm")
    return app


def _bodies(cloud: _Cloud) -> list[dict]:
    return [json.loads(request.content) for request in cloud.requests if request.method == "PUT"]


def _refused(code: str | None, message: str = "refused", **detail) -> httpx.Response:
    error: dict = {"message": message, "detail": detail}
    if code is not None:
        error["code"] = code
    return httpx.Response(422, json={"error": error})


def test_a_declaration_is_published_sorted_in_registration_and_manifest():
    async def scenario():
        app = await _declaring_app()
        assert [slot.name for slot in app.connection_slots()] == ["crm", "erp"]
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        (body,) = _bodies(cloud)
        assert body["connection_slots"] == [_ANY, _SAP]
        manifest = app.describe()
        assert [slot.model_dump(mode="json") for slot in manifest.connection_slots] == [_ANY, _SAP]
        await app.aclose()

    asyncio.run(scenario())


def test_a_release_without_connection_slots_leaves_the_field_out():
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        assert "connection_slots" not in _bodies(cloud)[0]
        assert "connection_slots" not in app.describe().model_dump(mode="json")
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"slot": "ERP"}, "lowercase"),
        ({"slot": "erp", "connectors": ["SAP"]}, "connector id"),
        ({"slot": "erp", "connectors": ["sap", "sap"]}, "each connector once"),
        ({"slot": "erp", "connectors": [f"c{index}" for index in range(17)]}, "at most 16"),
        ({"slot": "erp", "scopes": ["read all"]}, "without spaces"),
        ({"slot": "erp", "scopes": ["a", "a"]}, "each scope once"),
        ({"slot": "erp", "description": "two\nlines"}, "one non-blank line"),
        ({"slot": "erp", "description": "x" * 201}, "at most 200"),
    ],
)
def test_a_declaration_that_breaks_a_rule_is_refused(arguments, message):
    async def scenario():
        app = await _usage_app()
        slot = arguments.pop("slot")
        with pytest.raises(ValidationError, match=message):
            app.require_connection(slot, **arguments)
        assert app.connection_slots() == ()
        await app.aclose()

    asyncio.run(scenario())


def test_a_slot_is_declared_once_across_connection_and_model_slots():
    async def scenario():
        app = await _usage_app()
        app.require_connection("erp")
        with pytest.raises(ValueError, match="declared more than once"):
            app.require_connection("erp")
        with pytest.raises(ValueError, match="already declared as a connection slot"):
            app.require_model("erp", default=None)
        app.require_model("primary", default=None)
        with pytest.raises(ValueError, match="already declared as a model slot"):
            app.require_connection("primary")
        with pytest.raises(TypeError, match="not one string"):
            app.require_connection("crm", connectors="sap")
        assert [slot.name for slot in app.connection_slots()] == ["erp"]
        await app.aclose()

    asyncio.run(scenario())


def test_a_declaration_holding_a_secret_is_refused():
    app = CayuApp(secret_redactor=SecretRedactor(SECRET), enable_logging=False)
    with pytest.raises(ValueError, match="secret value"):
        app.require_connection("crm", description=f"Reads {SECRET}")
    assert app.connection_slots() == ()


def test_the_slot_type_is_frozen_and_lists_its_declaration():
    slot = ConnectionSlot(name="slack", connectors=("slack",))
    assert slot.declaration() == {"name": "slack", "connectors": ["slack"], "scopes": []}
    with pytest.raises(ValidationError):
        slot.name = "other"


@pytest.mark.parametrize(
    ("refusal", "kept"),
    [
        # A Cloud from before connection slots refuses the whole registration for the field.
        (
            _refused(
                "registration_invalid",
                "`registration` has an unknown field `connection_slots`.",
                field="registration",
            ),
            {"model_slots", "settings_schema"},
        ),
        # Cloud names the connection slots: only they are left out.
        (
            _refused("connection_slots_invalid", field="connection_slots[0].scopes"),
            {"model_slots", "settings_schema"},
        ),
        # Cloud refused the model slots and names the connection slots in `also`.
        (
            _refused(
                "model_slots_invalid",
                field="model_slots",
                also=[{"code": "connection_slots_invalid", "message": "no", "field": "x"}],
            ),
            {"settings_schema"},
        ),
        # An unnamed refusal leaves out the newest part first.
        (_refused(None), {"model_slots", "settings_schema"}),
        # A registration_invalid on another field is never about the connection slots.
        (
            _refused("registration_invalid", field="tools[0].name"),
            {"connection_slots", "model_slots"},
        ),
    ],
)
def test_a_refused_part_is_left_out_of_every_later_registration(refusal, kept, caplog):
    async def scenario():
        app = await _declaring_app()
        cloud = _Cloud()
        cloud.next_errors.append(refusal)
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            with caplog.at_level(logging.WARNING, logger="cayu.cloud_inventory"):
                await publisher.publish_once()
            first, second = _bodies(cloud)
            assert set(first) >= _PARTS
            assert _PARTS & set(second) == kept
            assert cloud.current == RELEASE.id
            await publisher.register()
            assert _PARTS & set(_bodies(cloud)[-1]) == kept
        await app.aclose()

    asyncio.run(scenario())


def test_a_body_over_clouds_limit_leaves_out_the_connection_slots_first(monkeypatch, caplog):
    async def scenario():
        app = await _declaring_app()
        for index in range(40):
            app.require_connection(
                f"slot-{index}",
                connectors=[f"connector-{number}" for number in range(16)],
                scopes=[f"scope-{number}.read" for number in range(64)],
                description="A long declaration " * 10,
            )
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        (whole,) = _bodies(cloud)
        declarations = len(json.dumps(whole["connection_slots"]))
        size = len(cloud.requests[0].content)
        monkeypatch.setattr(cloud_inventory, "MAX_BODY_BYTES", size - declarations // 2)
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            with caplog.at_level(logging.WARNING, logger="cayu.cloud_inventory"):
                await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        (fitted,) = _bodies(cloud)
        assert "connection_slots" not in fitted
        assert fitted["model_slots"] == whole["model_slots"]
        assert fitted["tools"] == whole["tools"]
        assert fitted["knowledge"] == whole["knowledge"]
        assert "without its connection slots" in caplog.text
        await app.aclose()

    asyncio.run(scenario())


def test_a_schema_18_manifest_still_loads_and_declares_no_connection_slots():
    async def scenario():
        app = await _declaring_app()
        recorded = app.describe().model_dump(mode="json")
        await app.aclose()
        assert recorded["schema_version"] == "19"
        older = {key: value for key, value in recorded.items() if key != "connection_slots"}
        assert AppManifest.model_validate({**older, "schema_version": "18"}).connection_slots == ()
        with pytest.raises(ValidationError, match="schema 18 manifest declares no connection"):
            AppManifest.model_validate({**recorded, "schema_version": "18"})

    asyncio.run(scenario())


def test_a_cloud_from_before_both_slot_kinds_leaves_out_each_field_it_names():
    def unknown(field: str) -> httpx.Response:
        return _refused(
            "registration_invalid",
            f"`registration` has an unknown field `{field}`.",
            field="registration",
        )

    async def scenario():
        app = await _declaring_app()
        cloud = _Cloud()
        cloud.next_errors.extend([unknown("connection_slots"), unknown("model_slots")])
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            await publisher.register()
            first, second, third = (_PARTS & set(body) for body in _bodies(cloud))
            assert (first, second, third) == (
                _PARTS,
                {"model_slots", "settings_schema"},
                {"settings_schema"},
            )
            assert cloud.current == RELEASE.id
        await app.aclose()

    asyncio.run(scenario())


def test_an_unknown_field_refusal_naming_nothing_sent_is_raised():
    async def scenario():
        app = await _declaring_app()
        cloud = _Cloud()
        cloud.next_errors.append(
            _refused(
                "registration_invalid",
                "`registration` has an unknown field `other`.",
                field="registration",
            )
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            with pytest.raises(cloud_inventory.CloudRegistryRefused):
                await publisher.register()
            assert len(_bodies(cloud)) == 1
            await publisher.register()
            assert set(_bodies(cloud)[-1]) >= _PARTS
        await app.aclose()

    asyncio.run(scenario())


_SAP_READS = [
    "API_BUSINESS_PARTNER.read",
    "API_MATERIAL_DOCUMENT_SRV.read",
    "API_PURCHASEORDER_PROCESS_SRV.read",
    "API_SUPPLIERINVOICE_PROCESS_SRV.read",
]
_LOCAL = LocalConnections({})


def _agent(app: CayuApp, name: str, tools) -> None:
    app.register_agent(AgentSpec(name=name, model="model"), tools=tools)


def _declared(app: CayuApp) -> list[dict]:
    return [slot.declaration() for slot in app.connection_slots()]


def test_a_toolkits_tools_declare_their_slot_with_nothing_else_written():
    app = CayuApp(enable_logging=False)
    _agent(app, "reader", sap.tools(_LOCAL, slot="erp"))
    assert _declared(app) == [
        {
            "name": "erp",
            "connectors": ["sap", "sap-certificate", "sap-sandbox"],
            "scopes": _SAP_READS,
            "description": "SAP S/4HANA reads",
        }
    ]


def test_writes_add_their_scopes_and_write_slots_their_own_declaration():
    reads = CayuApp(enable_logging=False)
    _agent(reads, "reader", slack.tools(_LOCAL))
    assert _declared(reads)[0]["scopes"] == [
        "channels:history",
        "channels:read",
        "users:read",
        "users:read.email",
    ]
    writes = CayuApp(enable_logging=False)
    _agent(writes, "poster", slack.tools(_LOCAL, include_writes=True))
    _agent(writes, "analyst", snowflake.tools(_LOCAL, slot="warehouse", include_writes=True))
    _agent(writes, "reviewer", github.tools(_LOCAL, include_writes=True, write_slot="github-write"))
    declared = {slot["name"]: slot for slot in _declared(writes)}
    assert declared["slack"]["scopes"] == [
        "channels:history",
        "channels:read",
        "chat:write",
        "users:read",
        "users:read.email",
    ]
    assert (
        declared["warehouse"]["connectors"]
        == declared["snowflake-write"]["connectors"]
        == ["snowflake"]
    )
    assert declared["github"]["connectors"] == ["github-app"]
    # Reading a pull request also reads its head commit, the compare and its workflow runs.
    assert declared["github"]["scopes"] == [
        "actions:read",
        "checks:read",
        "contents:read",
        "issues:read",
        "metadata:read",
        "pull_requests:read",
        "statuses:read",
    ]
    assert {"issues:write", "metadata:read"} <= set(declared["github-write"]["scopes"])
    one = CayuApp(enable_logging=False)
    _agent(one, "analyst", snowflake.tools(_LOCAL, write_slot="snowflake", include_writes=True))
    assert [slot["name"] for slot in _declared(one)] == ["snowflake"]


def _retargeted(tool, *, connectors, scopes=()):
    tool.connection_slots = (ConnectionSlot(name="erp", connectors=connectors, scopes=scopes),)
    return tool


def test_one_slot_from_several_tools_merges_and_disjoint_connectors_are_refused():
    app = CayuApp(enable_logging=False)
    _agent(app, "reader", sap.tools(_LOCAL, slot="erp"))
    _agent(
        app,
        "other",
        [
            _retargeted(
                slack.tools(_LOCAL)[0], connectors=("sap", "example-erp"), scopes=("x.read",)
            )
        ],
    )
    (merged,) = _declared(app)
    assert merged["connectors"] == ["sap"]
    assert merged["scopes"] == sorted([*_SAP_READS, "x.read"])
    assert merged["description"] == "SAP S/4HANA reads"
    # The first tool registered with a slot gives its description, whatever comes later.
    first = _retargeted(slack.tools(_LOCAL)[0], connectors=("sap",))
    first.connection_slots = (
        first.connection_slots[0].model_copy(update={"description": "First"}),
    )
    ordered = CayuApp(enable_logging=False)
    _agent(ordered, "first", [first])
    _agent(ordered, "second", sap.tools(_LOCAL, slot="erp"))
    assert _declared(ordered)[0]["description"] == "First"
    with pytest.raises(ValueError, match="sap and for example-erp, which share none"):
        _agent(app, "clash", [_retargeted(slack.tools(_LOCAL)[0], connectors=("example-erp",))])
    assert "clash" not in app._agent_registry.registrations
    assert _declared(app) == [merged]


def test_the_applications_own_declaration_wins_for_connectors_and_description():
    before = CayuApp(enable_logging=False)
    before.require_connection("erp", connectors=("sap-sandbox",), scopes=("extra.read",))
    _agent(before, "reader", sap.tools(_LOCAL, slot="erp"))
    after = CayuApp(enable_logging=False)
    _agent(after, "reader", sap.tools(_LOCAL, slot="erp"))
    after.require_connection("erp", connectors=("sap-sandbox",), scopes=("extra.read",))
    for app in (before, after):
        assert _declared(app) == [
            {
                "name": "erp",
                "connectors": ["sap-sandbox"],
                "scopes": sorted([*_SAP_READS, "extra.read"]),
                "description": "SAP S/4HANA reads",
            }
        ]
    described = CayuApp(enable_logging=False)
    described.require_connection("erp", description="Supplier invoices to reconcile")
    _agent(described, "reader", sap.tools(_LOCAL, slot="erp"))
    (slot,) = _declared(described)
    assert slot["connectors"] == ["sap", "sap-certificate", "sap-sandbox"]
    assert slot["description"] == "Supplier invoices to reconcile"


def test_a_toolkit_slot_named_like_a_model_slot_or_past_the_limit_is_refused():
    app = CayuApp(enable_logging=False)
    app.require_model("slack", default=None)
    with pytest.raises(ValueError, match="already declared as a model slot"):
        _agent(app, "poster", slack.tools(_LOCAL))
    _agent(app, "reader", sap.tools(_LOCAL))
    with pytest.raises(ValueError, match="already declared as a connection slot"):
        app.require_model("sap", default=None)
    for index in range(63):
        app.require_connection(f"slot-{index}")
    with pytest.raises(ValueError, match="at most 64 connection slots"):
        _agent(app, "writer", snowflake.tools(_LOCAL))
    assert "writer" not in app._agent_registry.registrations


def test_registration_carries_a_toolkits_slots():
    async def scenario():
        app = await _usage_app()
        _agent(app, "reader", sap.tools(_LOCAL, slot="erp"))
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        (body,) = _bodies(cloud)
        assert body["connection_slots"] == _declared(app)
        assert [
            slot["name"] for slot in app.describe().model_dump(mode="json")["connection_slots"]
        ] == ["erp"]
        await app.aclose()

    asyncio.run(scenario())


def _on_model_setting():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return model_setting()


def test_an_agent_on_model_setting_gets_no_slot_named_like_a_connection_slot():
    own = CayuApp(enable_logging=False)
    own.register_agent(
        AgentSpec(name="github", model="model"),
        tools=github.tools(_LOCAL),
        settings=[_on_model_setting()],
    )
    assert own.model_slots() == ()
    assert [slot.name for slot in own.connection_slots()] == ["github"]
    earlier = CayuApp(enable_logging=False)
    _agent(earlier, "poster", slack.tools(_LOCAL))
    earlier.register_agent(AgentSpec(name="slack", model="model"), settings=[_on_model_setting()])
    assert earlier.model_slots() == ()
    assert "slack" in earlier._agent_registry.registrations
    declared = CayuApp(enable_logging=False)
    declared.require_connection("crm")
    declared.register_agent(AgentSpec(name="crm", model="model"), settings=[_on_model_setting()])
    assert declared.model_slots() == ()
