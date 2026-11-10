"""A Release publishes its model slots in its registration and its manifest."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from tests.core.test_cloud_inventory import RELEASE, _Cloud, _config, _usage_app

from cayu.cloud_inventory import CloudInventoryPublisher


async def _slotted_app():
    app = await _usage_app()
    app.require_model("primary", default="bedrock/claude-sonnet-4-5", label="Primary model")
    app.require_model("review", default=None)
    return app


_DECLARED = [
    {"name": "primary", "default": "bedrock/claude-sonnet-4-5", "label": "Primary model"},
    {"name": "review", "default": None},
]


def _refused(code: str, **detail) -> httpx.Response:
    return httpx.Response(
        422, json={"error": {"code": code, "message": "refused", "detail": detail}}
    )


def _bodies(cloud: _Cloud) -> list[dict]:
    return [json.loads(request.content) for request in cloud.requests if request.method == "PUT"]


def test_registration_and_manifest_carry_the_declared_slots():
    async def scenario():
        app = await _slotted_app()
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        (body,) = _bodies(cloud)
        assert body["model_slots"] == _DECLARED
        manifest = app.describe()
        assert manifest.schema_version == "18"
        assert [slot.model_dump(mode="json") for slot in manifest.model_slots] == [
            _DECLARED[0],
            {"name": "review", "default": None},
        ]
        await app.aclose()

    asyncio.run(scenario())


def test_a_release_without_slots_leaves_the_field_out():
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        assert "model_slots" not in _bodies(cloud)[0]
        assert "model_slots" not in app.describe().model_dump(mode="json")
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("refusal", "kept"),
    [
        # Cloud names the model slots: only they are left out.
        (_refused("model_slots_invalid", field="model_slots[0].default"), {"settings_schema"}),
        # It refused both: the settings schema, with the slots' refusal in ``also``.
        (
            _refused(
                "settings_schema_invalid",
                field="settings_schema",
                also=[{"code": "model_slots_invalid", "message": "no", "field": "model_slots"}],
            ),
            set(),
        ),
        # It refused only the settings schema: the slots stay.
        (_refused("settings_schema_invalid", field="settings_schema"), {"model_slots"}),
        # An unnamed refusal leaves out the newest part first.
        (
            httpx.Response(422, json={"error": {"message": "refused", "detail": {}}}),
            {"settings_schema"},
        ),
        # Another named refusal is never about the slots.
        (_refused("invalid_settings_schema"), {"model_slots"}),
    ],
)
def test_a_refused_part_is_left_out_of_every_later_registration(refusal, kept, caplog):
    async def scenario():
        app = await _slotted_app()
        cloud = _Cloud()
        cloud.next_errors.append(refusal)
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            with caplog.at_level(logging.WARNING, logger="cayu.cloud_inventory"):
                await publisher.publish_once()
            first, second = _bodies(cloud)
            assert {"model_slots", "settings_schema"} <= set(first)
            assert {"model_slots", "settings_schema"} & set(second) == kept
            assert cloud.current == RELEASE.id
            await publisher.register()
            assert {"model_slots", "settings_schema"} & set(_bodies(cloud)[-1]) == kept
        await app.aclose()

    asyncio.run(scenario())
