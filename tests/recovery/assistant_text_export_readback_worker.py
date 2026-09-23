"""Fresh-interpreter exact replay; deliberately no projector registered."""

import asyncio
import json
import sys

from tests.core.test_assistant_text_peer_export import EXPORT_CONTEXT, Policy

from cayu.applications import CayuApp
from cayu.collaboration.exports import ExportLimits, SessionExportRegistration, SessionExportRequest
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode


async def main():
    value = json.loads(sys.stdin.read())
    expected = SessionExportRequest.model_validate(value["request"])
    store = (
        SQLiteSessionStore(value["location"], schema_mode=SchemaMode.VALIDATE)
        if value["backend"] == "sqlite"
        else PostgresSessionStore(value["location"], schema_mode=SchemaMode.VALIDATE)
    )
    policy = Policy()
    current = CayuApp(
        session_store=store,
        session_exports=SessionExportRegistration(
            owner=policy.ref.owner,
            policy=policy,
            projectors=(),
            limits=ExportLimits(max_exports=8, max_pending=4, max_retained_bytes=65536),
        ),
        enable_logging=False,
    )
    try:
        receipt = await current.export_session(expected, context=EXPORT_CONTEXT)
        payload = await current.read_session_export(expected, context=EXPORT_CONTEXT)
        print(json.dumps({"receipt": receipt.model_dump(mode="json"), "payload": payload}))
    finally:
        await current.drain_session_exports()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
