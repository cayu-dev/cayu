"""Fresh-process peer responsibility discovery; input travels over stdin."""

import asyncio
import json
import sys

from cayu.collaboration.peer_content import PeerContentAppendRequest, PeerContentExposureReceipt
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore


async def main():
    spec = json.load(sys.stdin)
    expected = PeerContentAppendRequest.model_validate(spec["request"])
    kwargs = {}
    if spec.get("qualification_alias_codec"):
        from tests.core.test_targeted_tool_grants import _codec

        kwargs["public_authority_alias_codec"] = _codec()
    store = (
        SQLiteSessionStore(spec["location"], **kwargs)
        if spec["backend"] == "sqlite"
        else PostgresSessionStore(spec["location"], schema_mode=SchemaMode.CREATE, **kwargs)
    )
    try:
        found = await store.read_peer_content(expected.append_key)
        pending = await store.list_pending_peer_content()
        result = {
            "status": None if found is None else found.status,
            "pending": expected in pending,
            "same_target": found is not None and found.append_key == expected.append_key,
        }
        if "exposures" in spec:
            matches = []
            for raw in spec["exposures"]:
                exposure = PeerContentExposureReceipt.model_validate(raw)
                reconstructed = await store.read_peer_content_exposure(
                    exposure.append_key, exposure.exposure_id
                )
                matches.append(reconstructed == exposure)
            result["exact_exposures"] = matches
        print(json.dumps(result))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
