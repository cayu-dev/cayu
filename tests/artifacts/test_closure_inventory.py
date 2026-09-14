import asyncio
import json

import pytest
from tests.core.session_closure_conformance import create_closure_session

from cayu import CayuApp
from cayu.artifacts import ArtifactListResult, ArtifactMetadata, ArtifactScope
from cayu.artifacts._closure import ArtifactClosureClaim, ArtifactClosureItem
from cayu.runtime.session_closure import (
    ArtifactSessionClosureStore,
    SessionClosureDisposition,
    SessionClosurePolicy,
)


class _InventoryStore:
    id = "inventory"

    def __init__(self):
        self.result = ArtifactListResult(
            artifacts=tuple(
                ArtifactMetadata(id=name, filename="item", size_bytes=1, session_id="session")
                for name in ("first", "second")
            ),
            total_count=2,
        )
        self.deleted = []

    async def list(self, **kwargs):
        assert kwargs["scope"] is ArtifactScope.SESSION
        assert kwargs["session_id"] == "session"
        return self.result

    async def delete(self, artifact_id):
        self.deleted.append(artifact_id)


def _corrupt(store, field):
    if field == "session":
        object.__setattr__(store.result.artifacts[1], "session_id", "foreign")
    elif field == "scope":
        object.__setattr__(store.result.artifacts[1], "scope", ArtifactScope.ENVIRONMENT)
    elif field == "duplicate":
        object.__setattr__(store.result.artifacts[1], "id", "first")
    elif field == "size":
        object.__setattr__(store.result.artifacts[1], "size_bytes", True)
    elif field == "truncated":
        object.__setattr__(store.result, "truncated", "false")
    else:
        object.__setattr__(store.result, "total_count", True)


@pytest.mark.parametrize("field", ["session", "scope", "duplicate", "size", "truncated", "count"])
@pytest.mark.parametrize("entrance", ["inspect", "erase", "export"])
def test_artifact_closure_revalidates_complete_inventory_before_use(field, entrance):
    async def run():
        store = _InventoryStore()
        _corrupt(store, field)
        adapter = ArtifactSessionClosureStore(store)
        kwargs = {"policy": SessionClosurePolicy()}
        if entrance == "erase":
            kwargs["plan_id"] = "plan"
        with pytest.raises(ValueError, match="Artifact closure"):
            await getattr(adapter, f"{entrance}_session_closure")("session", **kwargs)
        assert store.deleted == []

    asyncio.run(run())


def test_public_closure_rejects_foreign_artifact_without_any_deletion():
    async def run():
        store = _InventoryStore()
        _corrupt(store, "session")
        app = CayuApp(session_closure_stores=(ArtifactSessionClosureStore(store),))
        await create_closure_session(app.session_store, "session")
        report = await app.erase_session_closure("session")
        assert not report.complete
        assert store.deleted == []
        assert await app.session_store.load("session") is not None

    asyncio.run(run())


def test_artifact_closure_does_not_serialize_unused_extension_fields(capsys, caplog, recwarn):
    canary = "private-artifact-extension-value"

    class Private:
        def __repr__(self):
            return canary

    async def run():
        store = _InventoryStore()
        object.__setattr__(store.result.artifacts[0], "filename", Private())
        adapter = ArtifactSessionClosureStore(store)
        exported = await adapter.export_session_closure("session", policy=SessionClosurePolicy())
        assert len(exported["artifacts"]) == 2
        object.__setattr__(store.result.artifacts[1], "id", Private())
        with pytest.raises(ValueError, match="ownership evidence"):
            await adapter.erase_session_closure(
                "session", policy=SessionClosurePolicy(), plan_id="plan"
            )
        assert store.deleted == []

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert all(canary not in str(warning.message) for warning in recwarn)


def test_artifact_closure_detaches_selected_authority_before_mutating_awaits():
    class MutatingStore(_InventoryStore):
        supports_session_closure_claims = True

        async def claim_session_closure(self, session_id, plan_id, **kwargs):
            return ArtifactClosureClaim(
                self.id,
                session_id,
                plan_id,
                tuple(
                    ArtifactClosureItem(item.id, item.size_bytes, "0" * 64)
                    for item in self.result.artifacts
                ),
            )

        async def delete_session_closure_artifact(self, claim, artifact_id):
            await self.delete(artifact_id)

        async def delete(self, artifact_id):
            await super().delete(artifact_id)
            object.__setattr__(self.result.artifacts[1], "id", "foreign")

    async def run():
        store = MutatingStore()
        await ArtifactSessionClosureStore(store).erase_session_closure(
            "session", policy=SessionClosurePolicy(), plan_id="a" * 64
        )
        assert store.deleted == ["first", "second"]

    asyncio.run(run())


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_artifact_closure_identity_byte_bound_is_checked_before_deletion(delta):
    async def run():
        store = _InventoryStore()
        size = sum(
            len(
                json.dumps(
                    {"id": item.id, "size_bytes": item.size_bytes}, separators=(",", ":")
                ).encode()
            )
            for item in store.result.artifacts
        )
        policy = SessionClosurePolicy(max_bytes=size + delta)
        adapter = ArtifactSessionClosureStore(store)
        record = await adapter.inspect_session_closure("session", policy=policy)
        if delta < 0:
            assert record.disposition is SessionClosureDisposition.TRUNCATED
            with pytest.raises(ValueError, match="truncated"):
                await adapter.erase_session_closure("session", policy=policy, plan_id="plan")
            with pytest.raises(ValueError, match="truncated"):
                await adapter.export_session_closure("session", policy=policy)
            assert store.deleted == []
        else:
            assert record.count == 2
            assert record.disposition is SessionClosureDisposition.OWNED_ELIGIBLE
            # A complete read inventory does not authorize deletion through a
            # store that lacks a durable publication fence.
            with pytest.raises(NotImplementedError, match="cannot fence"):
                await adapter.erase_session_closure("session", policy=policy, plan_id="plan")
            assert store.deleted == []

    asyncio.run(run())


def test_public_closure_revalidates_artifact_inventory_after_initial_inspection():
    class ChangedInventory(_InventoryStore):
        calls = 0

        async def list(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                _corrupt(self, "session")
            return await super().list(**kwargs)

    async def run():
        store = ChangedInventory()
        app = CayuApp(session_closure_stores=(ArtifactSessionClosureStore(store),))
        await create_closure_session(app.session_store, "session")
        with pytest.raises(ValueError, match="Dependent closure admission"):
            await app.erase_session_closure("session")
        assert store.calls == 2
        assert store.deleted == []
        assert await app.session_store.load("session") is not None

    asyncio.run(run())
