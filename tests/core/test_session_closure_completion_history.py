"""Recursive completion receipts describe every admitted descendant across retries."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.session_closure_conformance import create_closure_session
from tests.core.test_tool_effect_store_conformance import _stores

from cayu import CayuApp
from cayu.runtime.session_closure import SessionClosurePolicy


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("failure_target", ["root", "child"])
def test_recursive_retry_preserves_complete_descendant_history(
    backend, failure_target, tmp_path, request, monkeypatch
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "completion.db", dsn) as open_store:
            store = open_store()
            root, child, grandchild = (uuid4().hex for _ in range(3))
            for target, parent in ((root, None), (child, root), (grandchild, child)):
                await create_closure_session(store, target, parent)
            policy = SessionClosurePolicy(child_policy="recursive")
            original = store.delete_session
            failed_id = root if failure_target == "root" else child
            deleted = []

            async def fail_before_commit(session_id, **kwargs):
                if session_id == failed_id:
                    raise RuntimeError("native deletion unavailable before commit")
                await original(session_id, **kwargs)
                deleted.append(session_id)

            monkeypatch.setattr(store, "delete_session", fail_before_commit)
            first = await CayuApp(session_store=store, enable_logging=False).erase_session_closure(
                root, policy=policy
            )
            assert not first.complete
            assert grandchild in deleted
            assert (child in deleted) is (failure_target == "root")
            assert await store.load(root) is not None
            assert await store.load_session_closure_receipt(root, first.plan_id) is None
            monkeypatch.setattr(store, "delete_session", original)

            # New coordinator and, for durable backends, a newly opened store.
            reopened = open_store()
            app = CayuApp(session_store=reopened, enable_logging=False)
            inspection = await app.inspect_session_closure(root, policy=policy)
            live_children = next(
                record for record in inspection.records if record.record_class == "child_sessions"
            )
            assert live_children.count == (0 if failure_target == "root" else 1)
            completed = await app.erase_session_closure(root, policy=policy)
            assert completed.complete and not completed.already_absent
            receipt = await reopened.load_session_closure_receipt(root, first.plan_id)
            replay = await CayuApp(
                session_store=open_store(), enable_logging=False
            ).erase_session_closure(root, policy=policy)
            assert replay.complete and replay.already_absent
            assert receipt is not None
            expected = {(child, root, "erased"), (grandchild, child, "erased")}
            for manifest in (
                completed.manifest.model_dump(mode="json"),
                receipt["manifest"],
                replay.manifest.model_dump(mode="json"),
            ):
                record = next(
                    item
                    for item in manifest["records"]
                    if item["store_id"] == "session-store"
                    and item["record_class"] == "child_sessions"
                )
                assert record["count"] == 2
                assert record["disposition"] == "erased"
                metadata = manifest["metadata"]
                assert set(metadata["descendant_session_ids"]) == {child, grandchild}
                assert len(metadata["descendant_dispositions"]) == 2
                assert {
                    (item["session_id"], item["parent_session_id"], item["disposition"])
                    for item in metadata["descendant_dispositions"]
                } == expected
            for target in (root, child, grandchild):
                assert await reopened.load(target) is None

    asyncio.run(scenario())
