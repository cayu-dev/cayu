"""Raw artifact IDs and qualified closure adapter IDs occupy distinct namespaces."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.session_closure_conformance import create_closure_session
from tests.core.test_app_registry import _UncalledEnvironmentFactory
from tests.core.test_tool_effect_store_conformance import _stores

from cayu import CayuApp
from cayu.artifacts import ArtifactScope, LocalArtifactStore
from cayu.environments import Environment, EnvironmentSpec
from cayu.runtime.session_closure import RetainedSessionClosureStore
from cayu.tasks.base import InMemoryTaskStore, TaskCreate


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("store_id", ["budget-store", "task-store"])
def test_colliding_raw_artifact_id_remains_in_public_closure(backend, store_id, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with _stores(backend, tmp_path / "registration.db", dsn) as open_store:
            sessions = open_store()
            tasks = InMemoryTaskStore() if store_id == "task-store" else None
            root = uuid4().hex
            await create_closure_session(sessions, root)
            if tasks is not None:
                await tasks.create_task(TaskCreate(type="test", task_id="task", session_id=root))
                await tasks.complete_task("task", {})
            artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id=store_id)
            await artifacts.put_bytes(b"owned", filename="owned.txt", session_id=root)
            await artifacts.put_bytes(b"other", filename="other.txt", session_id="other")
            await artifacts.put_bytes(
                b"shared",
                filename="shared.txt",
                scope=ArtifactScope.ENVIRONMENT,
                environment_name="first",
            )
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            for name in ("first", "second"):
                app.register_environment(
                    Environment(EnvironmentSpec(name=name), artifact_store=artifacts)
                )
            app.register_environment_factory(
                EnvironmentSpec(name="factory"),
                _UncalledEnvironmentFactory(),
                artifact_store=artifacts,
            )
            qualified = f"artifact-store:{store_id}"
            inspection = await app.inspect_session_closure(root)
            assert inspection.complete
            matches = [record for record in inspection.records if record.store_id == qualified]
            assert len(matches) == 1 and matches[0].count == 1
            assert any(record.store_id == store_id for record in inspection.records)
            exported = await app.export_session_closure(root)
            assert exported.manifest.complete
            assert len(exported.session_records[qualified]["artifacts"]) == 1
            assert exported.session_records[qualified]["artifacts"][0]["size_bytes"] == 5
            report = await app.erase_session_closure(root)
            assert report.complete
            matches = [record for record in report.manifest.records if record.store_id == qualified]
            assert len(matches) == 1 and matches[0].disposition.value == "erased"
            assert matches[0].count == 1
            assert (await artifacts.list(session_id=root)).total_count == 0
            assert (await artifacts.list(session_id="other")).total_count == 1
            assert (await artifacts.list(scope=ArtifactScope.ENVIRONMENT)).total_count == 1
            assert await sessions.load(root) is None
            if tasks is not None:
                assert await tasks.load_task("task") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["concrete", "factory"])
@pytest.mark.parametrize("existing_default", [False, True])
def test_qualified_adapter_rejection_preserves_registration_state(tmp_path, kind, existing_default):
    async def scenario():
        app = CayuApp(
            enable_logging=False,
            session_closure_stores=(RetainedSessionClosureStore("artifact-store:budget-store"),),
        )
        root = "registration-rejection"
        await create_closure_session(app.session_store, root)
        initial = LocalArtifactStore(tmp_path / "initial", store_id="initial")
        await initial.put_bytes(b"initial", filename="initial.txt", session_id=root)
        app.register_environment(
            Environment(EnvironmentSpec(name="first"), artifact_store=initial),
            default=existing_default,
        )
        environments = dict(app._environments)
        artifacts = dict(app._artifact_store_registrations_by_id)
        fingerprints = app.artifact_store_registration_fingerprints(limit=64)
        coordinator = app._session_closure
        inventory = (await app.inspect_session_closure(root)).records
        default = app._default_environment_name

        def register(store):
            spec = EnvironmentSpec(name="local")
            if kind == "concrete":
                app.register_environment(Environment(spec, artifact_store=store), default=True)
            else:
                app.register_environment_factory(
                    spec, _UncalledEnvironmentFactory(), artifact_store=store, default=True
                )

        conflicting = LocalArtifactStore(tmp_path / "conflicting", store_id="budget-store")
        with pytest.raises(ValueError, match="store identity"):
            register(conflicting)

        assert app.list_environments() == ("first",)
        assert app._environments == environments
        assert app._artifact_store_registrations_by_id == artifacts
        assert app.artifact_store_registration_count() == 1
        assert app.artifact_store_registration_fingerprints(limit=64) == fingerprints
        assert app._default_environment_name == default
        assert app._session_closure is coordinator
        assert (await app.inspect_session_closure(root)).records == inventory

        corrected = LocalArtifactStore(tmp_path / "corrected", store_id="corrected")
        await corrected.put_bytes(b"corrected", filename="corrected.txt", session_id=root)
        register(corrected)
        assert app.list_environments() == ("first", "local")
        assert app.artifact_store_registration_count() == 2
        assert app._default_environment_name == "local"
        if kind == "concrete":
            assert app.get_environment().spec.name == "local"
        else:
            assert isinstance(app.get_environment_factory(), _UncalledEnvironmentFactory)
        inventory = (await app.inspect_session_closure(root)).records
        assert sum(record.store_id == "artifact-store:corrected" for record in inventory) == 1
        exported = await app.export_session_closure(root)
        assert len(exported.session_records["artifact-store:corrected"]["artifacts"]) == 1
        report = await app.erase_session_closure(root)
        assert report.complete
        assert (await initial.list(session_id=root)).total_count == 0
        assert (await corrected.list(session_id=root)).total_count == 0

    asyncio.run(scenario())
