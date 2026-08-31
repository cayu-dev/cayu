from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cayu._validation import canonical_durable_json_bytes
from cayu.agent_bundles import (
    AGENT_BUNDLE_INDEX_FILENAME,
    AGENT_BUNDLE_MAX_OBJECT_BYTES,
    AGENT_BUNDLE_OBJECT_DIRECTORY,
    AgentBundle,
    AgentBundleCoordinator,
    AgentBundleError,
    AgentBundleInventory,
    AgentBundleMaterializationAuthority,
    AgentBundleMaterializationAuthorization,
    AgentBundleMaterializationRequest,
    AgentBundleMode,
    AgentBundleObjectKind,
    AgentBundleObjectRef,
    AgentBundleSizeReport,
    AgentExternalBindingKind,
    AgentExternalBindingRequirement,
    AgentExternalBindingResolution,
    AgentMaterializationFreshIdentities,
    AgentSnapshotMaterializationMode,
    AgentSnapshotProfile,
    AgentSnapshotSessionDisposition,
    AgentSnapshotTerminalAuthority,
    AgentSnapshotTerminalAuthorization,
    AgentSnapshotTerminalCaptureRequest,
    FileSystemAgentSnapshotObjectStore,
    PortableAgentSnapshotComponentProvider,
    agent_snapshot_component_package,
    load_portable_agent_snapshot_component_providers,
    store_agent_snapshot_component_package,
)
from cayu.agent_snapshots import (
    AgentSnapshotAccess,
    AgentSnapshotAuthorizationError,
    AgentSnapshotCaptureRequest,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentSelector,
    AgentSnapshotCoordinator,
    AgentSnapshotExecutionProfileComponent,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterializationError,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotPinRequest,
    AgentSnapshotProtection,
    AgentSnapshotResultBinding,
    AgentSnapshotRetentionClass,
    AgentSnapshotStoreConflict,
    AgentSnapshotSubject,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialBinding,
    AgentSnapshotTrialStateMode,
    InMemoryAgentSnapshotStore,
    SQLiteAgentSnapshotStore,
)
from cayu.vaults.redaction import SecretRedactor


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _run(coroutine):
    return asyncio.run(coroutine)


class _TestMaterializationAuthority(AgentBundleMaterializationAuthority):
    def __init__(self) -> None:
        self.requests: list[str] = []

    async def authorize_materialization(
        self,
        request: AgentBundleMaterializationRequest,
    ) -> AgentBundleMaterializationAuthorization:
        self.requests.append(request.request_fingerprint)
        return AgentBundleMaterializationAuthorization.create(
            request=request,
            authority_fingerprint=_digest("test-materialization-authority"),
            fresh_identities=AgentMaterializationFreshIdentities(
                runtime_identity=_digest("fresh-runtime"),
                session_identity=_digest("fresh-session"),
                operation_identity=_digest("fresh-operation"),
                budget_identity=_digest("fresh-budget"),
                lease_identity=_digest("fresh-lease"),
                scratch_identity=_digest("fresh-scratch"),
                evaluator_identity=_digest("fresh-evaluator"),
                discovery_grant_ids=(),
            ),
            external_bindings=(
                AgentExternalBindingResolution(
                    requirement=requirement,
                    resolution_fingerprint=_digest(f"fresh:{requirement.name}"),
                    authority_fingerprint=_digest(f"authority:{requirement.name}"),
                )
                for requirement in request.bundle.external_bindings
                if requirement.required
            ),
        )


class _TestTerminalAuthority(AgentSnapshotTerminalAuthority):
    def __init__(self) -> None:
        self.results: list[str] = []

    async def authorize_terminal_capture(
        self,
        *,
        request: AgentSnapshotTerminalCaptureRequest,
        materialization,
        trial,
        result: AgentSnapshotResultBinding,
    ) -> AgentSnapshotTerminalAuthorization:
        assert trial.materialization_fingerprint == materialization.fingerprint
        self.results.append(result.fingerprint)
        return AgentSnapshotTerminalAuthorization.create(
            request=request,
            result=result,
            authority_fingerprint=_digest("test-terminal-authority"),
        )


def _portable_fixture(
    root: Path,
    *,
    knowledge_marker: str = "procedure-1",
    body_content: bytes = b"def answer():\n    return 42\n",
) -> tuple[
    list[PortableAgentSnapshotComponentProvider],
    FileSystemAgentSnapshotObjectStore,
    AgentSnapshotSubject,
    str,
]:
    profile = AgentSnapshotProfile.REUSABLE_AGENT
    provider_id = "cayu.portable-component.v1"
    environment_binding = AgentExternalBindingRequirement(
        kind=AgentExternalBindingKind.HOSTED_PROVIDER,
        name="primary-model",
        requirement_fingerprint=_digest("openai-compatible:model-class:v1"),
    )
    credential_binding = AgentExternalBindingRequirement(
        kind=AgentExternalBindingKind.CREDENTIAL,
        name="primary-model-credential",
        requirement_fingerprint=_digest("credential-class:openai-compatible:v1"),
    )
    definitions: dict[AgentSnapshotComponentKind, dict[str, Any]] = {
        AgentSnapshotComponentKind.BODY: {
            "payload": {"entrypoint": "agent.py"},
            "files": {"agent.py": body_content},
            "types": {"agent.py": "text/x-python"},
            "executables": (),
        },
        AgentSnapshotComponentKind.EXECUTION_PROFILE: {
            "payload": {"provider_target": "bound-fresh"},
        },
        AgentSnapshotComponentKind.TOOL_CATALOGUE: {
            "payload": {"tools": ["search_tools", "call_tool", "exec_command"]},
        },
        AgentSnapshotComponentKind.KNOWLEDGE: {
            "payload": {"view": "knowledge-change:17", "entries": [knowledge_marker]},
        },
        AgentSnapshotComponentKind.SESSION: {
            "payload": {"disposition": "fresh_on_materialize"},
            "session_disposition": AgentSnapshotSessionDisposition.FRESH_ON_MATERIALIZE,
        },
        AgentSnapshotComponentKind.WORK_CONTEXT: {
            "payload": {"checkpoint": "work-context:9"},
        },
        AgentSnapshotComponentKind.RECALL_POLICY: {
            "payload": {"profile": "bounded-hybrid-v2"},
        },
        AgentSnapshotComponentKind.LEARNING_POLICY: {
            "payload": {"admission": "reviewed-only"},
        },
        AgentSnapshotComponentKind.WORKSPACE: {
            "payload": {"promoted_paths": ["tools/check.py"]},
            "files": {"tools/check.py": b"#!/usr/bin/env python3\nprint(42)\n"},
            "types": {"tools/check.py": "text/x-python"},
            "executables": ("tools/check.py",),
        },
        AgentSnapshotComponentKind.ARTIFACTS: {
            "payload": {"promoted": ["evidence/model-card.json"]},
            "files": {"evidence/model-card.json": b'{"validated":true}\n'},
            "types": {"evidence/model-card.json": "application/json"},
            "executables": (),
        },
        AgentSnapshotComponentKind.ENVIRONMENT: {
            "payload": {
                "runtime_commit": "6e0c0bfbf644c27e275ebcaccfb7909816d5b7a8",
                "runtime_artifact_sha256": _digest("runtime-wheel"),
                "dependency_manifests": [
                    {"path": "pyproject.toml", "sha256": _digest("pyproject.toml")},
                    {"path": "uv.lock", "sha256": _digest("uv.lock")},
                ],
                "resolved_artifacts": [
                    {
                        "kind": "wheel",
                        "name": "cayu",
                        "sha256": _digest("runtime-wheel"),
                    }
                ],
                "tool_binaries": [{"name": "python", "sha256": _digest("python-binary")}],
                "os": "linux",
                "architecture": "x86_64",
                "abi": "cp311",
                "system_packages": [{"name": "git", "version": "2.51.0"}],
                "entrypoints": ["agent.py"],
                "non_secret_config": {"temperature": 0},
            },
            "external_bindings": (credential_binding, environment_binding),
        },
        AgentSnapshotComponentKind.STANDING_POLICY: {
            "payload": {"command_policy": "governed", "egress": "virtual"},
        },
    }
    object_store = FileSystemAgentSnapshotObjectStore(root / "objects")
    packages = {}
    blobs = {}
    for kind, definition in definitions.items():
        package, package_blobs = agent_snapshot_component_package(
            kind=kind,
            provider_id=provider_id,
            component_schema=f"cayu.test.{kind.value}.v1",
            profile=profile,
            payload=definition["payload"],
            files=definition.get("files"),
            file_content_types=definition.get("types"),
            executable_paths=definition.get("executables", ()),
            external_bindings=definition.get("external_bindings", ()),
            session_disposition=definition.get("session_disposition"),
        )
        packages[kind] = package
        blobs[kind] = package_blobs
    execution = AgentSnapshotExecutionProfileRef(
        schema_version=1,
        fingerprint=packages[AgentSnapshotComponentKind.EXECUTION_PROFILE].digest,
        components=(
            AgentSnapshotExecutionProfileComponent(
                name="runtime",
                fingerprint=_digest("runtime-profile"),
                availability="available",
            ),
        ),
    )
    providers = [
        PortableAgentSnapshotComponentProvider(
            package,
            blobs[kind],
            object_store=object_store,
            materialization_root=root / "materialized",
            execution_profile=(
                execution if kind is AgentSnapshotComponentKind.EXECUTION_PROFILE else None
            ),
        )
        for kind, package in packages.items()
    ]
    subject = AgentSnapshotSubject(
        agent_id="compound-agent-v123",
        application_id="compound",
        project_id="n9",
        body_release=AgentSnapshotLogicalRef(
            fingerprint=packages[AgentSnapshotComponentKind.BODY].digest,
            revision=f"component:{packages[AgentSnapshotComponentKind.BODY].digest}",
        ),
    )
    return providers, object_store, subject, _digest("source-authority-scope")


def test_full_bundle_imports_identical_root_and_materializes_in_fresh_stores(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        providers, source_objects, subject, scope = _portable_fixture(tmp_path / "source")
        source_snapshots = InMemoryAgentSnapshotStore()
        snapshot_coordinator = AgentSnapshotCoordinator(
            providers,
            store=source_snapshots,
        )
        snapshot = await snapshot_coordinator.capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="capture-agent-v123",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted(
                        (provider.kind for provider in providers),
                        key=str,
                    )
                ),
            )
        )
        source_access = AgentSnapshotAccess(
            snapshot=snapshot.ref,
            binding_id=snapshot.identity_binding.binding_id,
            authority_scope_fingerprint=scope,
        )
        bundle_path = (tmp_path / "compound-agent-v123").resolve()
        export = await AgentBundleCoordinator(
            snapshot_store=source_snapshots,
            object_store=source_objects,
        ).export(
            operation_id="export-agent-v123",
            access=source_access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=bundle_path,
            mode=AgentBundleMode.FULL,
        )

        assert export.bundle.snapshot_ref == snapshot.ref
        assert {binding.name for binding in export.bundle.external_bindings} == {
            "primary-model",
            "primary-model-credential",
        }
        assert export.bundle.size_report.incremental_transfer_bytes > 0
        assert (bundle_path / AGENT_BUNDLE_INDEX_FILENAME).is_file()

        destination_objects = FileSystemAgentSnapshotObjectStore(
            (tmp_path / "destination" / "objects").resolve()
        )
        destination_snapshots = InMemoryAgentSnapshotStore()
        destination_bundle_coordinator = AgentBundleCoordinator(
            snapshot_store=destination_snapshots,
            object_store=destination_objects,
        )
        imported = await destination_bundle_coordinator.import_bundle(
            operation_id="import-agent-v123",
            source=bundle_path,
            subject=subject.model_copy(update={"agent_id": "compound-agent-imported"}),
            authority_scope_fingerprint=_digest("destination-authority-scope"),
            owner="n9-release",
        )

        assert imported.snapshot_ref == snapshot.ref
        rebound_snapshot = await destination_snapshots.load_snapshot(snapshot.snapshot_root)
        assert rebound_snapshot is not None
        assert rebound_snapshot.subject.agent_id == "compound-agent-imported"
        assert rebound_snapshot.evaluator is None
        assert rebound_snapshot.promotion_authority is None

        restarted_objects = FileSystemAgentSnapshotObjectStore(destination_objects.root)
        materialization_request = AgentBundleMaterializationRequest(
            operation_id="materialize-agent-v123",
            bundle=export.bundle,
            access=AgentSnapshotAccess(
                snapshot=rebound_snapshot.ref,
                binding_id=imported.binding_id,
                authority_scope_fingerprint=_digest("destination-authority-scope"),
            ),
            mode=AgentSnapshotMaterializationMode.FORK_AS_SEED,
            candidate_id="fresh-candidate",
            trial_id="fresh-trial",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
        materialization_root = (tmp_path / "fresh-materialization").resolve()
        with pytest.raises(AgentBundleError, match="materialization_authority_unavailable"):
            await AgentBundleCoordinator(
                snapshot_store=destination_snapshots,
                object_store=restarted_objects,
            ).materialize(
                materialization_request,
                materialization_root=materialization_root,
            )
        assert not materialization_root.exists()

        materialization_authority = _TestMaterializationAuthority()
        restarted_bundle_coordinator = AgentBundleCoordinator(
            snapshot_store=destination_snapshots,
            object_store=restarted_objects,
            materialization_authority=materialization_authority,
        )
        forged_request = materialization_request.model_copy(
            update={
                "access": materialization_request.access.model_copy(
                    update={"binding_id": _digest("forged-import-binding")}
                )
            }
        )
        with pytest.raises(AgentSnapshotAuthorizationError):
            await restarted_bundle_coordinator.materialize(
                forged_request,
                materialization_root=materialization_root,
            )
        assert materialization_authority.requests == []
        materialized = await restarted_bundle_coordinator.materialize(
            materialization_request,
            materialization_root=materialization_root,
        )

        assert materialized.snapshot_ref == snapshot.ref
        assert materialized.fresh_identities.discovery_grant_ids == ()
        assert materialized.fresh_identities.session_identity == _digest("fresh-session")
        assert materialization_authority.requests == [
            materialized.authorization.request_fingerprint
        ]
        assert len(materialized.materialization.components) == len(rebound_snapshot.components)
        workspace = (
            tmp_path
            / "fresh-materialization"
            / materialized.state_scope_id
            / AgentSnapshotComponentKind.WORKSPACE.value
            / "tools"
            / "check.py"
        )
        assert workspace.read_text(encoding="utf-8").endswith("print(42)\n")
        assert workspace.stat().st_mode & 0o100

        recovery_providers = await load_portable_agent_snapshot_component_providers(
            rebound_snapshot,
            object_store=restarted_objects,
            materialization_root=materialization_root,
        )
        recovered = await AgentSnapshotCoordinator(
            recovery_providers,
            store=destination_snapshots,
        ).recover_materialization(
            materialized.materialization.fingerprint,
            access=materialization_request.access,
        )
        assert recovered == materialized.materialization

        alternate_providers = await load_portable_agent_snapshot_component_providers(
            rebound_snapshot,
            object_store=restarted_objects,
            materialization_root=(tmp_path / "alternate-materialization").resolve(),
        )
        with pytest.raises(AgentSnapshotMaterializationError):
            await AgentSnapshotCoordinator(
                alternate_providers,
                store=destination_snapshots,
            ).recover_materialization(
                materialized.materialization.fingerprint,
                access=materialization_request.access,
            )

        workspace.write_bytes(b"corrupt materialized workspace\n")
        with pytest.raises(AgentSnapshotMaterializationError):
            await AgentSnapshotCoordinator(
                recovery_providers,
                store=destination_snapshots,
            ).recover_materialization(
                materialized.materialization.fingerprint,
                access=materialization_request.access,
            )

    _run(scenario())


def test_hostile_bundles_fail_before_snapshot_publication(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, source_objects, subject, scope = _portable_fixture(tmp_path / "source")
        source_snapshots = InMemoryAgentSnapshotStore()
        snapshot = await AgentSnapshotCoordinator(providers, store=source_snapshots).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="capture-hostile-fixture",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        clean_path = (tmp_path / "clean-bundle").resolve()
        exported = await AgentBundleCoordinator(
            snapshot_store=source_snapshots,
            object_store=source_objects,
        ).export(
            operation_id="export-hostile-fixture",
            access=AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=scope,
            ),
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=clean_path,
        )

        object_reference = next(
            reference
            for reference in exported.bundle.closure
            if reference.kind.value == "component_blob"
        )

        def object_path(root: Path) -> Path:
            return (
                root
                / AGENT_BUNDLE_OBJECT_DIRECTORY
                / object_reference.digest[:2]
                / object_reference.digest[2:]
            )

        cases: dict[str, Callable[[Path], object]] = {}

        def corrupt(root: Path) -> None:
            path = object_path(root)
            content = path.read_bytes()
            path.write_bytes(bytes([content[0] ^ 1]) + content[1:])

        cases["corrupt"] = corrupt
        cases["truncated"] = lambda root: object_path(root).write_bytes(
            object_path(root).read_bytes()[:-1]
        )
        cases["missing"] = lambda root: object_path(root).unlink()
        cases["extra"] = lambda root: (root / "unexpected.bin").write_bytes(b"extra")

        def unsupported_schema(root: Path) -> None:
            index = json.loads((root / AGENT_BUNDLE_INDEX_FILENAME).read_text())
            index["schema_version"] = 999
            (root / AGENT_BUNDLE_INDEX_FILENAME).write_bytes(
                canonical_durable_json_bytes(index, "hostile_bundle")
            )

        cases["unsupported-schema"] = unsupported_schema

        def oversized(root: Path) -> None:
            index = json.loads((root / AGENT_BUNDLE_INDEX_FILENAME).read_text())
            target = next(
                item for item in index["closure"] if item["digest"] == object_reference.digest
            )
            target["byte_count"] = AGENT_BUNDLE_MAX_OBJECT_BYTES + 1
            (root / AGENT_BUNDLE_INDEX_FILENAME).write_bytes(
                canonical_durable_json_bytes(index, "hostile_bundle")
            )

        cases["zip-bomb-style-size"] = oversized

        def wrong_root(root: Path) -> None:
            index = json.loads((root / AGENT_BUNDLE_INDEX_FILENAME).read_text())
            index["snapshot_ref"]["snapshot_root"] = "0" * 64
            (root / AGENT_BUNDLE_INDEX_FILENAME).write_bytes(
                canonical_durable_json_bytes(index, "hostile_bundle")
            )

        cases["wrong-root"] = wrong_root

        wrong_scope_bundle = AgentBundle.create(
            snapshot_ref=exported.bundle.snapshot_ref,
            export_binding_id=exported.bundle.export_binding_id,
            export_authority_scope_fingerprint=_digest("wrong-export-scope"),
            destination_inventory_fingerprint=(exported.bundle.destination_inventory_fingerprint),
            profile=exported.bundle.profile,
            mode=exported.bundle.mode,
            snapshot_document=exported.bundle.snapshot_document,
            closure=exported.bundle.closure,
            transferred_digests=exported.bundle.transferred_digests,
            external_bindings=exported.bundle.external_bindings,
            size_report=exported.bundle.size_report,
        )
        stripped_binding_bundle = AgentBundle.create(
            snapshot_ref=exported.bundle.snapshot_ref,
            export_binding_id=exported.bundle.export_binding_id,
            export_authority_scope_fingerprint=(exported.bundle.export_authority_scope_fingerprint),
            destination_inventory_fingerprint=(exported.bundle.destination_inventory_fingerprint),
            profile=exported.bundle.profile,
            mode=exported.bundle.mode,
            snapshot_document=exported.bundle.snapshot_document,
            closure=exported.bundle.closure,
            transferred_digests=exported.bundle.transferred_digests,
            external_bindings=(),
            size_report=AgentBundleSizeReport(
                root_manifest_bytes=exported.bundle.size_report.root_manifest_bytes,
                logical_closure_bytes=exported.bundle.size_report.logical_closure_bytes,
                unique_stored_bytes=exported.bundle.size_report.unique_stored_bytes,
                shared_stored_bytes=exported.bundle.size_report.shared_stored_bytes,
                incremental_transfer_bytes=(exported.bundle.size_report.incremental_transfer_bytes),
                materialized_disk_bytes=(exported.bundle.size_report.materialized_disk_bytes),
                unresolved_external_bindings=(),
            ),
        )

        for name, mutation in cases.items():
            hostile_path = (tmp_path / f"hostile-{name}").resolve()
            shutil.copytree(clean_path, hostile_path)
            mutation(hostile_path)
            destination_snapshots = InMemoryAgentSnapshotStore()
            with pytest.raises(AgentBundleError):
                await AgentBundleCoordinator(
                    snapshot_store=destination_snapshots,
                    object_store=FileSystemAgentSnapshotObjectStore(
                        (tmp_path / f"objects-{name}").resolve()
                    ),
                ).import_bundle(
                    operation_id=f"import-{name}",
                    source=hostile_path,
                    subject=subject,
                    authority_scope_fingerprint=_digest(f"destination-{name}"),
                    owner="hostile-test",
                )
            assert await destination_snapshots.load_snapshot(snapshot.snapshot_root) is None

        for name, hostile_bundle in (
            ("wrong-scope", wrong_scope_bundle),
            ("stripped-external-binding", stripped_binding_bundle),
        ):
            hostile_path = (tmp_path / f"hostile-{name}").resolve()
            shutil.copytree(clean_path, hostile_path)
            (hostile_path / AGENT_BUNDLE_INDEX_FILENAME).write_bytes(
                canonical_durable_json_bytes(
                    hostile_bundle.model_dump(mode="json"),
                    "hostile_bundle",
                )
            )
            destination_snapshots = InMemoryAgentSnapshotStore()
            with pytest.raises(AgentBundleError):
                await AgentBundleCoordinator(
                    snapshot_store=destination_snapshots,
                    object_store=FileSystemAgentSnapshotObjectStore(
                        (tmp_path / f"objects-{name}").resolve()
                    ),
                ).import_bundle(
                    operation_id=f"import-{name}",
                    source=hostile_path,
                    subject=subject,
                    authority_scope_fingerprint=_digest(f"destination-{name}"),
                    owner="hostile-test",
                )
            assert await destination_snapshots.load_snapshot(snapshot.snapshot_root) is None

    _run(scenario())


def test_component_packages_reject_traversal_and_private_state() -> None:
    with pytest.raises(ValueError, match="relative POSIX path"):
        agent_snapshot_component_package(
            kind=AgentSnapshotComponentKind.BODY,
            provider_id="provider",
            component_schema="schema.v1",
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            payload={"entrypoint": "agent.py"},
            files={"../agent.py": b"unsafe"},
        )

    with pytest.raises(ValidationError, match="cannot enter a component package"):
        agent_snapshot_component_package(
            kind=AgentSnapshotComponentKind.ENVIRONMENT,
            provider_id="provider",
            component_schema="schema.v1",
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            payload={"api_key": "must-not-export"},
        )


def test_known_secret_bytes_cannot_enter_portable_component_storage(tmp_path: Path) -> None:
    async def scenario() -> None:
        package, blobs = agent_snapshot_component_package(
            kind=AgentSnapshotComponentKind.BODY,
            provider_id="provider",
            component_schema="schema.v1",
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            payload={"entrypoint": "agent.py"},
            files={"agent.py": b"value = 'known-secret-value'\n"},
        )
        provider = PortableAgentSnapshotComponentProvider(
            package,
            blobs,
            object_store=FileSystemAgentSnapshotObjectStore((tmp_path / "objects").resolve()),
            materialization_root=(tmp_path / "materialized").resolve(),
            secret_redactor=SecretRedactor("known-secret-value"),
        )
        with pytest.raises(AgentBundleError, match="component_contains_secret"):
            await provider.capture(
                AgentSnapshotCaptureRequest(
                    capture_request_id="secret-capture",
                    subject=AgentSnapshotSubject(
                        agent_id="agent",
                        application_id="application",
                        project_id="project",
                        body_release=AgentSnapshotLogicalRef(
                            fingerprint=package.digest,
                            revision=f"component:{package.digest}",
                        ),
                    ),
                    authority_scope_fingerprint=_digest("secret-scope"),
                    components=(
                        AgentSnapshotComponentSelector(kind=AgentSnapshotComponentKind.BODY),
                        AgentSnapshotComponentSelector(
                            kind=AgentSnapshotComponentKind.EXECUTION_PROFILE
                        ),
                    ),
                ),
                AgentSnapshotComponentSelector(kind=AgentSnapshotComponentKind.BODY),
            )

    _run(scenario())


def test_import_rejects_registered_secrets_before_snapshot_publication(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        secret = "registered-import-secret"
        providers, source_objects, subject, scope = _portable_fixture(
            tmp_path / "secret-source",
            body_content=f"value = {secret!r}\n".encode(),
        )
        source_snapshots = InMemoryAgentSnapshotStore()
        snapshot = await AgentSnapshotCoordinator(
            providers,
            store=source_snapshots,
        ).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="secret-import-source",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        bundle_path = (tmp_path / "secret-bundle").resolve()
        await AgentBundleCoordinator(
            snapshot_store=source_snapshots,
            object_store=source_objects,
        ).export(
            operation_id="secret-export",
            access=AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=scope,
            ),
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=bundle_path,
        )
        destination_snapshots = InMemoryAgentSnapshotStore()
        with pytest.raises(AgentBundleError, match="bundle_contains_secret"):
            await AgentBundleCoordinator(
                snapshot_store=destination_snapshots,
                object_store=FileSystemAgentSnapshotObjectStore(
                    (tmp_path / "secret-destination-objects").resolve()
                ),
                secret_redactor=SecretRedactor(secret),
            ).import_bundle(
                operation_id="secret-import",
                source=bundle_path,
                subject=subject,
                authority_scope_fingerprint=_digest("secret-import-scope"),
                owner="secret-import-owner",
            )
        assert await destination_snapshots.load_snapshot(snapshot.snapshot_root) is None

    _run(scenario())


def test_filesystem_cas_rejects_symlinked_object_shards(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = (tmp_path / "cas").resolve()
        outside = (tmp_path / "outside").resolve()
        outside.mkdir()
        store = FileSystemAgentSnapshotObjectStore(root)
        content = b"must stay in the configured CAS\n"
        reference = AgentBundleObjectRef(
            digest=sha256(content).hexdigest(),
            kind=AgentBundleObjectKind.COMPONENT_BLOB,
            schema_id="cayu.agent-snapshot.component-blob.v1",
            byte_count=len(content),
        )
        shard = root / AGENT_BUNDLE_OBJECT_DIRECTORY / reference.digest[:2]
        shard.symlink_to(outside, target_is_directory=True)

        with pytest.raises(AgentBundleError, match="object_path_not_regular"):
            await store.put(reference, content)
        assert not (outside / reference.digest[2:]).exists()

    _run(scenario())


def test_component_files_can_stream_through_cas_without_inline_bytes(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = (tmp_path / "large-component.bin").resolve()
        source.write_bytes(b"portable-agent-state\n" * 262_144)
        object_store = FileSystemAgentSnapshotObjectStore((tmp_path / "objects").resolve())
        package = await store_agent_snapshot_component_package(
            kind=AgentSnapshotComponentKind.ARTIFACTS,
            provider_id="streaming-provider",
            component_schema="cayu.test.streaming-artifacts.v1",
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            payload={"promoted": ["large-component.bin"]},
            file_paths={"large-component.bin": source},
            object_store=object_store,
        )
        file = package.files[0]
        blob_reference = AgentBundleObjectRef(
            digest=file.digest,
            kind=AgentBundleObjectKind.COMPONENT_BLOB,
            schema_id="cayu.agent-snapshot.component-blob.v1",
            byte_count=file.byte_count,
        )
        assert await object_store.verify(blob_reference)
        copied = (tmp_path / "copied-component.bin").resolve()
        await object_store.copy_to(blob_reference, copied)
        assert copied.stat().st_size == source.stat().st_size
        assert sha256(copied.read_bytes()).hexdigest() == file.digest

    _run(scenario())


def test_public_example_crosses_fresh_process_export_import_materialization(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "fresh-process-demo").resolve()
    script = Path(__file__).parents[2] / "examples" / "portable_agent_bundle.py"
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
    }

    def run(command: str) -> dict[str, Any]:
        completed = subprocess.run(
            [sys.executable, str(script), command, "--root", str(root)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return json.loads(completed.stdout)

    exported = run("export")
    original = root / "compound-agent-v123.cayu"
    copied = root / "copied-agent-bundle.cayu"
    original.rename(copied)
    shutil.copyfile(copied, original)
    inspected = run("inspect")
    imported = run("import")
    materialized = run("materialize")

    exported_root = exported["bundle"]["snapshot_ref"]["snapshot_root"]
    assert inspected["snapshot_root"] == exported_root
    assert inspected["mode"] == "full"
    assert imported["snapshot_ref"]["snapshot_root"] == exported_root
    assert materialized["snapshot_ref"]["snapshot_root"] == exported_root
    assert materialized["fresh_identities"]["discovery_grant_ids"] == []
    assert (root / "materialized" / materialized["state_scope_id"]).is_dir()


def test_terminal_capture_publishes_a_descendant_snapshot_receipt(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, object_store, subject, scope = _portable_fixture(tmp_path / "terminal")
        snapshot_store = InMemoryAgentSnapshotStore()
        coordinator = AgentSnapshotCoordinator(providers, store=snapshot_store)
        parent = await coordinator.capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="terminal-parent",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        materialization = await coordinator.materialize(
            AgentSnapshotMaterializationRequest(
                access=AgentSnapshotAccess(
                    snapshot=parent.ref,
                    binding_id=parent.identity_binding.binding_id,
                    authority_scope_fingerprint=scope,
                ),
                candidate_id="candidate-terminal",
                trial_id="trial-terminal",
                state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            )
        )
        trial = AgentSnapshotTrialBinding.create(
            materialization=materialization,
            case_id="terminal-case",
            trial_id="trial-terminal",
            evaluator_fingerprint=_digest("terminal-evaluator"),
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
        )
        await snapshot_store.save_trial(trial)
        result = await coordinator.record_result(
            AgentSnapshotResultBinding.create(
                trial=trial,
                session_id="terminal-session",
                terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
                runtime_evidence_fingerprint=_digest("terminal-runtime-evidence"),
                eval_result_revision=_digest("terminal-result"),
                recorded_at=datetime(2026, 8, 29, tzinfo=UTC),
            )
        )
        descendant_providers, _, _, _ = _portable_fixture(
            tmp_path / "terminal",
            knowledge_marker="learned-procedure",
        )
        terminal_request = AgentSnapshotTerminalCaptureRequest(
            operation_id="terminal-capture",
            capture=AgentSnapshotCaptureRequest(
                capture_request_id="terminal-descendant",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted(
                        (provider.kind for provider in descendant_providers),
                        key=str,
                    )
                ),
                parent_snapshot_fingerprint=parent.snapshot_root,
                lineage=(parent.snapshot_root,),
            ),
            parent_materialization_fingerprint=materialization.fingerprint,
            parent_result_fingerprint=result.fingerprint,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            finalization_policy="explicit-consolidation",
        )
        with pytest.raises(AgentBundleError, match="terminal_authority_unavailable"):
            await AgentBundleCoordinator(
                snapshot_store=snapshot_store,
                object_store=object_store,
            ).capture_terminal(
                terminal_request,
                providers=descendant_providers,
            )
        terminal_authority = _TestTerminalAuthority()
        receipt = await AgentBundleCoordinator(
            snapshot_store=snapshot_store,
            object_store=object_store,
            terminal_authority=terminal_authority,
        ).capture_terminal(
            terminal_request,
            providers=descendant_providers,
        )
        assert receipt.parent_snapshot_ref == parent.ref
        assert receipt.parent_result_fingerprint == result.fingerprint
        assert receipt.terminal_authorization.result_fingerprint == result.fingerprint
        assert terminal_authority.results == [result.fingerprint]
        assert receipt.descendant_snapshot_ref != parent.ref
        descendant = await snapshot_store.load_snapshot(
            receipt.descendant_snapshot_ref.snapshot_root
        )
        assert descendant is not None
        assert descendant.parent_snapshot_fingerprint == parent.snapshot_root

    _run(scenario())


def test_terminal_capture_rejects_an_unsafe_durable_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, object_store, subject, scope = _portable_fixture(tmp_path / "terminal-reject")
        snapshot_store = InMemoryAgentSnapshotStore()
        snapshot_coordinator = AgentSnapshotCoordinator(providers, store=snapshot_store)
        parent = await snapshot_coordinator.capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="unsafe-terminal-parent",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        materialization = await snapshot_coordinator.materialize(
            AgentSnapshotMaterializationRequest(
                access=AgentSnapshotAccess(
                    snapshot=parent.ref,
                    binding_id=parent.identity_binding.binding_id,
                    authority_scope_fingerprint=scope,
                ),
                candidate_id="unsafe-candidate",
                trial_id="unsafe-trial",
                state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            )
        )
        trial = AgentSnapshotTrialBinding.create(
            materialization=materialization,
            case_id="unsafe-case",
            trial_id="unsafe-trial",
            evaluator_fingerprint=_digest("unsafe-evaluator"),
            created_at=datetime(2026, 8, 29, tzinfo=UTC),
        )
        await snapshot_store.save_trial(trial)
        result = await snapshot_coordinator.record_result(
            AgentSnapshotResultBinding.create(
                trial=trial,
                session_id="unsafe-session",
                terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
                runtime_evidence_fingerprint=_digest("unsafe-runtime-evidence"),
                eval_result_revision=_digest("unsafe-result"),
                open_operation_ids=("open-tool-call",),
                recorded_at=datetime(2026, 8, 29, tzinfo=UTC),
            )
        )
        terminal_authority = _TestTerminalAuthority()
        with pytest.raises(AgentBundleError, match="terminal_result_frontier_open"):
            await AgentBundleCoordinator(
                snapshot_store=snapshot_store,
                object_store=object_store,
                terminal_authority=terminal_authority,
            ).capture_terminal(
                AgentSnapshotTerminalCaptureRequest(
                    operation_id="unsafe-terminal",
                    capture=AgentSnapshotCaptureRequest(
                        capture_request_id="unsafe-terminal-descendant",
                        subject=subject,
                        authority_scope_fingerprint=scope,
                        components=tuple(
                            AgentSnapshotComponentSelector(kind=kind)
                            for kind in sorted((provider.kind for provider in providers), key=str)
                        ),
                        parent_snapshot_fingerprint=parent.snapshot_root,
                        lineage=(parent.snapshot_root,),
                    ),
                    parent_materialization_fingerprint=materialization.fingerprint,
                    parent_result_fingerprint=result.fingerprint,
                    profile=AgentSnapshotProfile.REUSABLE_AGENT,
                    finalization_policy="explicit-consolidation",
                ),
                providers=providers,
            )
        assert terminal_authority.results == []

    _run(scenario())


def test_concurrent_and_lost_ack_bundle_operations_converge(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, source_objects, subject, scope = _portable_fixture(tmp_path / "converge")
        source_snapshots = InMemoryAgentSnapshotStore()
        snapshot = await AgentSnapshotCoordinator(providers, store=source_snapshots).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="convergent-capture",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        source = AgentBundleCoordinator(
            snapshot_store=source_snapshots,
            object_store=source_objects,
        )
        destination_path = (tmp_path / "convergent-bundle").resolve()
        export_access = AgentSnapshotAccess(
            snapshot=snapshot.ref,
            binding_id=snapshot.identity_binding.binding_id,
            authority_scope_fingerprint=scope,
        )

        async def export_once():
            return await source.export(
                operation_id="convergent-export",
                access=export_access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination_path,
            )

        first_export, second_export = await asyncio.gather(
            export_once(),
            export_once(),
        )
        lost_ack_retry = await export_once()
        assert first_export == second_export == lost_ack_retry

        destination_snapshots = InMemoryAgentSnapshotStore()
        destination = AgentBundleCoordinator(
            snapshot_store=destination_snapshots,
            object_store=FileSystemAgentSnapshotObjectStore(
                (tmp_path / "convergent-objects").resolve()
            ),
        )

        async def import_once():
            return await destination.import_bundle(
                operation_id="convergent-import",
                source=destination_path,
                subject=subject,
                authority_scope_fingerprint=_digest("convergent-destination-scope"),
                owner="convergent-test",
            )

        first_import, second_import = await asyncio.gather(
            import_once(),
            import_once(),
        )
        lost_import_ack_retry = await import_once()
        assert first_import == second_import == lost_import_ack_retry
        assert first_import.snapshot_ref == snapshot.ref

    _run(scenario())


def test_export_retry_releases_protection_left_by_lost_ack(tmp_path: Path) -> None:
    class CrashBeforeReleaseStore(InMemoryAgentSnapshotStore):
        fail_release = True

        async def release_snapshot_protection(
            self,
            *,
            operation_id: str,
            access: AgentSnapshotAccess,
            protection_id: str,
        ) -> AgentSnapshotProtection:
            if self.fail_release:
                self.fail_release = False
                raise RuntimeError("simulated process loss before release")
            return await super().release_snapshot_protection(
                operation_id=operation_id,
                access=access,
                protection_id=protection_id,
            )

    async def scenario() -> None:
        providers, source_objects, subject, scope = _portable_fixture(tmp_path / "lost-ack")
        source_snapshots = CrashBeforeReleaseStore()
        snapshot = await AgentSnapshotCoordinator(providers, store=source_snapshots).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="lost-ack-capture",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        access = AgentSnapshotAccess(
            snapshot=snapshot.ref,
            binding_id=snapshot.identity_binding.binding_id,
            authority_scope_fingerprint=scope,
        )
        coordinator = AgentBundleCoordinator(
            snapshot_store=source_snapshots,
            object_store=source_objects,
        )
        destination = (tmp_path / "lost-ack-bundle").resolve()
        with pytest.raises(RuntimeError, match="simulated process loss"):
            await coordinator.export(
                operation_id="lost-ack-export",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )
        assert source_snapshots._root_is_protected(snapshot.snapshot_root)

        recovered = await coordinator.export(
            operation_id="lost-ack-export",
            access=access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=destination,
        )
        assert recovered.bundle.snapshot_ref == snapshot.ref
        assert not source_snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_sqlite_import_publication_rolls_back_root_when_pin_conflicts(tmp_path: Path) -> None:
    async def capture(knowledge_marker: str, capture_id: str):
        providers, _, subject, scope = _portable_fixture(
            tmp_path / "atomic-source",
            knowledge_marker=knowledge_marker,
        )
        snapshot = await AgentSnapshotCoordinator(
            providers,
            store=InMemoryAgentSnapshotStore(),
        ).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id=capture_id,
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        return snapshot

    async def scenario() -> None:
        first = await capture("first", "atomic-first")
        second = await capture("second", "atomic-second")
        store = SQLiteAgentSnapshotStore(tmp_path / "atomic.sqlite3")
        await store.save_snapshot(first)
        first_access = AgentSnapshotAccess(
            snapshot=first.ref,
            binding_id=first.identity_binding.binding_id,
            authority_scope_fingerprint=first.authority_scope_fingerprint,
        )
        await store.pin_snapshot(
            AgentSnapshotPinRequest(
                operation_id="occupied-import-pin",
                access=first_access,
                owner="first-owner",
                reason="first-pin",
                retention_class=AgentSnapshotRetentionClass.RELEASE,
            )
        )
        second_access = AgentSnapshotAccess(
            snapshot=second.ref,
            binding_id=second.identity_binding.binding_id,
            authority_scope_fingerprint=second.authority_scope_fingerprint,
        )
        with pytest.raises(AgentSnapshotStoreConflict, match="another request"):
            await store.put_snapshot_and_pin(
                second,
                second.identity_binding,
                AgentSnapshotPinRequest(
                    operation_id="occupied-import-pin",
                    access=second_access,
                    owner="second-owner",
                    reason="second-pin",
                    retention_class=AgentSnapshotRetentionClass.RELEASE,
                ),
            )
        assert await store.load_snapshot(second.snapshot_root) is None

    _run(scenario())


def test_thin_bundle_transfers_only_changed_reachable_objects(tmp_path: Path) -> None:
    async def capture(
        *,
        root: Path,
        snapshot_store: InMemoryAgentSnapshotStore,
        knowledge_marker: str,
        capture_id: str,
    ):
        providers, object_store, subject, scope = _portable_fixture(
            root,
            knowledge_marker=knowledge_marker,
        )
        snapshot = await AgentSnapshotCoordinator(providers, store=snapshot_store).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id=capture_id,
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        return snapshot, object_store, subject, scope

    async def scenario() -> None:
        source_root = tmp_path / "source"
        source_snapshots = InMemoryAgentSnapshotStore()
        first, source_objects, subject, scope = await capture(
            root=source_root,
            snapshot_store=source_snapshots,
            knowledge_marker="procedure-1",
            capture_id="capture-v123",
        )
        source_bundle_coordinator = AgentBundleCoordinator(
            snapshot_store=source_snapshots,
            object_store=source_objects,
        )
        first_bundle_path = (tmp_path / "bundle-v123").resolve()
        first_export = await source_bundle_coordinator.export(
            operation_id="export-v123",
            access=AgentSnapshotAccess(
                snapshot=first.ref,
                binding_id=first.identity_binding.binding_id,
                authority_scope_fingerprint=scope,
            ),
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=first_bundle_path,
        )

        destination_objects = FileSystemAgentSnapshotObjectStore(
            (tmp_path / "destination" / "objects").resolve()
        )
        destination_snapshots = InMemoryAgentSnapshotStore()
        destination_coordinator = AgentBundleCoordinator(
            snapshot_store=destination_snapshots,
            object_store=destination_objects,
        )
        destination_scope = _digest("thin-destination-scope")
        await destination_coordinator.import_bundle(
            operation_id="import-v123",
            source=first_bundle_path,
            subject=subject,
            authority_scope_fingerprint=destination_scope,
            owner="n9-release",
        )

        second, _, _, _ = await capture(
            root=source_root,
            snapshot_store=source_snapshots,
            knowledge_marker="procedure-2",
            capture_id="capture-v124",
        )
        assert second.snapshot_root != first.snapshot_root
        first_nodes = {node.schema_id: node.digest for node in first.component_nodes()}
        second_nodes = {node.schema_id: node.digest for node in second.component_nodes()}
        changed_schema = "cayu.agent-snapshot.component.knowledge.v1"
        assert first_nodes[changed_schema] != second_nodes[changed_schema]
        assert {
            schema: digest for schema, digest in first_nodes.items() if schema != changed_schema
        } == {schema: digest for schema, digest in second_nodes.items() if schema != changed_schema}

        thin_path = (tmp_path / "bundle-v124-thin").resolve()
        thin_export = await source_bundle_coordinator.export(
            operation_id="export-v124-thin",
            access=AgentSnapshotAccess(
                snapshot=second.ref,
                binding_id=second.identity_binding.binding_id,
                authority_scope_fingerprint=scope,
            ),
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=thin_path,
            mode=AgentBundleMode.THIN,
            destination_inventory=AgentBundleInventory(
                object_digests=tuple(
                    sorted(reference.digest for reference in first_export.bundle.closure)
                )
            ),
        )
        assert thin_export.bundle.size_report.incremental_transfer_bytes < (
            thin_export.bundle.size_report.logical_closure_bytes
        )
        assert first.component(AgentSnapshotComponentKind.BODY).logical.fingerprint not in (
            thin_export.bundle.transferred_digests
        )
        assert (
            second.component(AgentSnapshotComponentKind.KNOWLEDGE).logical.fingerprint
            in thin_export.bundle.transferred_digests
        )

        imported = await destination_coordinator.import_bundle(
            operation_id="import-v124-thin",
            source=thin_path,
            subject=subject,
            authority_scope_fingerprint=destination_scope,
            owner="n9-release",
        )
        assert imported.snapshot_ref == second.ref
        assert set(imported.reused_digests)
        destination_inventory = await destination_objects.inventory()
        assert first.component(AgentSnapshotComponentKind.BODY).logical.fingerprint in (
            destination_inventory.object_digests
        )
        assert second.component(AgentSnapshotComponentKind.KNOWLEDGE).logical.fingerprint in (
            destination_inventory.object_digests
        )

    _run(scenario())
