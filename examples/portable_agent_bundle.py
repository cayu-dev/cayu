"""Export, import, and freshly materialize one complete downloadable Cayu agent.

Run each command in a separate process to exercise the portable boundary::

    uv run python examples/portable_agent_bundle.py export --root /tmp/cayu-agent-demo
    uv run python examples/portable_agent_bundle.py inspect --root /tmp/cayu-agent-demo
    uv run python examples/portable_agent_bundle.py import --root /tmp/cayu-agent-demo
    uv run python examples/portable_agent_bundle.py materialize --root /tmp/cayu-agent-demo

The downloadable bundle is one ordinary ``compound-agent-v123.cayu`` file.  It
contains authenticated agent-state objects, never credentials.  The canonical
unpacked directory remains available for CAS and debugging.  The demo application's explicit materialization
authority resolves the declared hosted-model binding afresh and starts with a
new runtime/session/budget/lease/scratch identity and an empty tool-discovery
grant view.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from cayu import (
    AGENT_BUNDLE_INDEX_FILENAME,
    AgentBundle,
    AgentBundleCoordinator,
    AgentBundleMaterializationAuthority,
    AgentBundleMaterializationAuthorization,
    AgentBundleMaterializationRequest,
    AgentExternalBindingKind,
    AgentExternalBindingRequirement,
    AgentExternalBindingResolution,
    AgentMaterializationFreshIdentities,
    AgentSnapshotAccess,
    AgentSnapshotCaptureRequest,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentSelector,
    AgentSnapshotCoordinator,
    AgentSnapshotExecutionProfileComponent,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotIdentityBinding,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterializationMode,
    AgentSnapshotProfile,
    AgentSnapshotSessionDisposition,
    AgentSnapshotSubject,
    AgentSnapshotTrialStateMode,
    FileSystemAgentSnapshotObjectStore,
    PortableAgentSnapshotComponentProvider,
    SQLiteAgentSnapshotStore,
    agent_snapshot_component_package,
    inspect_agent_bundle_container,
    unpack_agent_bundle_container,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


class _DemoMaterializationAuthority(AgentBundleMaterializationAuthority):
    """Trusted demo-app boundary; production apps verify real provider receipts here."""

    async def authorize_materialization(
        self,
        request: AgentBundleMaterializationRequest,
    ) -> AgentBundleMaterializationAuthorization:
        expected_names = {"primary-model", "primary-model-credential"}
        declared_names = {requirement.name for requirement in request.bundle.external_bindings}
        if declared_names != expected_names:
            raise RuntimeError("The demo application does not authorize this binding profile.")
        return AgentBundleMaterializationAuthorization.create(
            request=request,
            authority_fingerprint=_digest("portable-demo-local-materialization-authority-v1"),
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


def _paths(root: Path) -> dict[str, Path]:
    return {
        "source_db": root / "source-snapshots.sqlite3",
        "source_objects": root / "source-objects",
        "container": root / "compound-agent-v123.cayu",
        "unpacked": root / "compound-agent-v123.cayu.d",
        "destination_db": root / "destination-snapshots.sqlite3",
        "destination_objects": root / "destination-objects",
        "materialized": root / "materialized",
    }


def _components(
    root: Path,
) -> tuple[
    tuple[PortableAgentSnapshotComponentProvider, ...],
    FileSystemAgentSnapshotObjectStore,
    AgentSnapshotSubject,
]:
    object_store = FileSystemAgentSnapshotObjectStore(root)
    profile = AgentSnapshotProfile.REUSABLE_AGENT
    provider_id = "cayu.portable-agent-demo.v1"
    model_binding = AgentExternalBindingRequirement(
        kind=AgentExternalBindingKind.HOSTED_PROVIDER,
        name="primary-model",
        requirement_fingerprint=_digest("openai-compatible-model:v1"),
    )
    credential_binding = AgentExternalBindingRequirement(
        kind=AgentExternalBindingKind.CREDENTIAL,
        name="primary-model-credential",
        requirement_fingerprint=_digest("credential-class:openai-compatible:v1"),
    )
    definitions: dict[AgentSnapshotComponentKind, dict[str, Any]] = {
        AgentSnapshotComponentKind.BODY: {
            "payload": {"entrypoint": "agent.py"},
            "files": {"agent.py": b"def answer():\n    return 42\n"},
            "types": {"agent.py": "text/x-python"},
        },
        AgentSnapshotComponentKind.EXECUTION_PROFILE: {
            "payload": {"provider_target": "resolve-on-materialize"},
        },
        AgentSnapshotComponentKind.ROLE_PROFILE: {
            "payload": {"role": "benchmark-solving-agent"},
        },
        AgentSnapshotComponentKind.TOOL_CATALOGUE: {
            "payload": {"tools": ["search_tools", "call_tool", "exec_command"]},
        },
        AgentSnapshotComponentKind.TOOL_EXPOSURE_POLICY: {
            "payload": {"direct": ["search_tools", "call_tool"]},
        },
        AgentSnapshotComponentKind.KNOWLEDGE: {
            "payload": {"authorized_view": "knowledge-change:17"},
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
            "payload": {"promoted": ["model-card.json"]},
            "files": {"model-card.json": b'{"validated":true}\n'},
            "types": {"model-card.json": "application/json"},
        },
        AgentSnapshotComponentKind.ENVIRONMENT: {
            "payload": {
                "runtime_source_revision": "portable-agent-demo-v1",
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
            "external_bindings": (credential_binding, model_binding),
        },
        AgentSnapshotComponentKind.STANDING_POLICY: {
            "payload": {"command_policy": "governed", "egress": "virtual"},
        },
    }
    packages = {}
    blobs = {}
    for kind, definition in definitions.items():
        package, package_blobs = agent_snapshot_component_package(
            kind=kind,
            provider_id=provider_id,
            component_schema=f"cayu.demo.{kind.value}.v1",
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
    execution_profile = AgentSnapshotExecutionProfileRef(
        schema_version=1,
        fingerprint=packages[AgentSnapshotComponentKind.EXECUTION_PROFILE].digest,
        components=(
            AgentSnapshotExecutionProfileComponent(
                name="runtime",
                fingerprint=_digest("portable-demo-runtime-profile"),
                availability="available",
            ),
        ),
    )
    providers = tuple(
        PortableAgentSnapshotComponentProvider(
            package,
            blobs[kind],
            object_store=object_store,
            materialization_root=root.parent / "source-materialized",
            execution_profile=(
                execution_profile if kind is AgentSnapshotComponentKind.EXECUTION_PROFILE else None
            ),
        )
        for kind, package in packages.items()
    )
    subject = AgentSnapshotSubject(
        agent_id="compound-agent-v123",
        application_id="compound",
        project_id="n9",
        body_release=AgentSnapshotLogicalRef(
            fingerprint=packages[AgentSnapshotComponentKind.BODY].digest,
            revision=f"component:{packages[AgentSnapshotComponentKind.BODY].digest}",
        ),
    )
    return providers, object_store, subject


async def _export(root: Path) -> dict[str, object]:
    paths = _paths(root)
    providers, object_store, subject = _components(paths["source_objects"])
    scope = _digest("portable-demo-source-scope")
    snapshot_store = SQLiteAgentSnapshotStore(paths["source_db"])
    snapshot = await AgentSnapshotCoordinator(providers, store=snapshot_store).capture(
        AgentSnapshotCaptureRequest(
            capture_request_id="portable-demo-capture",
            subject=subject,
            authority_scope_fingerprint=scope,
            components=tuple(
                AgentSnapshotComponentSelector(kind=kind)
                for kind in sorted((provider.kind for provider in providers), key=str)
            ),
        )
    )
    receipt = await AgentBundleCoordinator(
        snapshot_store=snapshot_store,
        object_store=object_store,
    ).export_container(
        operation_id="portable-demo-export",
        access=AgentSnapshotAccess(
            snapshot=snapshot.ref,
            binding_id=snapshot.identity_binding.binding_id,
            authority_scope_fingerprint=scope,
        ),
        profile=AgentSnapshotProfile.REUSABLE_AGENT,
        destination=paths["container"],
    )
    return receipt.model_dump(mode="json")


def _inspect(root: Path) -> dict[str, object]:
    return inspect_agent_bundle_container(_paths(root)["container"]).model_dump(mode="json")


def _bundle(container: Path, unpacked: Path) -> AgentBundle:
    unpack_agent_bundle_container(container, unpacked)
    return AgentBundle.model_validate_json((unpacked / AGENT_BUNDLE_INDEX_FILENAME).read_bytes())


async def _import(root: Path) -> dict[str, object]:
    paths = _paths(root)
    _, _, subject = _components(paths["source_objects"])
    receipt = await AgentBundleCoordinator(
        snapshot_store=SQLiteAgentSnapshotStore(paths["destination_db"]),
        object_store=FileSystemAgentSnapshotObjectStore(paths["destination_objects"]),
    ).import_container(
        operation_id="portable-demo-import",
        source=paths["container"],
        subject=subject.model_copy(update={"agent_id": "compound-agent-imported"}),
        authority_scope_fingerprint=_digest("portable-demo-destination-scope"),
        owner="portable-demo",
    )
    return receipt.model_dump(mode="json")


async def _materialize(root: Path) -> dict[str, object]:
    paths = _paths(root)
    bundle = _bundle(paths["container"], paths["unpacked"])
    _, _, subject = _components(paths["source_objects"])
    scope = _digest("portable-demo-destination-scope")
    destination_subject = subject.model_copy(update={"agent_id": "compound-agent-imported"})
    binding = AgentSnapshotIdentityBinding.create(
        subject=destination_subject.model_copy(
            update={
                "body_release": subject.body_release.model_copy(update={"scope_fingerprint": scope})
            }
        ),
        snapshot=bundle.snapshot_ref,
        authority_scope_fingerprint=scope,
    )
    coordinator = AgentBundleCoordinator(
        snapshot_store=SQLiteAgentSnapshotStore(paths["destination_db"]),
        object_store=FileSystemAgentSnapshotObjectStore(paths["destination_objects"]),
        materialization_authority=_DemoMaterializationAuthority(),
    )
    receipt = await coordinator.materialize(
        AgentBundleMaterializationRequest(
            operation_id="portable-demo-materialize",
            bundle=bundle,
            access=AgentSnapshotAccess(
                snapshot=bundle.snapshot_ref,
                binding_id=binding.binding_id,
                authority_scope_fingerprint=scope,
            ),
            mode=AgentSnapshotMaterializationMode.FORK_AS_SEED,
            candidate_id="portable-demo-candidate",
            trial_id="portable-demo-trial",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        ),
        materialization_root=paths["materialized"],
    )
    return receipt.model_dump(mode="json")


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("export", "inspect", "import", "materialize"))
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if arguments.command == "inspect":
        value = _inspect(root)
    else:
        result = {
            "export": _export,
            "import": _import,
            "materialize": _materialize,
        }[arguments.command]
        value = await result(root)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    asyncio.run(_main())
