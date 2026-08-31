from __future__ import annotations

import argparse
import asyncio
import json
from hashlib import sha256
from pathlib import Path

from cayu.agent_bundle_containers import (
    inspect_agent_bundle_container,
    pack_agent_bundle,
    unpack_agent_bundle_container,
)
from cayu.agent_bundles import (
    AgentBundleCoordinator,
    AgentBundleError,
    AgentSnapshotProfile,
    FileSystemAgentSnapshotObjectStore,
)
from cayu.agent_snapshots import (
    AgentSnapshotAccess,
    AgentSnapshotRef,
    AgentSnapshotRetentionClass,
    AgentSnapshotSubject,
    SQLiteAgentSnapshotStore,
)

_SUBJECT_DOCUMENT_MAX_BYTES = 64 * 1024


def _add_store_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--snapshot-store",
        type=Path,
        required=True,
        help="Durable SQLite AgentSnapshot store.",
    )
    parser.add_argument(
        "--object-store",
        type=Path,
        required=True,
        help="Durable filesystem AgentSnapshot object store.",
    )


def _derived_operation_id(kind: str, material: dict[str, str]) -> str:
    digest = sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"cayu-cli-agent-bundle-{kind}:{digest}"


def _load_subject(path: Path) -> AgentSnapshotSubject:
    with path.open("rb") as stream:
        document = stream.read(_SUBJECT_DOCUMENT_MAX_BYTES + 1)
    if len(document) > _SUBJECT_DOCUMENT_MAX_BYTES:
        raise AgentBundleError("bundle_import_subject_too_large")
    return AgentSnapshotSubject.model_validate_json(document)


def add_agent_parser(subparsers: argparse._SubParsersAction) -> None:
    agent = subparsers.add_parser(
        "agent",
        help="Manage portable Cayu agents.",
        description="Manage portable Cayu agent representations.",
    )
    commands = agent.add_subparsers(dest="agent_command", required=True)
    bundle = commands.add_parser(
        "bundle",
        help="Inspect and convert AgentBundle representations.",
        description=(
            "Inspect or convert the single-file .cayu transport and its canonical "
            "unpacked AgentBundle directory."
        ),
    )
    actions = bundle.add_subparsers(dest="bundle_command", required=True)

    export = actions.add_parser(
        "export",
        help="Export an authorized durable snapshot as one .cayu file.",
        description=(
            "Export and protect an authorized snapshot from durable SQLite/CAS storage, then "
            "publish one deterministic full .cayu file."
        ),
    )
    _add_store_arguments(export)
    export.add_argument("--snapshot-root", required=True, help="Authorized snapshot root SHA-256.")
    export.add_argument("--binding-id", required=True, help="Authorized binding SHA-256.")
    export.add_argument(
        "--authority-scope-fingerprint",
        required=True,
        help="Authorized source scope SHA-256.",
    )
    export.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in AgentSnapshotProfile),
        default=AgentSnapshotProfile.REUSABLE_AGENT.value,
        help="Portable snapshot profile (default: reusable_agent).",
    )
    export.add_argument("--operation-id", help="Stable retry identity; derived when omitted.")
    export.add_argument("--output", type=Path, required=True, help="Destination .cayu file.")

    inspect = actions.add_parser(
        "inspect",
        help="Validate and inspect one .cayu file.",
        description=(
            "Validate every transferred entry in one .cayu file and report bounded safe "
            "metadata before import or materialization."
        ),
    )
    inspect.add_argument("source", type=Path, help="Source .cayu file.")
    inspect.add_argument("--json", action="store_true", help="Emit canonical JSON metadata.")

    import_parser = actions.add_parser(
        "import",
        help="Validate, import, and pin a .cayu file into durable stores.",
        description=(
            "Validate one .cayu file, import its exact closure into SQLite/CAS storage, and "
            "atomically publish and pin the rebound snapshot root."
        ),
    )
    import_parser.add_argument("source", type=Path, help="Source .cayu file.")
    _add_store_arguments(import_parser)
    import_parser.add_argument(
        "--subject",
        type=Path,
        required=True,
        help="Destination AgentSnapshotSubject JSON document.",
    )
    import_parser.add_argument(
        "--authority-scope-fingerprint",
        required=True,
        help="Authorized destination scope SHA-256.",
    )
    import_parser.add_argument("--owner", required=True, help="Durable pin owner.")
    import_parser.add_argument(
        "--retention-class",
        choices=tuple(item.value for item in AgentSnapshotRetentionClass),
        default=AgentSnapshotRetentionClass.RELEASE.value,
        help="Durable pin retention class (default: release).",
    )
    import_parser.add_argument(
        "--operation-id", help="Stable retry identity; derived when omitted."
    )

    unpack = actions.add_parser(
        "unpack",
        help="Unpack one .cayu file without changing bundle identity.",
        description=(
            "Validate and convert one .cayu file to its canonical directory for CAS, "
            "debugging, or a later governed import."
        ),
    )
    unpack.add_argument("source", type=Path, help="Source .cayu file.")
    unpack.add_argument("--destination", type=Path, required=True)

    pack = actions.add_parser(
        "pack",
        help="Pack one canonical AgentBundle directory as .cayu.",
        description=(
            "Validate and convert one canonical AgentBundle directory into deterministic "
            ".cayu bytes for copying or download."
        ),
    )
    pack.add_argument("source", type=Path, help="Canonical unpacked AgentBundle directory.")
    pack.add_argument("--output", type=Path, required=True, help="Destination .cayu file.")


def _inspection_payload(inspection) -> dict[str, object]:
    return inspection.model_dump(mode="json")


def _print_inspection(inspection, *, as_json: bool) -> None:
    payload = _inspection_payload(inspection)
    if as_json:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    print(f"snapshot root: {inspection.snapshot_root}")
    print(f"bundle id: {inspection.bundle_id}")
    print(f"profile: {inspection.profile}")
    print(f"mode: {inspection.mode.value}")
    print(f"transport sha256: {inspection.transport_sha256}")
    print(f"container bytes: {inspection.container_bytes}")
    print(f"logical bytes: {inspection.logical_closure_bytes}")
    print(f"transferred bytes: {inspection.transferred_bytes}")
    print(f"objects: {inspection.object_count}")
    print(f"transferred objects: {inspection.transferred_object_count}")
    dependency = (
        inspection.destination_inventory_fingerprint
        if inspection.requires_preexisting_objects
        else "none (full self-contained bundle)"
    )
    print(f"destination inventory dependency: {dependency}")
    bindings = ", ".join(inspection.unresolved_external_bindings) or "none"
    print(f"unresolved external bindings: {bindings}")


async def _export_container(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    access = AgentSnapshotAccess(
        snapshot=AgentSnapshotRef(snapshot_root=args.snapshot_root),
        binding_id=args.binding_id,
        authority_scope_fingerprint=args.authority_scope_fingerprint,
    )
    operation_id = args.operation_id or _derived_operation_id(
        "export",
        {
            "snapshot_root": access.snapshot.snapshot_root,
            "binding_id": access.binding_id,
            "authority_scope_fingerprint": access.authority_scope_fingerprint,
            "profile": args.profile,
            "output": str(output),
        },
    )
    receipt = await AgentBundleCoordinator(
        snapshot_store=SQLiteAgentSnapshotStore(args.snapshot_store.resolve()),
        object_store=FileSystemAgentSnapshotObjectStore(args.object_store.resolve()),
    ).export_container(
        operation_id=operation_id,
        access=access,
        profile=AgentSnapshotProfile(args.profile),
        destination=output,
    )
    _print_inspection(receipt.inspection, as_json=True)
    return 0


async def _import_container(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    subject_path = args.subject.resolve()
    subject = _load_subject(subject_path)
    operation_id = args.operation_id or _derived_operation_id(
        "import",
        {
            "source": str(source),
            "subject": json.dumps(subject.model_dump(mode="json"), sort_keys=True),
            "authority_scope_fingerprint": args.authority_scope_fingerprint,
            "owner": args.owner,
            "retention_class": args.retention_class,
        },
    )
    receipt = await AgentBundleCoordinator(
        snapshot_store=SQLiteAgentSnapshotStore(args.snapshot_store.resolve()),
        object_store=FileSystemAgentSnapshotObjectStore(args.object_store.resolve()),
    ).import_container(
        operation_id=operation_id,
        source=source,
        subject=subject,
        authority_scope_fingerprint=args.authority_scope_fingerprint,
        owner=args.owner,
        retention_class=AgentSnapshotRetentionClass(args.retention_class),
    )
    print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True, separators=(",", ":")))
    return 0


def run_agent(args: argparse.Namespace) -> int:
    if args.agent_command != "bundle":
        raise RuntimeError(f"unsupported agent command: {args.agent_command}")
    try:
        if args.bundle_command == "export":
            return asyncio.run(_export_container(args))
        if args.bundle_command == "pack":
            receipt = pack_agent_bundle(args.source.resolve(), args.output.resolve())
            _print_inspection(receipt.inspection, as_json=True)
            return 0
        if args.bundle_command == "inspect":
            inspection = inspect_agent_bundle_container(args.source.resolve())
            _print_inspection(inspection, as_json=args.json)
            return 0
        if args.bundle_command == "import":
            return asyncio.run(_import_container(args))
        if args.bundle_command == "unpack":
            receipt = unpack_agent_bundle_container(
                args.source.resolve(),
                args.destination.resolve(),
            )
            _print_inspection(receipt.inspection, as_json=True)
            return 0
    except (AgentBundleError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True, separators=(",", ":")))
        return 2
    raise RuntimeError(f"unsupported bundle command: {args.bundle_command}")


__all__ = ["add_agent_parser", "run_agent"]
