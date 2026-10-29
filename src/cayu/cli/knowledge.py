"""`cayu knowledge`: read knowledge entries from a local store or a Cayu Cloud Agent."""

from __future__ import annotations

import argparse
import json
from typing import Any

from cayu.cli._cloud_agent_api import open_cloud_agent_api
from cayu.cli._knowledge_readers import (
    MAX_KNOWLEDGE_PAGE,
    ApiKnowledgeReader,
    KnowledgeListRequest,
    KnowledgeReader,
    open_store_knowledge_reader,
)
from cayu.cli._output import (
    add_output_options,
    print_table,
    terminal_safe,
)
from cayu.cli._read_targets import ReadTarget, add_read_target_options, run_read_command
from cayu.cli.session import _parse_labels, _redact_sensitive
from cayu.knowledge.records import KnowledgeStatus

CLI_SCHEMA_VERSION = "1"
_PREVIEW_CHARS = 60
_LISTED_STATUSES = (KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED)


def add_knowledge_parser(subparsers: Any) -> None:
    """Register the ``cayu knowledge`` command group."""

    knowledge = subparsers.add_parser(
        "knowledge",
        help="Inspect kept and pending knowledge entries.",
        description=(
            "Read knowledge entries from a local store (--sqlite, --postgres, or the "
            "project's store) or from a Cayu Cloud Agent (--application NAME). Start with "
            "`cayu knowledge list`."
        ),
    )
    commands = knowledge.add_subparsers(dest="knowledge_command", required=True)

    list_parser = commands.add_parser(
        "list",
        help="List knowledge entries.",
        description=(
            "List knowledge entries with a short preview. `cayu knowledge show ENTRY_ID` "
            "prints one with its text."
        ),
    )
    list_parser.add_argument(
        "--status",
        choices=tuple(status.value for status in _LISTED_STATUSES),
        default=KnowledgeStatus.ACTIVE.value,
        help=(
            "Kept (active, the default), pending proposals, or archived entries "
            "(archived only from a local store)."
        ),
    )
    list_parser.add_argument("--namespace", help="Only entries in this namespace.")
    list_parser.add_argument(
        "--kind",
        action="append",
        default=[],
        help="Only entries of this kind; repeat for any of several kinds.",
    )
    list_parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Only entries with this label; repeat to require several.",
    )
    list_parser.add_argument(
        "--limit",
        type=_page_limit,
        default=50,
        help=f"Entries per page, 1-{MAX_KNOWLEDGE_PAGE} (default: 50).",
    )
    list_parser.add_argument("--cursor", help="Continue from a listing's next_cursor.")
    add_read_target_options(list_parser)
    add_output_options(list_parser, formats=("table", "json", "jsonl"), default="table")

    show_parser = commands.add_parser(
        "show",
        help="Show one knowledge entry with its text.",
        description="Show one knowledge entry, whatever its status, with its text and chunks.",
    )
    show_parser.add_argument("entry_id")
    add_read_target_options(show_parser)
    add_output_options(show_parser, formats=("table", "json"), default="table")


def run_knowledge(args: argparse.Namespace) -> int:
    return run_read_command(
        args,
        _run_knowledge_command,
        error_code="KNOWLEDGE_INSPECTION_FAILED",
        schema_version=CLI_SCHEMA_VERSION,
    )


async def _run_knowledge_command(args: argparse.Namespace, target: ReadTarget) -> int:
    reader = open_knowledge_reader(target)
    try:
        if args.knowledge_command == "list":
            return await _list_entries(args, reader)
        return await _show_entry(args, reader)
    finally:
        await reader.close()


def open_knowledge_reader(target: ReadTarget) -> KnowledgeReader:
    """The knowledge reader for a resolved ``--sqlite``/``--postgres``/``--application``."""

    if target.store is not None:
        return open_store_knowledge_reader(target.store)
    assert target.application is not None
    api = open_cloud_agent_api(
        target.application, context=target.context, timeout_seconds=target.timeout_seconds
    )
    return ApiKnowledgeReader(api, source={"kind": "cayu_cloud", "application": target.application})


async def _list_entries(args: argparse.Namespace, reader: KnowledgeReader) -> int:
    namespace = None if args.namespace is None else args.namespace.strip() or None
    kinds = tuple(dict.fromkeys(kind.strip() for kind in args.kind if kind.strip()))
    page = await reader.list_entries(
        KnowledgeListRequest(
            status=KnowledgeStatus(args.status),
            namespace=namespace,
            kinds=kinds,
            labels=_parse_labels(args.label),
            limit=args.limit,
            cursor=args.cursor,
        )
    )
    payload = _redact_sensitive(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "source": reader.source,
            "status": args.status,
            "entries": page.entries,
            "next_cursor": page.next_cursor,
            "has_more": page.next_cursor is not None or page.truncated,
        }
    )
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif args.output_format == "jsonl":
        for entry in payload["entries"]:
            print(json.dumps(entry, ensure_ascii=False, sort_keys=True))
    else:
        if payload["entries"]:
            print_table(
                ("entry_id", "kind", "namespace", "labels", "updated_at", "preview"),
                [_table_row(entry) for entry in payload["entries"]],
            )
        else:
            print(
                "No knowledge entries on this page."
                if page.next_cursor
                else "No knowledge entries."
            )
        if page.next_cursor is not None:
            print(f"More entries follow; add --cursor {terminal_safe(page.next_cursor)}")
        elif page.truncated:
            print("More entries exist than this page shows; narrow the filters.")
    return 0


async def _show_entry(args: argparse.Namespace, reader: KnowledgeReader) -> int:
    entry = await reader.load_entry(args.entry_id)
    if entry is None:
        raise ValueError(f"Knowledge entry not found: {args.entry_id}")
    payload = _redact_sensitive(
        {"schema_version": CLI_SCHEMA_VERSION, "source": reader.source, "entry": entry}
    )
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    entry = payload["entry"]
    fields = (
        "entry_id",
        "revision",
        "status",
        "namespace",
        "kind",
        "title",
        "labels",
        "visibility",
        "source_type",
        "source_id",
        "source_uri",
        "created_by",
        "created_at",
        "updated_at",
        "expires_at",
    )
    print_table(
        ("field", "value"),
        [
            {"field": name, "value": _display(entry.get(name))}
            for name in fields
            if entry.get(name) not in (None, {}, [])
        ],
    )
    text = entry.get("text")
    if isinstance(text, str) and text:
        print("\nText")
        print(terminal_safe(text, keep_lines=True))
    chunks = entry.get("chunks")
    if isinstance(chunks, list) and len(chunks) > 1:
        print(f"\n{len(chunks)} chunks; --json prints each.")
    return 0


def _table_row(entry: dict[str, Any]) -> dict[str, Any]:
    preview = entry.get("title") or entry.get("text_preview") or ""
    preview = " ".join(str(preview).split())
    if len(preview) > _PREVIEW_CHARS:
        preview = preview[: _PREVIEW_CHARS - 1] + "…"
    return {
        "entry_id": entry.get("entry_id"),
        "kind": entry.get("kind"),
        "namespace": entry.get("namespace"),
        "labels": _display(entry.get("labels")) or None,
        "updated_at": entry.get("updated_at"),
        "preview": preview or None,
    }


def _display(value: object) -> str:
    if isinstance(value, dict):
        return ",".join(f"{key}={item}" for key, item in sorted(value.items()))
    return "" if value is None else str(value)


def _page_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if not 1 <= limit <= MAX_KNOWLEDGE_PAGE:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_KNOWLEDGE_PAGE}")
    return limit
