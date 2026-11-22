"""`cayu knowledge`: read, add and review knowledge entries, locally or on a Cayu Cloud Agent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
from cayu.storage.knowledge_review import (
    DEFAULT_KNOWLEDGE_CLUSTER_ENTRIES as DEFAULT_GROUPED_PROPOSALS,
)
from cayu.storage.knowledge_review import (
    MAX_KNOWLEDGE_CLUSTER_ENTRIES as MAX_GROUPED_PROPOSALS,
)

CLI_SCHEMA_VERSION = "1"
_PREVIEW_CHARS = 60
_LISTED_STATUSES = (KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING, KnowledgeStatus.ARCHIVED)


def add_knowledge_parser(subparsers: Any) -> None:
    """Register the ``cayu knowledge`` command group."""

    knowledge = subparsers.add_parser(
        "knowledge",
        help="Inspect and add knowledge, keep or drop proposals, and archive kept entries.",
        description=(
            "Read knowledge entries from a local store (--sqlite, --postgres, or the "
            "project's store) or from a Cayu Cloud Agent (--application NAME). Start with "
            "`cayu knowledge list`."
        ),
    )
    commands = knowledge.add_subparsers(dest="knowledge_command", required=True)

    from cayu.cli._control_plane_actions import add_action_target_options

    for name, summary, detail in (
        (
            "keep",
            "Keep a pending knowledge proposal.",
            "Keep a pending knowledge proposal, so knowledge search finds it.",
        ),
        (
            "drop",
            "Drop a pending knowledge proposal.",
            "Drop a pending knowledge proposal: it's archived, never used.",
        ),
        (
            "archive",
            "Archive a kept knowledge entry, with the reason.",
            "Archive a kept (active) knowledge entry that is stale or wrong, so knowledge "
            "search no longer finds it. The archived entry records the reason and who "
            "archived it.",
        ),
    ):
        action = commands.add_parser(
            name,
            help=summary,
            description=(
                f"{detail} Inspect it first with `cayu knowledge show ENTRY_ID`. "
                "Mutations use the running agent's review permissions."
            ),
        )
        action.add_argument("entry_id")
        action.add_argument(
            "--operation-id",
            # Keep has always required one; drop and archive take one to make a retry safe.
            required=name == "keep",
            help=(
                "Retain this ID to retry the same operation: a retry replays it instead of "
                "reporting a conflict."
            ),
        )
        action.add_argument(
            "--reason",
            required=name == "archive",
            help=(
                "Why the entry is archived, recorded with it."
                if name == "archive"
                else f"Why you {name} it, recorded with the decision."
            ),
        )
        if name == "archive":
            action.add_argument(
                "--revision",
                type=_revision,
                help=(
                    "Archive only if the entry is still at this revision (from "
                    "`cayu knowledge show`)."
                ),
            )
        add_action_target_options(action)

    add_parser = commands.add_parser(
        "add",
        help="Add a knowledge entry, kept or for review.",
        description=(
            "Add one knowledge entry to a running agent (--server-url) or a Cayu Cloud "
            "Agent (--application NAME), through the same store rules as the library. The "
            "entry is kept (active) at once, so knowledge search finds it; with --pending it "
            "is a proposal that someone else keeps or drops (`cayu knowledge keep`/`drop`), "
            "since whoever adds a proposal can't keep it too (a server without "
            "authentication refuses --pending for that reason). When the agent's knowledge "
            "scope pins knowledge to a conversation's labels (session_label_keys, such as "
            "`organization`), the entry must carry each one: pass --label organization=ID. "
            "Adding the same text, kind and labels again returns the entry it added "
            '("created": false). Prints the entry as one JSON document.'
        ),
    )
    text_source = add_parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument("text", nargs="?", help="The entry's text.")
    text_source.add_argument(
        "--file",
        type=Path,
        metavar="PATH",
        help="Read the entry's text from this UTF-8 file; - reads standard input.",
    )
    add_parser.add_argument(
        "--kind", help="What the entry is, such as policy or faq (default: fact)."
    )
    add_parser.add_argument("--title", help="A short title, shown in lists.")
    add_parser.add_argument(
        "--namespace",
        help=(
            "The namespace to add to (default: the agent's review namespace, or its only "
            "allowed one)."
        ),
    )
    add_parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="A label on the entry, such as organization=ID; repeat for several.",
    )
    add_parser.add_argument("--source-uri", help="Where the text comes from, recorded with it.")
    add_parser.add_argument(
        "--pending",
        action="store_true",
        help="Add it as a proposal for review instead of keeping it at once.",
    )
    add_action_target_options(add_parser, output_format="json")

    duplicates = commands.add_parser(
        "drop-duplicates",
        help="Drop pending proposals as duplicates of one entry, in one step.",
        description=(
            "Decide a cluster of similar proposals (from `cayu knowledge list --status "
            "pending --group`): keep one with --keep and drop the rest as its duplicates, "
            "or drop them all as duplicates of an entry already kept with --of. Each entry "
            "is kept or dropped on its own, recording the reviewer and its reason "
            '("Duplicate of ENTRY_ID." and --reason). Run the same command again to '
            "finish after a failure: decisions already made replay."
        ),
    )
    duplicates.add_argument("entry_ids", nargs="+", metavar="ENTRY_ID", help="Proposals to drop.")
    target = duplicates.add_mutually_exclusive_group(required=True)
    target.add_argument("--keep", metavar="ENTRY_ID", help="Keep this proposal first.")
    target.add_argument("--of", metavar="ENTRY_ID", help="The kept entry they duplicate.")
    duplicates.add_argument("--reason", help="Why, recorded with every decision.")
    duplicates.add_argument(
        "--operation-id",
        required=True,
        help="Retain this ID to retry: the same command replays decisions already made.",
    )
    add_action_target_options(duplicates)

    list_parser = commands.add_parser(
        "list",
        help="List knowledge entries.",
        description=(
            "List knowledge entries with a short preview: an entry's title, or the start "
            "of its text when it has none. `--json` gives both, as `title` and "
            "`text_preview` (the start of the text). `cayu knowledge show ENTRY_ID` "
            "prints one with its text."
        ),
    )
    list_parser.add_argument(
        "--status",
        choices=tuple(status.value for status in _LISTED_STATUSES),
        default=KnowledgeStatus.ACTIVE.value,
        help=(
            "Kept (active, the default), pending proposals, or archived entries (kept "
            "entries archived and proposals dropped; from an Agent, with full access)."
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
    list_parser.add_argument(
        "--group",
        action="store_true",
        help=(
            "With --status pending: group similar proposals, each group with the kept "
            "entries it resembles, the groups with the most sources first. Reads up to "
            "--max-proposals; grouping only orders them."
        ),
    )
    list_parser.add_argument(
        "--max-proposals",
        type=_group_limit,
        default=DEFAULT_GROUPED_PROPOSALS,
        help=(
            f"With --group, how many proposals to read, 1-{MAX_GROUPED_PROPOSALS} "
            f"(default: {DEFAULT_GROUPED_PROPOSALS})."
        ),
    )
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

    lookups = commands.add_parser(
        "lookups",
        help="Check the stored knowledge lookup counts.",
        description=(
            "Knowledge lists show each entry's lookups and last lookup time from counts "
            "the session store keeps beside its events. `cayu knowledge lookups check` "
            "recomputes every count from the retained knowledge.lookup events and lists "
            "any entry whose stored count differs; it exits 1 when one does."
        ),
    )
    lookup_commands = lookups.add_subparsers(dest="lookups_command", required=True)
    check_parser = lookup_commands.add_parser(
        "check",
        help="Recompute lookup counts from events and compare them with the stored ones.",
        description=(
            "Read a local session store (--sqlite, --postgres, or the project's store) "
            "read-only. Run it while the store isn't taking lookups, since a lookup "
            "recorded between the two reads shows up as a difference."
        ),
    )
    store_target = check_parser.add_mutually_exclusive_group()
    store_target.add_argument("--sqlite", metavar="PATH", help="Check this SQLite database.")
    store_target.add_argument("--postgres", metavar="DSN", help="Check this Postgres database.")
    check_parser.set_defaults(
        application=None,
        server_url=None,
        context=None,
        authorization_env="CAYU_API_AUTHORIZATION",
        timeout_seconds=30.0,
        wake_timeout_seconds=180.0,
    )
    add_output_options(check_parser, formats=("table", "json"), default="table")


def run_knowledge(args: argparse.Namespace) -> int:
    if args.knowledge_command == "drop-duplicates":
        from cayu.cli._knowledge_duplicates import run_drop_duplicates

        return run_drop_duplicates(args)
    if args.knowledge_command in {"add", "keep", "drop", "archive"}:
        from cayu.cli._control_plane_actions import run_action

        return run_action(args)
    return run_read_command(
        args,
        _run_knowledge_command,
        error_code="KNOWLEDGE_INSPECTION_FAILED",
        schema_version=CLI_SCHEMA_VERSION,
    )


async def _run_knowledge_command(args: argparse.Namespace, target: ReadTarget) -> int:
    if args.knowledge_command == "lookups":
        return await _check_lookups(args, target)
    reader = open_knowledge_reader(target)
    try:
        if args.knowledge_command == "list":
            return await _list_entries(args, reader)
        return await _show_entry(args, reader)
    finally:
        await reader.close()


def open_knowledge_reader(target: ReadTarget) -> KnowledgeReader:
    """The knowledge reader for a resolved ``--sqlite``/``--postgres``/``--application``."""

    if target.server_url is not None:
        from cayu.cli._cloud_agent_api import CayuApi
        from cayu.cli.diagnostics import read_authorization

        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
        return ApiKnowledgeReader(api, source={"kind": "server", "url": target.server_url})
    if target.store is not None:
        return open_store_knowledge_reader(target.store)
    assert target.application is not None
    api = open_cloud_agent_api(
        target.application,
        context=target.context,
        timeout_seconds=target.timeout_seconds,
        wake_timeout_seconds=target.wake_timeout_seconds,
        offers_wake_option=True,
    )
    return ApiKnowledgeReader(api, source={"kind": "cayu_cloud", "application": target.application})


async def _list_entries(args: argparse.Namespace, reader: KnowledgeReader) -> int:
    if args.group:
        return await _list_clusters(args, reader)
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


async def _list_clusters(args: argparse.Namespace, reader: KnowledgeReader) -> int:
    if args.status != KnowledgeStatus.PENDING.value:
        raise ValueError("--group lists pending proposals: add --status pending.")
    if args.cursor is not None:
        raise ValueError("--group reads one bounded set of proposals; it takes no --cursor.")
    namespace = None if args.namespace is None else args.namespace.strip() or None
    kinds = tuple(dict.fromkeys(kind.strip() for kind in args.kind if kind.strip()))
    grouped = await reader.list_clusters(
        KnowledgeListRequest(
            status=KnowledgeStatus.PENDING,
            namespace=namespace,
            kinds=kinds,
            labels=_parse_labels(args.label),
            limit=args.max_proposals,
        )
    )
    payload = _redact_sensitive(
        {"schema_version": CLI_SCHEMA_VERSION, "source": reader.source, **grouped}
    )
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    if args.output_format == "jsonl":
        for cluster in payload["clusters"]:
            print(json.dumps(cluster, ensure_ascii=False, sort_keys=True))
        return 0
    rows = []
    for number, cluster in enumerate(payload["clusters"], start=1):
        for role, entries in (("proposal", cluster["members"]), ("kept", cluster["similar_kept"])):
            rows.extend(
                {
                    "group": number,
                    "role": role,
                    "sources": entry.get("sources"),
                    **_table_row(entry),
                }
                for entry in entries
            )
    if rows:
        print_table(("group", "role", "sources", "entry_id", "kind", "preview"), rows)
    else:
        print("No pending knowledge proposals.")
    if payload["truncated"]:
        print(
            "More proposals exist than were grouped; raise --max-proposals or narrow the filters."
        )
    return 0


async def _show_entry(args: argparse.Namespace, reader: KnowledgeReader) -> int:
    entry = await reader.load_entry(args.entry_id)
    if entry is None:
        raise ValueError(f"Knowledge entry not found: {args.entry_id}")
    entry = {**entry, **_source_fields(entry.get("metadata"))}
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
        "source_count",
    )
    rows = [
        {"field": name, "value": _display(entry.get(name))}
        for name in fields
        if entry.get(name) not in (None, {}, [])
    ]
    rows.extend(
        {
            "field": "corroborated_by",
            "value": " ".join(
                part
                for part in (
                    _display(source.get("session_id")),
                    _display(source.get("recorded_at")),
                    _display(source.get("reason")),
                )
                if part
            ),
        }
        for source in entry.get("corroborations") or ()
    )
    rows.extend(_review_rows(entry.get("metadata")))
    print_table(("field", "value"), rows)
    text = entry.get("text")
    if isinstance(text, str) and text:
        print("\nText")
        print(terminal_safe(text, keep_lines=True))
    chunks = entry.get("chunks")
    if isinstance(chunks, list) and len(chunks) > 1:
        print(f"\n{len(chunks)} chunks; --json prints each.")
    return 0


def _source_fields(metadata: object) -> dict[str, Any]:
    """The entry's source count and the sessions that corroborate it, from its metadata."""

    from cayu.knowledge.corroboration import knowledge_corroborations_in_metadata

    recorded = cast("dict[str, Any]", metadata) if isinstance(metadata, dict) else {}
    sources = knowledge_corroborations_in_metadata(recorded)
    return {
        "source_count": 1 + len(sources),
        "corroborations": [source.metadata_value() for source in sources],
    }


# Where an archived revision records the reviewed drop or archive that wrote it.
_REVIEW_METADATA = (("cayu:review_archive", "archived"), ("cayu:review_rejection", "dropped"))


def _review_rows(metadata: object) -> list[dict[str, Any]]:
    """Who archived or dropped the entry and why, as its archived revision records it."""

    if not isinstance(metadata, dict):
        return []
    recorded = cast("dict[str, Any]", metadata)
    for key, verb in _REVIEW_METADATA:
        decision = recorded.get(key)
        if not isinstance(decision, dict):
            continue
        return [
            {"field": f"{verb}_{name}", "value": _display(decision.get(field))}
            for name, field in (("by", "reviewer_identity"), ("reason", "reason"))
            if isinstance(decision.get(field), str)
        ]
    return []


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


def _revision(value: str) -> int:
    try:
        revision = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if revision < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return revision


def _bounded_int(maximum: int) -> Callable[[str], int]:
    """An argparse type for an integer from 1 to ``maximum``."""

    def parse(value: str) -> int:
        try:
            limit = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("must be an integer") from None
        if not 1 <= limit <= maximum:
            raise argparse.ArgumentTypeError(f"must be between 1 and {maximum}")
        return limit

    return parse


_page_limit = _bounded_int(MAX_KNOWLEDGE_PAGE)
_group_limit = _bounded_int(MAX_GROUPED_PROPOSALS)


async def _check_lookups(args: argparse.Namespace, target: ReadTarget) -> int:
    from cayu.cli._task_readers import _session_store
    from cayu.sessions.knowledge_usage import check_knowledge_lookup_totals

    if target.store is None:
        raise ValueError("Lookup counts are checked in a local store: pass --sqlite or --postgres.")
    store = _session_store(target.store)
    close = getattr(store, "close", None)
    try:
        check = await check_knowledge_lookup_totals(store)
    finally:
        if close is not None:
            await close()

    def total(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "lookups": value.lookups,
            "sessions": value.sessions,
            "last_lookup_at": value.last_lookup_at.isoformat(),
        }

    rows = [
        {
            "namespace": mismatch.namespace,
            "entry_id": mismatch.entry_id,
            "stored": total(mismatch.stored),
            "recomputed": total(mismatch.recomputed),
        }
        for mismatch in check.mismatches
    ]
    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "entries_stored": check.entries_stored,
                    "matches": not rows,
                    "mismatches": rows,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    elif rows:
        print_table(
            ("namespace", "entry", "stored", "recomputed"),
            [
                {
                    "namespace": row["namespace"],
                    "entry": row["entry_id"],
                    "stored": _lookup_cell(row["stored"]),
                    "recomputed": _lookup_cell(row["recomputed"]),
                }
                for row in rows
            ],
        )
    else:
        print(f"Lookup counts match the events for {check.entries_stored} entries.")
    return 1 if rows else 0


def _lookup_cell(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value['lookups']} in {value['sessions']} sessions, last {value['last_lookup_at']}"
