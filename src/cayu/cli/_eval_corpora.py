"""`cayu eval corpora list|import|download`: move portable corpora in and out of a server from
the terminal.

These are the Developer dashboard's Evals → Settings → **Import / export**, through the same
server API: `list` is its list of saved corpora (`GET /evals/corpora`), `import` uploads a
portable corpus file (`POST /evals/corpora`), and `download` writes a saved revision as its
file (`GET /evals/corpora/{revision}/download`). A corpus revision is its content, so
importing it again changes nothing, and nothing runs.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi
from cayu.cli._cloud_evals import MAX_LIST_LIMIT, CloudEvalError, _eval_revision, _list_limit, _text
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli._eval_curation import _page, _query


def add_corpus_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> list[tuple[argparse.ArgumentParser, Callable[[Mapping[str, Any]], str]]]:
    """Add `corpora` to `cayu eval`; return its parsers with their text renderers."""

    corpora = commands.add_parser(
        "corpora",
        help="List, import and download saved eval corpora.",
        description=(
            "List an eval target's saved corpora, import a portable corpus file, and download "
            "a saved revision, as Evals → Settings → Import / export does. Nothing runs."
        ),
    )
    actions = corpora.add_subparsers(dest="eval_corpora_command", required=True)
    listing = actions.add_parser(
        "list",
        help="List saved corpora.",
        description="List an eval target's saved corpus revisions, newest first.",
    )
    listing.add_argument(
        "--target-key", metavar="KEY", help="The eval target (default: the server's default)."
    )
    listing.add_argument(
        "--limit",
        type=_list_limit,
        metavar="N",
        help=f"Corpora per page, at most {MAX_LIST_LIMIT} (default: the server's).",
    )
    listing.add_argument("--cursor", help="The next page: the previous page's next_cursor.")
    upload = actions.add_parser(
        "import",
        help="Import a portable corpus file.",
        description=(
            "Import a portable corpus file into the server's eval store, as Import does, "
            "checked against the eval target it names. Importing it again changes nothing."
        ),
    )
    upload.add_argument("file", type=Path, metavar="FILE", help="A portable corpus JSON file.")
    download = actions.add_parser(
        "download",
        help="Download a saved corpus revision.",
        description=(
            "Write a saved corpus revision as its portable corpus file, as Download does: the "
            "file `cayu eval validate`, `cayu eval run --corpus` and `corpora import` read."
        ),
    )
    download.add_argument("revision", type=_eval_revision, metavar="REVISION")
    download.add_argument(
        "--json-output",
        type=Path,
        required=True,
        metavar="FILE",
        help="Write the corpus file to FILE.",
    )
    return [(listing, render_corpora), (upload, render_import), (download, render_download)]


def read_corpus_file(args: argparse.Namespace) -> Any:
    """The corpus `import` uploads, read and checked before any request."""

    from cayu.evals.corpus import load_eval_corpus

    if args.eval_corpora_command != "import":
        return None
    try:
        return load_eval_corpus(args.file)
    except OSError as exc:
        raise ValueError(
            f"Could not read {args.file}: {exc.strerror or type(exc).__name__}."
        ) from None
    except ValueError as exc:
        raise ValueError(f"{args.file} is not a valid eval corpus: {exc}") from None


def run_corpus_command(
    api: CayuApi, args: argparse.Namespace, corpus: Any, *, target: Sequence[str]
) -> dict[str, Any]:
    action = args.eval_corpora_command
    if action == "list":
        query = _query(target_key=args.target_key, limit=args.limit, cursor=args.cursor)
        return _page(api.get("/evals/corpora", query=query, resource="eval corpora"), "corpus list")
    if action == "import":
        return _import(api, corpus, target=target)
    return _download(api, args)


def _import(api: CayuApi, corpus: Any, *, target: Sequence[str]) -> dict[str, Any]:
    from cayu.evals.read_views import EvalSuiteViewIdentity

    entry = api.post("/evals/corpora", corpus.model_dump(mode="json"), resource="eval corpus")
    if not isinstance(entry, dict) or entry.get("revision") != corpus.revision:
        raise CloudEvalError(
            "api_response_invalid", "The server did not confirm the corpus it imported."
        )
    suites = []
    for suite in corpus.suites:
        suite_id = EvalSuiteViewIdentity(
            source="corpus",
            target_key=corpus.target_key,
            revision=corpus.revision,
            suite_id=suite.id,
        ).id
        suites.append(
            {
                "suite_id": suite.id,
                "id": suite_id,
                "show": shlex.join(["cayu", "eval", "suites", "show", suite_id, *target]),
            }
        )
    return {"entry": entry, "suites": suites}


def _download(api: CayuApi, args: argparse.Namespace) -> dict[str, Any]:
    """Write the revision as the server's Download renders it."""

    from cayu.evals.corpus import eval_corpus_from_json, eval_corpus_to_json

    document = api.get(
        f"/evals/corpora/{quote(args.revision, safe='')}/download", resource="eval corpus"
    )
    try:
        corpus = eval_corpus_from_json(json.dumps(document, ensure_ascii=False))
    except (TypeError, ValueError):
        corpus = None
    if corpus is None or corpus.revision != args.revision:
        raise CloudEvalError("api_response_invalid", "The server's eval corpus is invalid.")
    try:
        # The canonical text as it is, with no platform newline translation.
        args.json_output.write_text(eval_corpus_to_json(corpus), encoding="utf-8", newline="\n")
    except OSError as exc:
        raise CloudEvalError(
            "output_unwritable",
            f"Could not write {args.json_output}: {exc.strerror or type(exc).__name__}.",
        ) from None
    return {
        "revision": corpus.revision,
        "target_key": corpus.target_key,
        "suites": [suite.id for suite in corpus.suites],
        "file": str(args.json_output),
    }


def render_corpora(answer: Mapping[str, Any]) -> str:
    listed = answer["result"]
    if not listed["items"]:
        return "No saved corpora."
    rows = [
        (
            _cell(item.get("suite_count")),
            _cell(item.get("case_count")),
            _cell(item.get("created_at")),
            _text(item.get("revision")),
        )
        for item in listed["items"]
    ]
    lines = _table(("SUITES", "CASES", "SAVED", "REVISION"), rows)
    if listed["has_more"]:
        lines.append(f"More corpora follow: pass --cursor {_text(listed['next_cursor'])}.")
    return "\n".join(lines)


def render_import(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    lines = [f"Imported corpus {_text(result['entry']['revision'])}."]
    for suite in result["suites"]:
        lines.append(f"  {_text(suite['suite_id'])}: {_text(suite['show'])}")
    return "\n".join(lines)


def render_download(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    return f"Wrote corpus {_text(result['revision'])} to {_text(result['file'])}."
