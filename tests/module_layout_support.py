"""Collect the module-layout facts that behaviour-neutral file splits must keep.

Moving code between modules can change things no ordinary test notices: names
persisted in durable data, store capabilities authenticated by MRO position,
monkeypatches that stop reaching their reader, documented import paths, and
FastAPI route order. ``tests/test_module_layout.py`` compares these facts with
the snapshots committed under ``tests/fixtures/module_layout/``.

Most snapshots are one-way: every committed entry must still hold, and new
entries never fail a check. Only the route order and the browser worker files
are compared exactly, because adding an entry there changes behaviour.

Regenerate a snapshot only when a change to it is intended and reviewed::

    uv run python -m tests.module_layout_support --write server_routes
    uv run python -m tests.module_layout_support --write --all

The monkeypatch allowlist (``monkeypatch_allowlist.json``) is edited by hand.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import dis
import enum
import importlib
import importlib.machinery
import inspect
import json
import re
import sys
import tempfile
import types
import typing
from collections.abc import Callable, Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "src"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "fixtures" / "module_layout"
MONKEYPATCH_ALLOWLIST = SNAPSHOT_DIR / "monkeypatch_allowlist.json"
REGENERATE_COMMAND = "uv run python -m tests.module_layout_support --write"

# Modules that keep established import and pickle paths by re-exporting moved
# names (``from new import X as X``). A split that leaves a shim behind adds the
# old module here so its re-exports stay bound to the canonical objects.
COMPATIBILITY_SHIM_MODULES = (
    "cayu.runtime._invocation_lifecycle",
    "cayu.runtime._session_continuation",
    "cayu.runtime.execution_profiles",
)

# Logger names that operators filter on, besides every ``getLogger(__name__)``.
_LITERAL_LOGGER = re.compile(r"""getLogger\(\s*["'](cayu(?:\.\w+)+)["']\s*\)""")
_MODULE_LOGGER = re.compile(r"getLogger\(\s*__name__\s*\)")
_IDENTITY_STRING = re.compile(r"""["'](cayu(?:\.[A-Za-z_]\w*)+:[A-Za-z_][\w.]*)["']""")
_FROM_IMPORT = re.compile(
    r"^([ \t]*)from[ \t]+(cayu(?:\.\w+)*)[ \t]+import[ \t]+(\([^)]*\)|[^\n]+)", re.MULTILINE
)
_PLAIN_IMPORT = re.compile(
    r"^[ \t]*import[ \t]+(cayu(?:\.\w+)*)(?:[ \t]+as[ \t]+(\w+))?", re.MULTILINE
)
_STRING_PATCH = re.compile(
    r"""\b(?:setattr|delattr|patch)\(\s*[rbu]?["'](cayu(?:\.\w+)+)["']\s*[,)]"""
)
_OBJECT_PATCH = re.compile(
    r"""\b(?:setattr|delattr|patch\.object)\(\s*([A-Za-z_][\w.]*)\s*,\s*["'](\w+)["']\s*[,)]"""
)


# --------------------------------------------------------------------------- helpers


def identity(value: object) -> str | None:
    """Return ``module:qualname`` for a class or function, else ``None``."""

    if not (inspect.isclass(value) or inspect.isroutine(value)):
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if type(module) is not str or type(qualname) is not str:
        return None
    return f"{module}:{qualname}"


def resolve(path: str) -> object:
    """Resolve ``module:qualname`` to the object it names."""

    module_name, _, qualname = path.partition(":")
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


@cache
def source_modules() -> dict[str, Path]:
    """Map every ``cayu`` module name to its source file, without importing it."""

    modules: dict[str, Path] = {}
    for path in sorted((SOURCE_ROOT / "cayu").rglob("*.py")):
        parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    return modules


@cache
def code_files() -> tuple[Path, ...]:
    """Source modules plus the repository scripts that import them."""

    return (*source_modules().values(), *sorted((REPO_ROOT / "scripts").rglob("*.py")))


@cache
def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imported_names(block: str) -> Iterator[tuple[str, str]]:
    """Yield ``(name, bound_as)`` for each name of one ``from … import`` clause."""

    for item in re.split(r"[,\n]", re.sub(r"#[^\n]*", "", block).strip().strip("()")):
        words = item.split()
        if not words or not words[0].isidentifier():
            continue
        bound = words[2] if len(words) == 3 and words[1] == "as" else words[0]
        yield words[0], bound


def _module_bindings(text: str) -> list[tuple[int, str, str]]:
    """``(offset, local name, cayu module)`` for each module this file's imports bind."""

    modules = source_modules()
    bindings: list[tuple[int, str, str]] = []
    for match in _PLAIN_IMPORT.finditer(text):
        module, alias = match.groups()
        bindings.append((match.start(), alias or "cayu", module if alias else "cayu"))
    for match in _FROM_IMPORT.finditer(text):
        for name, bound in _imported_names(match.group(3)):
            if f"{match.group(2)}.{name}" in modules:
                bindings.append((match.start(), bound, f"{match.group(2)}.{name}"))
    return sorted(bindings)


def _dotted_module(
    expression: str, bindings: list[tuple[int, str, str]], offset: int
) -> str | None:
    """The module ``expression`` names at ``offset``, using the latest earlier import."""

    head, *rest = expression.split(".")
    base = None
    for position, name, module in bindings:
        if position > offset:
            break
        if name == head:
            base = module
    if base is None:
        return None
    full = ".".join([base, *rest])
    return full if full in source_modules() else None


def _changed_lines(expected: object, actual: object, prefix: str = "") -> list[str]:
    """Every difference between ``expected`` and ``actual``, in order where order matters."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        lines: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            where = f"{prefix}{key}"
            if key not in actual:
                lines.append(f"  - {where}: {json.dumps(expected[key])}")
            elif key not in expected:
                lines.append(f"  + {where}: {json.dumps(actual[key])}")
            else:
                lines.extend(_changed_lines(expected[key], actual[key], f"{where}."))
        return lines
    if isinstance(expected, list) and isinstance(actual, list):
        if expected == actual:
            return []
        expected_items = [json.dumps(item) for item in expected]
        actual_items = [json.dumps(item) for item in actual]
        if sorted(expected_items) == sorted(actual_items):
            return [f"  ~ {prefix.rstrip('.')}: same entries in a different order"]
        lines = [
            f"  - {prefix}{item}"
            for item in dict.fromkeys(expected_items)
            if expected_items.count(item) > actual_items.count(item)
        ]
        lines += [
            f"  + {prefix}{item}"
            for item in dict.fromkeys(actual_items)
            if actual_items.count(item) > expected_items.count(item)
        ]
        return lines
    if expected != actual:
        return [f"  ~ {prefix.rstrip('.')}: {json.dumps(expected)} -> {json.dumps(actual)}"]
    return []


def _lost_lines(expected: object, actual: object, prefix: str = "") -> list[str]:
    """Snapshot entries that ``actual`` no longer has, or has with another value.

    Entries that only ``actual`` has are additions and are ignored.
    """

    if isinstance(expected, dict) and isinstance(actual, dict):
        lines: list[str] = []
        for key, value in expected.items():
            if key not in actual:
                lines.append(f"  - {prefix}{key}: {json.dumps(value)}")
            else:
                lines.extend(_lost_lines(value, actual[key], f"{prefix}{key}."))
        return lines
    if isinstance(expected, list) and isinstance(actual, list):
        present = {json.dumps(item) for item in actual}
        return [
            f"  - {prefix}{json.dumps(item)}"
            for item in expected
            if json.dumps(item) not in present
        ]
    if expected != actual:
        return [f"  ~ {prefix.rstrip('.')}: {json.dumps(expected)} -> {json.dumps(actual)}"]
    return []


def truncated(lines: list[str], limit: int = 60) -> list[str]:
    shown = lines[:limit]
    if len(lines) > limit:
        shown.append(f"  ... and {len(lines) - limit} more")
    return shown


def snapshot_mismatch(
    label: str,
    expected: object,
    actual: object,
    why: str,
    *,
    snapshot: str,
    exact: bool = False,
) -> str | None:
    """Explain how ``actual`` breaks the committed snapshot, or return ``None``.

    By default the snapshot is one-way: each committed entry must still be
    present with the same value, and additions pass. ``exact`` also fails on
    additions and, for lists, on a different order.
    """

    lines = _changed_lines(expected, actual) if exact else _lost_lines(expected, actual)
    if not lines:
        return None
    if expected and not actual:
        return (
            f"{label}: the collector in {Path(__file__).name} found nothing. Its discovery no "
            "longer matches the source; update the collector to the new layout instead of "
            "regenerating the snapshot."
        )
    legend = (
        "(- removed, + added, ~ changed)"
        if exact
        else "(- no longer present, ~ changed; additions are allowed)"
    )
    return (
        f"{label} changed {legend}. {why}\n"
        + "\n".join(truncated(lines))
        + f"\nIf this change is intended, review it and run `{REGENERATE_COMMAND} {snapshot}`."
    )


# ------------------------------------------------------------ durable identities

DURABLE_IDENTITY_REASONS = {
    "execution_profile_components": (
        "Execution-profile material records `module:qualname` for each built-in tool, "
        "runner, policy and provider (runtime/_execution_profile_admission.py). A new "
        "identity makes stored sessions fail admission or need adoption."
    ),
    "execution_profile_material_owners": (
        "Classes that own Cayu-versioned execution-profile material are named by "
        "`module:qualname` in persisted profiles."
    ),
    "loop_policies": (
        "Loop-policy implementations are persisted as `module:qualname` in execution-profile "
        "adoption digests (sessions/_execution_profile_rules.py, approvals/user_input.py)."
    ),
    "transcript_digest_progress_key": (
        "TranscriptDigestCompactor hashes its class and `compact` `module:qualname` into the "
        "checkpoint compaction progress key (context/base.py)."
    ),
    "behavior_identity_strings": (
        "Source code and scripts name these objects by string; persisted identities and "
        "process entry points stay valid only while each object stays at that path."
    ),
    "support_bundle_stores": (
        "support_bundles.py recognises built-in stores by exact `(module, class)` pairs; a "
        "moved store silently drops out of schema-readiness reporting."
    ),
    "eval_digest_types": (
        "Eval trajectory digests frame every enum and model by `module.qualname` "
        "(evals/models.py); a moved type changes stored suite and corpus digests."
    ),
    "eval_process_assertions": (
        "Process evals accept revision-less assertions only from `cayu.evals.assertions` and "
        "send them by `module:qualname` (cli/_eval_processes.py)."
    ),
    "isolated_tool_worker": (
        "Isolated tool invocations start the worker by module path and supervisor script "
        "(runtime/_isolated_tool_process.py); queued and recovered invocations depend on it."
    ),
    "browser_worker_files": (
        "Browser images are verified by the digests of these packaged guest sources "
        "(runners/workloads.py); changing them needs a worker_version bump and an image rebuild."
    ),
    "compatibility_re_exports": (
        "These modules keep established import and pickle paths; each re-export must stay "
        "bound to the same canonical object."
    ),
    "logger_names": (
        "Operators filter logs by logger name; code that moves out of a module renames its "
        "logger unless the new module pins the old name."
    ),
}


# Categories where an added entry is itself a behaviour change, so the snapshot is
# compared exactly. Every other category is one-way: committed entries must stay,
# and new ones (a new tool, logger or identity string) pass.
EXACT_DURABLE_IDENTITIES = {
    "browser_worker_files": (
        "A new guest file is packaged into the browser image too, so it changes the image "
        "digests and needs a worker_version bump just like a renamed one."
    ),
}


def _execution_profile_components() -> list[str]:
    from cayu.runtime import _execution_profile_admission as admission

    components: set[str] = set()
    for name in dir(admission):
        if re.fullmatch(r"_cayu_\w+_material_extractors", name):
            components.update(
                identity(owner) or repr(owner) for owner in getattr(admission, name)()
            )
    # Policies whose material is built by hand in _cayu_policy_material.
    for path in (
        "cayu.approvals.authorization_grant_policy:ToolAuthorizationGrantPolicy",
        "cayu.tools.policy:GuardedToolPolicy",
        "cayu.tools.policy:RequiredArguments",
    ):
        components.add(identity(resolve(path)) or path)
    return sorted(components)


def _execution_profile_material_owners() -> list[str]:
    owners: set[str] = set()
    for module_name, path in source_modules().items():
        if "def _execution_profile_material" not in source_text(path):
            continue
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if (
                inspect.isclass(value)
                and value.__module__ == module_name
                and "_execution_profile_material" in vars(value)
            ):
                owners.add(identity(value) or "")
    return sorted(owners)


def _source_subclasses(root: type) -> list[str]:
    """Every ``cayu`` class deriving from ``root``, independent of what is already imported.

    Modules that name a known class as a base are imported until no new subclass
    appears, so the result doesn't depend on which tests ran first.
    """

    names = {root.__name__}
    modules: set[str] = set()
    found: set[type] = set()
    while True:
        pattern = re.compile(
            rf"^[ \t]*class[ \t]+\w+\([^)]*\b(?:{'|'.join(sorted(names))})\b", re.MULTILINE
        )
        new = {
            module_name
            for module_name, path in source_modules().items()
            if module_name not in modules and pattern.search(source_text(path))
        }
        if not new:
            break
        for module_name in sorted(new):
            importlib.import_module(module_name)
        modules |= new
        pending = [root]
        while pending:
            for subclass in pending.pop().__subclasses__():
                pending.append(subclass)
                if subclass.__module__ in modules and "<locals>" not in subclass.__qualname__:
                    found.add(subclass)
        names |= {subclass.__name__ for subclass in found}
    return sorted(identity(subclass) or "" for subclass in found)


def _loop_policies() -> list[str]:
    from cayu.runtime.loop_policies import LoopPolicy

    return _source_subclasses(LoopPolicy)


def _transcript_digest_progress_key() -> dict[str, str]:
    from cayu.context.base import TranscriptDigestCompactor

    keys = {identity(TranscriptDigestCompactor) or "": TranscriptDigestCompactor()._progress_key()}
    for subclass in _source_subclasses(TranscriptDigestCompactor):
        compactor = typing.cast("Any", resolve(subclass))
        if not inspect.isabstract(compactor):
            keys[subclass] = compactor()._progress_key()
    return keys


def _top_level_binding(module_name: str, name: str) -> str:
    """How ``module_name`` binds ``name``, read from source so entry points are not run."""

    path = source_modules().get(module_name)
    if path is None:
        return "<missing module>"
    text = source_text(path)
    escaped = re.escape(name)
    if re.search(rf"^(?:class|def|async def)[ \t]+{escaped}\b", text, re.MULTILINE):
        return "defined"
    if re.search(rf"^{escaped}[ \t]*[:=]", text, re.MULTILINE):
        return "assigned"
    return "<not defined here>"


def _behavior_identity_strings() -> dict[str, str]:
    strings: dict[str, str] = {}
    for path in code_files():
        for match in _IDENTITY_STRING.finditer(source_text(path)):
            module_name, _, qualname = match.group(1).partition(":")
            strings[match.group(1)] = _top_level_binding(module_name, qualname.split(".")[0])
    return dict(sorted(strings.items()))


def _support_bundle_stores() -> dict[str, str | None]:
    from cayu import support_bundles

    stores: dict[str, str | None] = {}
    for name in dir(support_bundles):
        if name.endswith("_STORE_IDENTITIES"):
            for module_name, class_name in getattr(support_bundles, name):
                path = f"{module_name}:{class_name}"
                try:
                    stores[path] = identity(resolve(path))
                except (ImportError, AttributeError):
                    stores[path] = "<unresolved>"
    return dict(sorted(stores.items()))


def _annotation_types(annotation: object) -> Iterator[type]:
    if inspect.isclass(annotation):
        yield annotation
    for argument in typing.get_args(annotation):
        yield from _annotation_types(argument)


def _eval_digest_types() -> list[str]:
    from pydantic import BaseModel

    from cayu.evals.models import Trajectory

    seen: set[type] = set()
    pending: list[type] = [Trajectory]
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        if issubclass(model, BaseModel):
            for field in model.model_fields.values():
                for found in _annotation_types(field.annotation):
                    if issubclass(found, BaseModel | enum.Enum):
                        pending.append(found)
    return sorted(f"{model.__module__}.{model.__qualname__}" for model in seen)


def _eval_process_assertions() -> list[str]:
    from cayu.evals import assertions

    return sorted(
        identity(value) or ""
        for value in vars(assertions).values()
        if inspect.isclass(value)
        and issubclass(value, assertions.EvalAssertion)
        and value.__module__ == assertions.__name__
    )


def _isolated_tool_worker() -> dict[str, str]:
    from cayu.runtime import _isolated_tool_process as process

    worker = importlib.import_module(process._WORKER_MODULE)
    supervisor = Path(process._SUPERVISOR_SCRIPT)
    return {
        "worker_module": worker.__name__,
        "supervisor_script": supervisor.relative_to(SOURCE_ROOT).as_posix()
        if supervisor.is_file()
        else "<missing>",
    }


def _browser_worker_files() -> dict[str, str]:
    from cayu.runners.workloads import BROWSER_WORKER_FILES

    return {guest: f"cayu/tools/{source}" for guest, source in BROWSER_WORKER_FILES}


def _compatibility_re_exports() -> dict[str, dict[str, str]]:
    exports: dict[str, dict[str, str]] = {}
    for module_name in COMPATIBILITY_SHIM_MODULES:
        module = importlib.import_module(module_name)
        names: dict[str, str] = {}
        for match in _FROM_IMPORT.finditer(source_text(source_modules()[module_name])):
            if match.group(1):
                continue
            # Only the explicit ``X as X`` form declares a deliberate re-export.
            for name in re.findall(r"\b(\w+)\s+as\s+\1\b", match.group(3)):
                names[name] = identity(getattr(module, name)) or f"{match.group(2)}.{name}"
        exports[module_name] = dict(sorted(names.items()))
    return exports


def _logger_names() -> list[str]:
    names: set[str] = set()
    for module_name, path in source_modules().items():
        text = source_text(path)
        if "getLogger" not in text:
            continue
        if _MODULE_LOGGER.search(text):
            names.add(module_name)
        names.update(_LITERAL_LOGGER.findall(text))
    return sorted(names)


DURABLE_IDENTITY_COLLECTORS: dict[str, Callable[[], object]] = {
    "execution_profile_components": _execution_profile_components,
    "execution_profile_material_owners": _execution_profile_material_owners,
    "loop_policies": _loop_policies,
    "transcript_digest_progress_key": _transcript_digest_progress_key,
    "behavior_identity_strings": _behavior_identity_strings,
    "support_bundle_stores": _support_bundle_stores,
    "eval_digest_types": _eval_digest_types,
    "eval_process_assertions": _eval_process_assertions,
    "isolated_tool_worker": _isolated_tool_worker,
    "browser_worker_files": _browser_worker_files,
    "compatibility_re_exports": _compatibility_re_exports,
    "logger_names": _logger_names,
}


def collect_durable_identities() -> dict[str, object]:
    return {name: collect() for name, collect in DURABLE_IDENTITY_COLLECTORS.items()}


# ------------------------------------------------------------ store capabilities

_PROBE_NAME = re.compile(r"_?supports_\w+")


def _bare(store_type: type) -> Any:
    """An uninitialised instance: the probes inspect the class MRO, not live state."""

    return store_type.__new__(store_type)


def _zero_argument_probes(store: object, base: type) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for name in sorted(dir(base)):
        if not _PROBE_NAME.fullmatch(name):
            continue
        if not isinstance(inspect.getattr_static(base, name), types.FunctionType):
            continue
        probe = getattr(store, name)
        if inspect.signature(probe).parameters:
            continue
        results[name] = probe()
    return results


@cache
def session_message_lifecycle_operations() -> tuple[str, ...]:
    """Every operation name the coordinator passes to ``_require_lifecycle_capability``."""

    from cayu.runtime._session_message_coordinator import SessionMessageCoordinator

    tree = ast.parse(source_text(source_modules()[SessionMessageCoordinator.__module__]))
    operations: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_require_lifecycle_capability"
        ):
            operations.update(
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            )
    return tuple(sorted(operations))


def _session_store_capabilities(store_type: type) -> dict[str, object]:
    from cayu.runtime._session_message_coordinator import SessionMessageCoordinator
    from cayu.sessions import store as session_store_module
    from cayu.sessions.store import SessionStore

    store = _bare(store_type)
    probes: dict[str, object] = dict(_zero_argument_probes(store, SessionStore))
    for name in sorted(vars(session_store_module)):
        probe = getattr(session_store_module, name)
        if (
            re.fullmatch(r"_session_\w+_methods_owned", name)
            and len(inspect.signature(probe).parameters) == 1
        ):
            probes[name] = probe(store)

    coordinator = typing.cast("Any", types.SimpleNamespace(_store=store))
    attested: dict[str, bool] = {}
    for operation in session_message_lifecycle_operations():
        try:
            SessionMessageCoordinator._require_lifecycle_capability(coordinator, operation)
        except NotImplementedError:
            attested[operation] = False
        else:
            attested[operation] = True
    probes["session_message_lifecycle_operations"] = attested

    # SessionEngine requires the class that defines queued delivery to declare the
    # queued-profile handoff version itself.
    owner = next(
        (
            candidate
            for candidate in store_type.__mro__
            if "deliver_queued_session_messages" in vars(candidate)
        ),
        None,
    )
    probes["queued_interaction_profile_handoff"] = (
        getattr(store_type, "queued_interaction_profile_handoff_version", None) == 1
        and owner is not None
        and vars(owner).get("queued_interaction_profile_handoff_version") == 1
    )
    return dict(sorted(probes.items()))


def _task_store_capabilities(store_type: type) -> dict[str, object]:
    from cayu.runtime import _task_store_operation_boundary as boundary
    from cayu.tasks.store import TaskStore

    store = _bare(store_type)
    results: dict[str, object] = {}
    for name in sorted(dir(boundary)):
        if re.fullmatch(r"task_store_\w+_capability_is_complete", name):
            results[name] = getattr(boundary, name)(store)
    task_methods = sorted(
        name for name, value in vars(TaskStore).items() if isinstance(value, types.FunctionType)
    )
    results["cancellation_quiescent_methods"] = {
        name: boundary.task_store_mutation_is_cancellation_quiescent(store, name)
        for name in task_methods
    }
    results["stable_concrete_methods"] = {
        name: boundary._task_store_method_has_stable_concrete_implementation(store, name)
        for name in task_methods
    }
    return results


def _budget_ledger_capabilities(ledger_type: type) -> dict[str, object]:
    ledger = _bare(ledger_type)
    return {"_supports_producer_budget_readback": ledger._supports_producer_budget_readback()}


SESSION_STORES = (
    "cayu.sessions.memory:InMemorySessionStore",
    "cayu.storage.sqlite:SQLiteSessionStore",
    "cayu.storage.postgres:PostgresSessionStore",
)
TASK_STORES = (
    "cayu.tasks.memory:InMemoryTaskStore",
    "cayu.storage.tasks_sqlite:SQLiteTaskStore",
    "cayu.storage.tasks_postgres:PostgresTaskStore",
)
BUDGET_LEDGERS = (
    "cayu.budgets.base:InMemoryBudgetLedger",
    "cayu.storage.budget_ledger:SQLiteBudgetLedger",
    "cayu.storage.budget_postgres:PostgresBudgetLedger",
)


@cache
def collect_store_capabilities() -> dict[str, dict[str, object]]:
    """The result of every capability probe, per shipped store class.

    Each value is what the Runtime's own probe returns for an uninitialised
    instance; methods are recorded as ``name: bool`` so a new store method or
    probe is an addition, not a flip.
    """

    capabilities: dict[str, dict[str, object]] = {}
    for paths, collect in (
        (SESSION_STORES, _session_store_capabilities),
        (TASK_STORES, _task_store_capabilities),
        (BUDGET_LEDGERS, _budget_ledger_capabilities),
    ):
        for path in paths:
            capabilities[path] = collect(typing.cast("type", resolve(path)))
    return capabilities


# --------------------------------------------------------- public import surface


def public_packages() -> list[str]:
    return json.loads((REPO_ROOT / "docs" / "public-api-packages.json").read_text())


def documented_import_files() -> list[Path]:
    paths = [REPO_ROOT / "README.md"]
    for root in ("docs", "examples"):
        paths.extend(sorted((REPO_ROOT / root).rglob("*")))
    paths.extend(sorted((SOURCE_ROOT / "cayu" / "guides").glob("*.md")))
    return [path for path in paths if path.is_file() and path.suffix in {".md", ".py"}]


def documented_imports() -> dict[str, set[str]]:
    """Every ``from cayu… import name`` written in README, docs, guides and examples."""

    imports: dict[str, set[str]] = {}
    for path in documented_import_files():
        text = path.read_text(errors="ignore")
        if path == REPO_ROOT / "docs" / "release-notes.md":
            # Published notes describe their release's API; audit the unreleased part.
            text = re.split(r"(?m)^## v\d+\.\d+\.\d+\b", text, maxsplit=1)[0]
        for match in _FROM_IMPORT.finditer(text):
            for name, _ in _imported_names(match.group(3)):
                imports.setdefault(match.group(2), set()).add(name)
    return imports


def pinned_identity(value: object) -> str | None:
    """``module:qualname`` of a Cayu class or function; ``None`` for anything else.

    Names that resolve to third-party or standard-library objects (pydantic,
    typing, enum, hashlib, collections.abc) or to plain values are checked for
    presence only, since their defining module isn't Cayu's to keep.
    """

    found = identity(value)
    return found if found is not None and found.startswith("cayu.") else None


def _surface_entry(module: types.ModuleType, names: Iterable[str]) -> dict[str, object]:
    """Group names by the Cayu module that defines them under the same name.

    ``defined_in`` maps a defining module to names whose qualname equals the
    exported name; ``identities`` holds the few names bound to an object of
    another qualname; ``present`` holds names checked for presence only.
    """

    defined_in: dict[str, list[str]] = {}
    identities: dict[str, str] = {}
    present: list[str] = []
    for name in sorted(names):
        pinned = pinned_identity(getattr(module, name))
        if pinned is None:
            present.append(name)
        elif pinned.partition(":")[2] == name:
            defined_in.setdefault(pinned.partition(":")[0], []).append(name)
        else:
            identities[name] = pinned
    return {
        "defined_in": dict(sorted(defined_in.items())),
        "identities": identities,
        "present": present,
    }


def surface_identities(entry: dict[str, Any]) -> dict[str, str]:
    """The ``module:qualname`` each pinned name of a ``public_surface`` entry must keep."""

    pinned = {
        name: f"{module_name}:{name}"
        for module_name, names in entry["defined_in"].items()
        for name in names
    }
    return {**pinned, **entry["identities"]}


def collect_public_surface() -> dict[str, dict[str, dict[str, object]]]:
    """Package exports and documented imports, with the Cayu identity each resolves to.

    A documented import that a package export already covers is recorded once,
    under ``package_exports``. A documented module or name that doesn't resolve
    fails the collection instead of being left out of the snapshot.
    """

    exports: dict[str, dict[str, object]] = {}
    exported: dict[str, set[str]] = {}
    for package_name in public_packages():
        manifest = importlib.import_module(f"{package_name}._exports")
        package = importlib.import_module(package_name)
        exported[package_name] = set(manifest.EXPORTS)
        exports[package_name] = _surface_entry(package, manifest.EXPORTS)
    documented: dict[str, dict[str, object]] = {}
    unresolved: list[str] = []
    for module_name, names in sorted(documented_imports().items()):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            unresolved.append(f"import {module_name}: {exc}")
            continue
        unresolved.extend(
            f"{module_name}.{name}" for name in sorted(names) if not hasattr(module, name)
        )
        remaining = {name for name in names if hasattr(module, name)}
        remaining -= exported.get(module_name, set())
        if remaining:
            documented[module_name] = _surface_entry(module, remaining)
    if unresolved:
        raise RuntimeError(
            "README, docs, guides or examples import names that don't resolve; fix the "
            "documentation or the code before regenerating:\n  " + "\n  ".join(unresolved)
        )
    return {"package_exports": exports, "documented_imports": documented}


# ----------------------------------------------------------------- server routes


def collect_server_routes() -> list[str]:
    """Route registration order of the full server, as the dashboard schema builds it.

    The server is configured like ``dashboard/scripts/generate-api-types.py``
    (``_openapi_json``), which builds it inside a function that only returns the
    schema text, so it can't be reused here. The operation IDs are also enforced
    by the "Dashboard API drift" CI step (``npm run check:api:repo``); this
    snapshot adds the registration order, the route classes and the routes the
    schema leaves out. It is compared exactly and in order, because FastAPI
    matches routes in registration order and an added route can shadow a later
    one.
    """

    from cayu import AgentSpec, CayuApp, RunRequest
    from cayu.evals import CorpusTarget
    from cayu.server import EvalsConfig, EvaluationPromotionConfig, ServerConfig, create_server
    from cayu.storage import SQLiteEvalStore

    app = CayuApp()
    app.register_agent(AgentSpec(name="assistant", model="schema-only"))
    with tempfile.TemporaryDirectory(prefix="cayu-module-layout-") as directory:
        eval_store = SQLiteEvalStore(Path(directory) / "evals.db")
        try:
            server = create_server(
                app,
                config=ServerConfig.protected(
                    lambda _request: {"subject": "module-layout"},
                    evaluation_promotion=EvaluationPromotionConfig(
                        target_key="schema",
                        source_agent_name="assistant",
                        application_release_id="schema",
                    ),
                    evals=EvalsConfig(
                        target=CorpusTarget(
                            key="schema",
                            app=app,
                            request_base=RunRequest(agent_name="assistant", messages=[]),
                            application_release_id="schema",
                        ),
                        store=eval_store,
                    ),
                ),
            )
            paths = server.openapi()["paths"]
            routes = []
            for route in server.routes:
                path = getattr(route, "path", "")
                methods = sorted(getattr(route, "methods", None) or ())
                operations = paths.get(path, {})
                operation_ids = [
                    f"{method}={operations[method.lower()]['operationId']}"
                    for method in methods
                    if method.lower() in operations
                ]
                routes.append(
                    " ".join(
                        [
                            ",".join(methods) or "*",
                            path,
                            f"name={getattr(route, 'name', None)}",
                            f"class={type(route).__name__}",
                            *operation_ids,
                        ]
                    )
                )
        finally:
            asyncio.run(eval_store.close())
    return routes


# ------------------------------------------------------------- monkeypatch drift


def _test_files() -> Iterable[Path]:
    return sorted(Path(__file__).resolve().parent.rglob("*.py"))


def monkeypatch_targets() -> dict[tuple[str, str], set[str]]:
    """Map each patched ``(cayu module, attribute)`` to the test files that patch it."""

    modules = source_modules()
    targets: dict[tuple[str, str], set[str]] = {}
    for path in _test_files():
        text = path.read_text(encoding="utf-8")
        if "setattr(" not in text and "patch" not in text:
            continue
        where = path.relative_to(REPO_ROOT).as_posix()
        for match in _STRING_PATCH.finditer(text):
            module_name, _, attribute = match.group(1).rpartition(".")
            if module_name in modules:
                targets.setdefault((module_name, attribute), set()).add(where)
        if not _OBJECT_PATCH.search(text):
            continue
        bindings = _module_bindings(text)
        for match in _OBJECT_PATCH.finditer(text):
            module_name = _dotted_module(match.group(1), bindings, match.start())
            if module_name is not None:
                targets.setdefault((module_name, match.group(2)), set()).add(where)
    return targets


def _module_code(module_name: str) -> types.CodeType:
    """The module's code object, from the bytecode cache when it is current."""

    path = str(source_modules()[module_name])
    code = importlib.machinery.SourceFileLoader(module_name, path).get_code(module_name)
    assert code is not None, path
    return code


_LOAD_GLOBAL = dis.opmap["LOAD_GLOBAL"]
_EXTENDED_ARG = dis.opmap["EXTENDED_ARG"]
_COMPREHENSIONS = frozenset({"<listcomp>", "<setcomp>", "<dictcomp>", "<genexpr>"})


def _call_time_global_reads(code: types.CodeType, wanted: set[str]) -> dict[str, set[str]]:
    """Map each name in ``wanted`` to the qualnames of the functions that read it when called.

    Code that runs at import is skipped: the module body, class bodies and the
    comprehensions they contain (separate code objects before Python 3.12), and
    the lazily evaluated ``__annotate__`` functions of Python 3.14. A
    comprehension inside a function counts as a read by that function, so the
    result doesn't depend on whether the interpreter inlines comprehensions.
    """

    readers: dict[str, set[str]] = {}
    # (code object, qualname of the function that runs it, or None at import time)
    pending: list[tuple[types.CodeType, str | None]] = [(code, None)]
    while pending:
        current, reader = pending.pop()
        for const in current.co_consts:
            if not isinstance(const, types.CodeType) or const.co_name == "__annotate__":
                continue
            if const.co_name in _COMPREHENSIONS:
                pending.append((const, reader))
            elif const.co_flags & inspect.CO_OPTIMIZED:
                pending.append((const, const.co_qualname))
            else:
                pending.append((const, None))
        if reader is None or not wanted.intersection(current.co_names):
            continue
        raw = current.co_code
        extended = 0
        for offset in range(0, len(raw), 2):
            opcode, argument = raw[offset], raw[offset + 1] | extended
            if opcode == _EXTENDED_ARG:
                extended = argument << 8
                continue
            extended = 0
            # Since Python 3.11 the low bit of LOAD_GLOBAL's argument flags a NULL push.
            if opcode == _LOAD_GLOBAL and current.co_names[argument >> 1] in wanted:
                readers.setdefault(current.co_names[argument >> 1], set()).add(reader)
    return readers


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements of one scope that run with it, without nested functions or classes."""

    for statement in body:
        yield statement
        if isinstance(statement, ast.If):
            if not _is_type_checking(statement.test):
                yield from _runtime_statements(statement.body)
            yield from _runtime_statements(statement.orelse)
        elif isinstance(statement, ast.Try | ast.TryStar):
            for block in (statement.body, statement.orelse, statement.finalbody):
                yield from _runtime_statements(block)
            for handler in statement.handlers:
                yield from _runtime_statements(handler.body)
        elif isinstance(statement, ast.With | ast.AsyncWith | ast.For | ast.AsyncFor | ast.While):
            yield from _runtime_statements(statement.body)
            yield from _runtime_statements(getattr(statement, "orelse", []))


def _scope_bindings(body: list[ast.stmt], package: str) -> dict[str, str]:
    """Local names that one scope's imports bind to a ``cayu`` module."""

    modules = source_modules()
    bindings: dict[str, str] = {}
    for statement in _runtime_statements(body):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name.split(".")[0] != "cayu":
                    continue
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    bindings[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(statement, ast.ImportFrom):
            source = _absolute_module(statement, package)
            for alias in statement.names:
                if source is not None and f"{source}.{alias.name}" in modules:
                    bindings[alias.asname or alias.name] = f"{source}.{alias.name}"
    return bindings


def _absolute_module(statement: ast.ImportFrom, package: str) -> str | None:
    if not statement.level:
        return statement.module
    if not package:
        return None
    base = package.rsplit(".", statement.level - 1)[0] if statement.level > 1 else package
    return f"{base}.{statement.module}" if statement.module else base


class _ExternalReadScanner(ast.NodeVisitor):
    """Find call-time reads of patched names from outside the patched module.

    A function-local ``from module import name`` and a ``module.name`` access
    inside a function see the patch. Module-level imports, code under
    ``if TYPE_CHECKING:``, module-level code and annotations don't count.
    """

    def __init__(self, owner: str, package: str, targets: dict[str, set[str]]) -> None:
        self.owner = owner
        self.package = package
        self.targets = targets
        self.reads: dict[tuple[str, str], set[str]] = {}
        self._qualname: list[str] = []
        self._in_function = False
        self._reader: str | None = None
        self._bindings: list[dict[str, str]] = []

    def scan(self, tree: ast.Module) -> dict[tuple[str, str], set[str]]:
        self._bindings = [_scope_bindings(tree.body, self.package)]
        self.visit(tree)
        return self.reads

    def _record(self, module_name: str, attribute: str) -> None:
        if self._reader is not None and attribute in self.targets.get(module_name, ()):
            self.reads.setdefault((module_name, attribute), set()).add(
                f"{self.owner}:{self._reader}"
            )

    def _resolve(self, node: ast.expr) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        for bindings in reversed(self._bindings):
            if node.id in bindings:
                return ".".join([bindings[node.id], *reversed(parts)])
        return None

    def _child_qualname(self, name: str) -> list[str]:
        return [*self._qualname, *(["<locals>"] if self._in_function else []), name]

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, name: str
    ) -> None:
        if not isinstance(node, ast.Lambda):
            for decorator in node.decorator_list:
                self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        saved = (self._qualname, self._in_function, self._reader)
        self._qualname = self._child_qualname(name)
        self._in_function = True
        self._reader = ".".join(self._qualname)
        body: list[ast.stmt] = [ast.Expr(node.body)] if isinstance(node, ast.Lambda) else node.body
        self._bindings.append(_scope_bindings(body, self.package))
        for statement in body:
            self.visit(statement)
        self._bindings.pop()
        self._qualname, self._in_function, self._reader = saved

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_function(node, "<lambda>")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in [*node.decorator_list, *node.bases, *node.keywords]:
            self.visit(expression)
        saved = (self._qualname, self._in_function)
        self._qualname = self._child_qualname(node.name)
        self._in_function = False
        for statement in node.body:
            self.visit(statement)
        self._qualname, self._in_function = saved

    def visit_If(self, node: ast.If) -> None:
        if not _is_type_checking(node.test):
            self.visit(node.test)
            for statement in node.body:
                self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        source = _absolute_module(node, self.package)
        if source is not None:
            for alias in node.names:
                self._record(source, alias.name)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load):
            module_name = self._resolve(node.value)
            if module_name is not None:
                self._record(module_name, node.attr)
        self.generic_visit(node)


def _external_readers(targets: dict[str, set[str]]) -> dict[tuple[str, str], set[str]]:
    """Functions in other source modules or scripts that read a patched name at call time."""

    modules = {path: name for name, path in source_modules().items()}
    readers: dict[tuple[str, str], set[str]] = {}
    for path in code_files():
        text = source_text(path)
        words = set(re.findall(r"[A-Za-z_]\w*", text))
        relative = "from ." in text
        # Cheap textual prefilter before parsing: a reader names the module (by
        # full path, from its parent package or relatively) and reaches the
        # attribute through a dot or an indented ``from … import``.
        reachable = set(re.findall(r"\.\s*([A-Za-z_]\w*)", text))
        for match in _FROM_IMPORT.finditer(text):
            if match.group(1):
                reachable.update(name for name, _ in _imported_names(match.group(3)))
        candidates = {
            module_name: attributes
            for module_name, attributes in targets.items()
            if module_name.rpartition(".")[2] in words
            and (relative or module_name.rpartition(".")[0].rpartition(".")[2] in words)
            and not attributes.isdisjoint(reachable)
        }
        if not candidates:
            continue
        module_name = modules.get(path)
        if module_name is None:
            owner, package = path.relative_to(REPO_ROOT).as_posix(), ""
        else:
            owner = module_name
            package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
        scanner = _ExternalReadScanner(owner, package, candidates)
        for key, found in scanner.scan(ast.parse(text, str(path))).items():
            if owner != key[0]:
                readers.setdefault(key, set()).update(found)
    return readers


@cache
def monkeypatch_readers() -> dict[str, list[str]]:
    """Each patched ``module:attribute`` with the functions that read it at call time.

    Readers are the qualnames of functions in the patched module that read the
    name as a global, plus ``module:qualname`` (or ``script path:qualname``) of
    functions elsewhere that import it or access ``module.name`` when they run.
    An empty list means the patch reaches nothing.
    """

    targets = monkeypatch_targets()
    by_module: dict[str, set[str]] = {}
    for module_name, attribute in targets:
        by_module.setdefault(module_name, set()).add(attribute)
    readers: dict[tuple[str, str], set[str]] = {target: set() for target in targets}
    for module_name, attributes in by_module.items():
        for name, qualnames in _call_time_global_reads(
            _module_code(module_name), attributes
        ).items():
            readers[module_name, name].update(qualnames)
    for target, found in _external_readers(by_module).items():
        readers[target].update(found)
    return {
        f"{module_name}:{attribute}": sorted(readers[module_name, attribute])
        for module_name, attribute in sorted(readers)
    }


def collect_monkeypatch_readers() -> dict[str, list[str]]:
    return {target: found for target, found in monkeypatch_readers().items() if found}


def monkeypatched_in() -> dict[str, list[str]]:
    """Each patched ``module:attribute`` with the test files that patch it."""

    return {
        f"{module_name}:{attribute}": sorted(files)
        for (module_name, attribute), files in sorted(monkeypatch_targets().items())
    }


def load_monkeypatch_allowlist() -> dict[str, str]:
    return json.loads(MONKEYPATCH_ALLOWLIST.read_text(encoding="utf-8"))


# -------------------------------------------------------------------- snapshots

SNAPSHOTS: dict[str, Callable[[], object]] = {
    "durable_identities": collect_durable_identities,
    "store_capabilities": collect_store_capabilities,
    "public_surface": collect_public_surface,
    "server_routes": collect_server_routes,
    "monkeypatch_readers": collect_monkeypatch_readers,
}


def snapshot_path(name: str) -> Path:
    return SNAPSHOT_DIR / f"{name}.json"


def load_snapshot(name: str) -> Any:
    return json.loads(snapshot_path(name).read_text(encoding="utf-8"))


def write_snapshot(name: str, value: object) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path(name).write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the named snapshots (or every snapshot with --all) from the current source.",
    )
    parser.add_argument("--all", action="store_true", help="Select every snapshot.")
    parser.add_argument("names", nargs="*", metavar="NAME", help=f"any of {', '.join(SNAPSHOTS)}")
    arguments = parser.parse_args(argv)
    if arguments.all and arguments.names:
        parser.error("pass snapshot names or --all, not both")
    if arguments.write and not (arguments.all or arguments.names):
        parser.error(f"--write needs snapshot names ({', '.join(SNAPSHOTS)}) or --all")
    names = arguments.names or list(SNAPSHOTS)
    unknown = sorted(set(names) - set(SNAPSHOTS))
    if unknown:
        parser.error(f"unknown snapshots: {', '.join(unknown)}")
    stale = []
    for name in names:
        current = SNAPSHOTS[name]()
        if arguments.write:
            write_snapshot(name, current)
            print(f"wrote {snapshot_path(name).relative_to(REPO_ROOT)}")
        elif load_snapshot(name) != current:
            stale.append(name)
    if stale:
        print(f"stale snapshots: {', '.join(stale)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
