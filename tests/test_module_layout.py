"""Guardrails that keep behaviour-neutral module splits behaviour-neutral.

Each check compares the current source with a snapshot committed under
``tests/fixtures/module_layout/``. A failure names what a move changed and why
it matters. The checks are one-way: every committed entry must still hold, and
additions (a new logger, tool, export or patch reader) pass. The route order
and the browser worker files are compared exactly, because an addition there
changes behaviour. Regenerate a snapshot only for an intended, reviewed
change::

    uv run python -m tests.module_layout_support --write NAME [NAME ...]
"""

from __future__ import annotations

import importlib

import pytest

from tests import module_layout_support as layout


@pytest.mark.parametrize("category", sorted(layout.DURABLE_IDENTITY_COLLECTORS))
def test_durable_identities_do_not_move(category: str) -> None:
    expected = layout.load_snapshot("durable_identities")
    assert set(expected) == set(layout.DURABLE_IDENTITY_COLLECTORS), (
        "durable_identities.json and DURABLE_IDENTITY_COLLECTORS list different categories; "
        f"run `{layout.REGENERATE_COMMAND} durable_identities`."
    )
    exact = category in layout.EXACT_DURABLE_IDENTITIES
    why = "It is persisted: " + layout.DURABLE_IDENTITY_REASONS[category]
    if exact:
        why += " Compared exactly: " + layout.EXACT_DURABLE_IDENTITIES[category]
    mismatch = layout.snapshot_mismatch(
        f"Durable identity {category!r}",
        expected[category],
        layout.DURABLE_IDENTITY_COLLECTORS[category](),
        why,
        snapshot="durable_identities",
        exact=exact,
    )
    assert mismatch is None, mismatch


def test_pinned_identities_name_the_defining_module() -> None:
    identities = layout.load_snapshot("durable_identities")
    unresolved = {
        target: binding
        for target, binding in identities["behavior_identity_strings"].items()
        if binding not in {"defined", "assigned"}
    }
    assert not unresolved, f"identity strings no longer name their definition: {unresolved}"
    moved_stores = {
        target: actual
        for target, actual in identities["support_bundle_stores"].items()
        if actual != target
    }
    assert not moved_stores, f"support-bundle store identities resolve elsewhere: {moved_stores}"
    for module_name, names in identities["compatibility_re_exports"].items():
        shim = importlib.import_module(module_name)
        for name, canonical in names.items():
            if ":" in canonical:
                assert getattr(shim, name) is layout.resolve(canonical), f"{module_name}.{name}"


@pytest.mark.parametrize(
    "store",
    [*layout.SESSION_STORES, *layout.TASK_STORES, *layout.BUDGET_LEDGERS],
)
def test_store_capabilities_do_not_flip(store: str) -> None:
    expected = layout.load_snapshot("store_capabilities")
    actual = layout.collect_store_capabilities()
    assert set(expected) == set(actual), (
        f"The shipped store list changed; run `{layout.REGENERATE_COMMAND} store_capabilities`."
    )
    mismatch = layout.snapshot_mismatch(
        f"Capabilities of {store}",
        expected[store],
        actual[store],
        "Stores attest optional protocols by which class in the MRO defines each method and "
        "declares the version flag (sessions/store.py, runtime/_task_store_operation_boundary.py). "
        "A ~ entry is a flipped probe: the runtime takes a refusal or fallback path while the "
        "store still passes ordinary tests, and a mixin that carries attested methods must "
        "declare the matching version ClassVar. A - entry is a probe or method the collector no "
        "longer discovers: if it was renamed or moved, update the collector in "
        "tests/module_layout_support.py.",
        snapshot="store_capabilities",
    )
    assert mismatch is None, mismatch


def test_monkeypatches_reach_a_reader() -> None:
    allowlist = layout.load_monkeypatch_allowlist()
    readers = layout.monkeypatch_readers()
    patched_in = layout.monkeypatched_in()
    unread = {target for target, found in readers.items() if not found}
    unexpected = sorted(unread - set(allowlist))
    assert not unexpected, (
        "These tests patch a module attribute that no function in that module (and no "
        "call-time import or `module.name` access elsewhere) reads, so the patch changes "
        "nothing. Retarget each patch to the module that now reads the name, or, if the patch "
        f"is deliberately inert, add it with a reason to {layout.MONKEYPATCH_ALLOWLIST.name}:\n"
        + "\n".join(
            f"  {target} patched in {', '.join(patched_in[target])}" for target in unexpected
        )
    )
    stale = sorted(set(allowlist) - unread)
    assert not stale, f"Remove allowlist entries that are now read or no longer patched: {stale}"


def test_monkeypatched_names_keep_their_readers() -> None:
    expected = layout.load_snapshot("monkeypatch_readers")
    readers = layout.monkeypatch_readers()
    patched_in = layout.monkeypatched_in()
    lost = [
        f"  {target} (patched in {', '.join(patched_in[target])}) is no longer read by "
        + ", ".join(sorted(set(previous) - set(readers[target])))
        for target, previous in expected.items()
        if target in readers and not set(previous) <= set(readers[target])
    ]
    assert not lost, (
        "A function that read a patched name at call time no longer does, so the patch may "
        "no longer reach the code it was written for. If the reader moved to another module, "
        "retarget the patch to the module that now reads the name (and move the module-level "
        "names it reads with it). If the reader was removed on purpose, regenerate the "
        f"snapshot with `{layout.REGENERATE_COMMAND} monkeypatch_readers`:\n"
        + "\n".join(layout.truncated(lost))
    )


def test_public_import_surface_still_resolves() -> None:
    surface = layout.load_snapshot("public_surface")
    failures: list[str] = []
    for section, modules in surface.items():
        for module_name, entry in modules.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                failures.append(f"{section}: import {module_name} failed: {exc}")
                continue
            for name in entry["present"]:
                if not hasattr(module, name):
                    failures.append(f"{section}: {module_name}.{name} no longer resolves")
            for name, expected in layout.surface_identities(entry).items():
                if not hasattr(module, name):
                    failures.append(f"{section}: {module_name}.{name} no longer resolves")
                    continue
                actual = layout.identity(getattr(module, name))
                if actual != expected:
                    failures.append(f"{section}: {module_name}.{name} is {actual}, was {expected}")
    assert not failures, (
        "Public exports and imports written in README, docs, guides and examples must keep "
        "resolving to the same objects. Keep the name importable at its documented path "
        "(re-export it) and keep public classes in their defining module:\n"
        + "\n".join(layout.truncated([f"  {failure}" for failure in failures]))
        + f"\nIf a change is intended, run `{layout.REGENERATE_COMMAND} public_surface`."
    )


def test_server_route_registration_is_unchanged() -> None:
    mismatch = layout.snapshot_mismatch(
        "Server route registration",
        layout.load_snapshot("server_routes"),
        layout.collect_server_routes(),
        "FastAPI matches routes in registration order, so the list is compared exactly and in "
        "order, and handler names become the OpenAPI operationIds the dashboard API types are "
        "generated from.",
        snapshot="server_routes",
        exact=True,
    )
    assert mismatch is None, mismatch


def test_route_order_check_counts_duplicated_routes() -> None:
    routes = ["GET /api/a", "GET /api/{id}", "GET /api/b"]

    def mismatch(actual: list[str]) -> str | None:
        return layout.snapshot_mismatch(
            "routes", routes, actual, "order matters", snapshot="server_routes", exact=True
        )

    assert mismatch(list(routes)) is None
    assert mismatch(["GET /api/{id}", *routes]) is not None
    assert mismatch(routes + routes) is not None
