"""The host example uses public owners and documents separate settlement states."""

import ast
from pathlib import Path


def test_retained_producer_example_uses_public_owner_steps():
    root = Path(__file__).resolve().parents[1]
    source = (root / "examples/collaboration/retained_producer_output.py").read_text()
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
    ]
    assert set(calls) == {
        "register_producer_output",
        "execute_producer_output",
        "retain_producer_completion",
        "export_producer_output",
        "publish_producer_outcome",
        "deliver_producer_output",
    }
    assert all(not name.startswith("_") for name in calls)
    assert "aclosing" in source
    section = (
        (root / "docs/collaboration-requests.md")
        .read_text()
        .split("## Retained producer output\n", 1)[1]
        .split("\n## ", 1)[0]
    )
    normalized = " ".join(section.split())
    from cayu import CayuApp

    disposition_doc = CayuApp.service_producer_disposition.__doc__
    assert disposition_doc is not None
    assert "frozen stop scope" in disposition_doc
    assert "detach" not in disposition_doc
    for contract in (
        "Historical admission readback is not permission to execute",
        "requires the request's `stop` cancellation disposition",
        "performs no registration or dispatch",
        "Provider exposure still requires",
        "a terminal session status is insufficient",
        "match/not-found/conflict/unavailable",
        "cannot turn an already appended delivery into an exclusion",
        "Mandatory owner-internal cleanup remains separate",
        "Memory, SQLite and PostgreSQL",
        "not a scheduler",
    ):
        assert contract in normalized
