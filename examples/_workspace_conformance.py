from __future__ import annotations

from collections.abc import Awaitable, Callable

from cayu import Workspace

CONFORMANCE_PATH = ".cayu-conformance/workspace.txt"


async def verify_portable_workspace_round_trip(
    workspace: Workspace,
    *,
    adapter: str,
) -> None:
    """Exercise portable Workspace behavior against a deterministic or live adapter."""

    await workspace.delete(CONFORMANCE_PATH)
    try:
        await _require_not_found(
            lambda: workspace.read_bytes(CONFORMANCE_PATH),
            adapter=adapter,
            operation="read-missing",
        )
        await workspace.write_bytes(CONFORMANCE_PATH, b"first")
        first = await workspace.read_bytes(CONFORMANCE_PATH)
        _require(
            first.content == b"first" and first.total_bytes == 5 and first.truncated is False,
            adapter=adapter,
            operation="read-after-write",
            observed=first,
        )
        await workspace.write_bytes(CONFORMANCE_PATH, b"second")
        second = await workspace.read_bytes(CONFORMANCE_PATH)
        _require(
            second.content == b"second",
            adapter=adapter,
            operation="read-after-overwrite",
            observed=second,
        )
        listed = await workspace.list("**/*")
        _require(
            CONFORMANCE_PATH in listed.paths,
            adapter=adapter,
            operation="list-after-write",
            observed=listed,
        )
    finally:
        await workspace.delete(CONFORMANCE_PATH)
    await _require_not_found(
        lambda: workspace.read_bytes(CONFORMANCE_PATH),
        adapter=adapter,
        operation="read-after-delete",
    )


async def verify_portable_workspace_path_safety(
    workspace: Workspace,
    *,
    adapter: str,
) -> None:
    """Prove every portable operation rejects invalid raw relative paths."""

    for path in (
        "",
        ".",
        "./",
        "nested/..",
        "nested/../accepted.txt",
        "/absolute.txt",
        "../outside.txt",
        "nested/../../outside.txt",
    ):
        await _require_validation_rejection(
            lambda path=path: workspace.read_bytes(path),
            adapter=adapter,
            operation="read",
            value=path,
        )
        await _require_validation_rejection(
            lambda path=path: workspace.write_bytes(path, b"blocked"),
            adapter=adapter,
            operation="write",
            value=path,
        )
        await _require_validation_rejection(
            lambda path=path: workspace.delete(path),
            adapter=adapter,
            operation="delete",
            value=path,
        )
    for pattern in ("", "/**/*", "../*", "nested/../../*"):
        await _require_validation_rejection(
            lambda pattern=pattern: workspace.list(pattern),
            adapter=adapter,
            operation="list",
            value=pattern,
        )


async def _require_not_found(
    action: Callable[[], Awaitable[object]],
    *,
    adapter: str,
    operation: str,
) -> None:
    try:
        await action()
    except FileNotFoundError:
        return
    except Exception as exc:
        raise AssertionError(
            f"scenario=workspace-portable-round-trip adapter={adapter} "
            f"operation={operation} expected=FileNotFoundError "
            f"observed={type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(
        f"scenario=workspace-portable-round-trip adapter={adapter} "
        f"operation={operation} expected=FileNotFoundError observed=success"
    )


async def _require_validation_rejection(
    action: Callable[[], Awaitable[object]],
    *,
    adapter: str,
    operation: str,
    value: str,
) -> None:
    try:
        await action()
    except (TypeError, ValueError):
        return
    except Exception as exc:
        raise AssertionError(
            f"scenario=workspace-portable-path-safety adapter={adapter} "
            f"operation={operation} value={value!r} "
            f"expected=validation-error observed={type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(
        f"scenario=workspace-portable-path-safety adapter={adapter} "
        f"operation={operation} value={value!r} expected=validation-error observed=success"
    )


def _require(
    condition: bool,
    *,
    adapter: str,
    operation: str,
    observed: object,
) -> None:
    if condition:
        return
    raise AssertionError(
        f"scenario=workspace-portable-round-trip adapter={adapter} "
        f"operation={operation} observed={observed!r}"
    )
