"""Detached, credential-safe dependency failure evidence."""

from __future__ import annotations

import asyncio

from cayu.vaults.redaction import SecretRedactor


def safe_cancelled_group_failure(
    error: BaseExceptionGroup, *, redactor: SecretRedactor
) -> BaseException | None:
    """Replace delivered cancellation with its causal evidence, in group order.

    A SQLite worker failure can be carried by the cancellation's explicit
    cause before rollback adds a group. Retain that evidence, not the delivered
    signal itself. Shared causes and cycles must not duplicate diagnostics.
    """
    seen: set[int] = set()

    def visit(value: BaseException, depth: int) -> BaseException | None:
        if id(value) in seen:
            return None
        if len(seen) >= 256:
            return RuntimeError("Additional collaboration failures withheld.")
        seen.add(id(value))
        if depth >= 32:
            return RuntimeError("Additional collaboration failures withheld.")
        if isinstance(value, asyncio.CancelledError):
            return visit(value.__cause__, depth + 1) if value.__cause__ is not None else None
        if isinstance(value, BaseExceptionGroup):
            children = []
            for item in value.exceptions:
                if len(seen) >= 256:
                    children.append(RuntimeError("Additional collaboration failures withheld."))
                    break
                child = visit(item, depth + 1)
                if child is not None:
                    children.append(child)
            return (
                BaseExceptionGroup("Collaboration dependency failures.", children)
                if children
                else None
            )
        return value

    evidence = visit(error, 0)
    return safe_failure(evidence, redactor=redactor) if evidence is not None else None


def safe_failure(error: BaseException, *, redactor: SecretRedactor) -> BaseException:
    from cayu.runtime._diagnostics import (
        credential_safe_runtime_exception,
        credential_safe_runtime_exception_group,
        exception_diagnostic,
    )

    def leaf(value: BaseException) -> BaseException:
        kind = (
            type(value)
            if type(value)
            in (
                ConnectionError,
                OSError,
                TimeoutError,
                ValueError,
                RuntimeError,
                asyncio.CancelledError,
            )
            else RuntimeError
        )
        return credential_safe_runtime_exception(
            kind,
            exception_diagnostic(value, redactor=redactor).message,
            redactor=redactor,
            fallback_message="Collaboration dependency failed.",
        )

    if isinstance(error, BaseExceptionGroup):
        return credential_safe_runtime_exception_group(
            error,
            group_message="Collaboration dependency failures.",
            leaf_mapper=leaf,
            invalid_leaf_factory=lambda: RuntimeError("Invalid dependency failure."),
            truncated_leaf_factory=lambda: RuntimeError("Additional dependency failures withheld."),
            fallback_leaf_mapper=lambda value: RuntimeError(),
            redactor=redactor,
        )
    return leaf(error)
