"""The application operation the current code runs in.

Kept below both the runtime and the components it composes, so a component can
attribute work it leaves running to the operation's owner without depending on
the runtime that defines owners.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Operations running in this context, innermost last; each exposes ``admission``,
# its owner. A stack, because one application's operation can call into another's.
operation_stack: ContextVar[tuple[Any, ...]] = ContextVar("cayu_application_operation", default=())


def current_operation_owner() -> object | None:
    """The owner of the innermost operation in this context, if any."""

    operations = operation_stack.get()
    return operations[-1].admission if operations else None
