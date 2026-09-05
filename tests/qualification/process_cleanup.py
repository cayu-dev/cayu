"""Bounded cleanup of process groups created by qualification."""

import os
import signal
import time
from contextlib import suppress
from pathlib import Path

GROUP_REGISTRY_ENV = "CAYU_QUALIFICATION_GROUP_REGISTRY"


def register_process_group(group_id):
    registry = os.environ.get(GROUP_REGISTRY_ENV)
    if registry is None:
        return
    # One append per launch keeps identities available if pytest cannot finish.
    descriptor = os.open(registry, os.O_WRONLY | os.O_APPEND)
    try:
        record = f"{group_id}\n".encode("ascii")
        if os.write(descriptor, record) != len(record):
            raise OSError("Incomplete qualification process-group registration")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def registered_process_groups(registry):
    invalid = False
    for record in Path(registry).read_text(encoding="ascii").splitlines(keepends=True):
        identity = record.rstrip("\n")
        if not record.endswith("\n") or not identity.isdecimal() or int(identity) <= 1:
            invalid = True
            continue
        yield int(identity)
    # Let the caller retain all complete identities for cleanup before failing.
    if invalid:
        raise ValueError("Invalid qualification process-group identity")


def process_group_exists(group_id):
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # Inability to inspect an owned group cannot certify its absence.
        return True
    return True


def drain_process_groups(group_ids, *, timeout=5):
    pending = set(group_ids)
    for group_id in pending:
        with suppress(OSError):
            os.killpg(group_id, signal.SIGKILL)
    deadline = time.monotonic() + timeout
    while pending:
        pending = {group_id for group_id in pending if process_group_exists(group_id)}
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    return len(pending)
