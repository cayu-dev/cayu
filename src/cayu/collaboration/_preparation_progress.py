"""Private per-attempt evidence, never inferred from missing durable receipts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparationReadFailure:
    error: Exception


class PreparationProgress:
    """Native owner records entry before any potentially mutating dispatch.

    One instance belongs to one retained task. It cannot settle earlier attempts
    or certify that a dispatched write aborted, even if no receipt is visible.
    """

    def __init__(self):
        self._dispatched = False

    def enter_mutation(self):
        self._dispatched = True

    def read_failure(self, error: Exception):
        return None if self._dispatched else PreparationReadFailure(error)
