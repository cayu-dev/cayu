"""Errors of the fork-controller entrance; ``cayu.fork_controllers`` exports them."""

from __future__ import annotations


class ForkControllerDenied(PermissionError):
    """The caller is not an enrolled controller, or the target is outside its enrollment."""
