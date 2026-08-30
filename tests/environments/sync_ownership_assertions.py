from __future__ import annotations

import cayu.environments.bindings as bindings_module
from cayu.environments import BoundWorkspace
from cayu.workspaces import Workspace


def assert_sync_resources_owned(
    source_or_bound: Workspace | BoundWorkspace,
    target: Workspace | None = None,
    *,
    generation: str | None = None,
    expected: bool,
) -> None:
    """Assert both of one generation's claims in the authoritative registry."""

    if isinstance(source_or_bound, BoundWorkspace):
        assert target is None
        assert generation is None
        source = source_or_bound.source_workspace
        target = source_or_bound.workspace
        generation = source_or_bound.state_key
    else:
        source = source_or_bound
    assert source is not None
    assert target is not None
    assert generation is not None
    source_key = source.resource_key
    target_key = target.resource_key
    assert source_key is not None
    assert target_key is not None
    with bindings_module._SYNC_RESOURCE_OWNERS_LOCK:
        actual = {
            "source": bindings_module._SYNC_RESOURCE_OWNERS.get(source_key) == generation,
            "target": bindings_module._SYNC_RESOURCE_OWNERS.get(target_key) == generation,
        }
    assert actual == {"source": expected, "target": expected}
