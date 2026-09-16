"""Count container allocations in compound readers under an installed wheel."""

import json
import sys
from unittest.mock import patch

import cayu._validation as validation
from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
from cayu.sessions.pending_actions import pending_action_evidence_round_from_checkpoint

original_walk = validation._walk_bounded_durable_json
original_frame = validation._BoundedDurableJsonFrame
original_path = validation._durable_child_path
counts = {}


def walk(value, field_name, **kwargs):
    if field_name == "checkpoint":
        counts["full_checkpoint_admissions"] += 1
    return original_walk(value, field_name, **kwargs)


def frame(*args, **kwargs):
    counts["container_frames_and_detached_containers"] += 1
    return original_frame(*args, **kwargs)


def path(*args, **kwargs):
    counts["child_paths"] += 1
    return original_path(*args, **kwargs)


with (
    patch.object(validation, "_walk_bounded_durable_json", walk),
    patch.object(validation, "_BoundedDurableJsonFrame", frame),
    patch.object(validation, "_durable_child_path", path),
):
    for reader in [
        user_input_lifecycle_authority_from_checkpoint,
        pending_action_evidence_round_from_checkpoint,
    ]:
        for size in [0, 1000]:
            counts = {
                "full_checkpoint_admissions": 0,
                "container_frames_and_detached_containers": 0,
                "child_paths": 0,
            }
            reader({"retained": [{"text": "history"} for _ in range(size)]})
            print(
                json.dumps(
                    dict(label=sys.argv[1], reader=reader.__name__, retained_records=size, **counts)
                )
            )
