from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
from pathlib import Path


def run(request: dict[str, object]) -> str:
    state = Path("trial-state")
    if state.exists():
        raise RuntimeError("fresh trial inherited mutable state")
    state.write_text("created by exactly one trial\n", encoding="utf-8")
    envelope = request["envelope"]
    if not isinstance(envelope, dict) or "trial" not in envelope:
        raise RuntimeError("trusted trial identity is missing")
    candidate_request = request["request"]
    if not isinstance(candidate_request, dict) or not candidate_request.get("messages"):
        raise RuntimeError("candidate input is missing")
    serialized = json.dumps(candidate_request, sort_keys=True)
    if '"assertions"' in serialized or '"expected"' in serialized:
        raise RuntimeError("hidden evaluator truth crossed the candidate boundary")
    if "CAYU_EVAL_TRUTH_SENTINEL" in os.environ:
        raise RuntimeError("evaluator environment authority crossed the container boundary")
    resolved = candidate_request.get("options", {}).get("cayu_file_attachments", {})
    for attachment in resolved.values():
        content = base64.b64decode(attachment["data_base64"], validate=True)
        if hashlib.sha256(content).hexdigest() != attachment["content_sha256"]:
            raise RuntimeError("attachment integrity is unproven")
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=0.1)
    except OSError:
        pass
    else:
        raise RuntimeError("direct candidate egress was available")
    return "Approved"
