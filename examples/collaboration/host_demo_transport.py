"""Deterministic transport for the real Responses adapter; never opens a socket."""

import json
import re

from cayu import ExecutionProfileBehaviorIdentity
from cayu.providers.openai import OpenAIProvider


class OfflineProvider(OpenAIProvider):
    """Real adapter with an explicitly versioned application-owned transport.

    This declaration covers this finite synthetic responder only, not arbitrary
    OpenAI-compatible endpoints or user-replaceable transport behavior.
    """

    def __init__(self, transport):
        if type(transport) is not OfflineResponses:
            raise TypeError("The offline identity only qualifies the example transport.")
        super().__init__(
            api_key="offline-example-not-a-credential",
            name="offline",
            transport=transport,
            streaming=False,
        )

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="examples:collaboration-offline-responses",
            behavior_version="2",
            implementation_version="2",
        )


class OfflineResponses:
    def __init__(self):
        self.requests = []

    async def create_response(self, *, url, headers, payload, timeout_s):
        # Capture only serialized model input, never transport credentials.
        if len(self.requests) >= 48:
            raise RuntimeError("The finite example exhausted its provider-call allowance.")
        snapshot = json.loads(json.dumps(payload))
        self.requests.append(snapshot)
        text = json.dumps(snapshot.get("input", []), ensure_ascii=False)
        if "DEMO_RESUME" in text:
            answer = "Integrated the authorized specialist result."
        elif "DEMO_PARK" in text:
            answer = "Waiting for the specialist."
        elif matches := re.findall(r"DEMO_REVIEW_REVISION:([0-9a-f]{64})", text):
            if len(matches) != 1:
                raise ValueError("The finite responder requires one exact reviewed revision.")
            answer = "APPROVE_REVISION:" + matches[0]
        else:
            answer = "Specialist analysis completed for the selected input."
        identity = str(len(self.requests))
        return {
            "id": "resp_offline_" + identity,
            "model": payload["model"],
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_offline_" + identity,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": answer, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        }

    async def stream_response_events(self, **kwargs):
        raise RuntimeError("This example explicitly selects final-response transport.")
        yield  # pragma: no cover
