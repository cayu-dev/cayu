"""Minimal CLI protocols with a controlled exit after the final answer."""

import json
import os
import sys
import time


def send(value):
    print(json.dumps(value), flush=True)


def finish():
    if os.environ.get("CLI_EXIT") == "hold":
        while True:
            time.sleep(1)
    sys.exit(int(os.environ.get("CLI_EXIT", "1")))


if "--version" in sys.argv:
    print("2.1.0")
    sys.exit(0)

for line in sys.stdin:
    message = json.loads(line)
    if "app-server" not in sys.argv:
        if message.get("type") == "control_request":
            send(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": {},
                    },
                }
            )
        elif message.get("type") == "user":
            send(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "num_turns": 1,
                    "result": "answer",
                    "session_id": "test",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "total_cost_usd": 0,
                }
            )
            finish()
    else:
        method, rid = message.get("method"), message.get("id")
        if method == "initialize":
            send({"id": rid, "result": {"userAgent": "test"}})
        elif method == "thread/start":
            thread = {
                "id": "thread",
                "sessionId": "thread",
                "createdAt": 1,
                "updatedAt": 1,
                "cwd": os.getcwd(),
                "ephemeral": True,
                "modelProvider": "openai",
                "preview": "",
                "source": "appServer",
                "status": {"type": "idle"},
                "turns": [],
                "cliVersion": "0.163.0",
            }
            send(
                {
                    "id": rid,
                    "result": {
                        "thread": thread,
                        "model": "test",
                        "modelProvider": "openai",
                        "cwd": os.getcwd(),
                        "approvalPolicy": "untrusted",
                        "approvalsReviewer": "user",
                        "sandbox": {"type": "readOnly"},
                    },
                }
            )
        elif method == "turn/start":
            send(
                {
                    "id": rid,
                    "result": {
                        "turn": {"id": "turn", "status": "inProgress", "items": [], "error": None}
                    },
                }
            )
            send(
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread",
                        "turnId": "turn",
                        "completedAtMs": 1,
                        "item": {
                            "id": "answer",
                            "type": "agentMessage",
                            "phase": "final_answer",
                            "text": "answer",
                        },
                    },
                }
            )
            send(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread",
                        "turn": {"id": "turn", "status": "completed", "items": [], "error": None},
                    },
                }
            )
            finish()
