from __future__ import annotations

import asyncio
import json
import sys

from cayu import (
    AgentSpec,
    CayuApp,
    EvaluationEvidencePolicySpec,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    build_promotion_candidate,
    export_promotion_corpus,
    score_promotion_candidate,
    trajectory_from_session,
)


async def _document() -> dict[str, object]:
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("deterministic answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ),
            ],
            name="scripted",
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="deterministic-model",
            system_prompt="Answer deterministically.",
        )
    )
    session_id = "deterministic-promotion-source"
    async for _ in app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "deterministic request")],
        )
    ):
        pass
    trajectory = await trajectory_from_session(app, session_id)
    candidate = build_promotion_candidate(
        app,
        trajectory,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-deterministic",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    score = score_promotion_candidate(
        app,
        trajectory,
        candidate,
        target_key="assistant",
        source_agent_name="assistant",
        application_release_id="release-deterministic",
    )
    return {
        "candidate": candidate.model_dump(mode="json"),
        "score": score.model_dump(mode="json"),
        "corpus": export_promotion_corpus(candidate).decode("utf-8"),
    }


def main() -> None:
    sys.stdout.write(
        json.dumps(
            asyncio.run(_document()),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
