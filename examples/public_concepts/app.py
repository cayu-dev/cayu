"""Run a real Cayu session locally with a deterministic model provider."""

from __future__ import annotations

import asyncio

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context import DefaultContextPolicy
from cayu.evals import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore, RunRequest


def build_app() -> CayuApp:
    app = CayuApp(
        session_store=InMemorySessionStore(),
        enable_logging=False,
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("Hello from the public concept layout."),
                ModelStreamEvent.completed(),
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="greeter", model="local-script"),
        context_policy=DefaultContextPolicy(),
    )
    return app


async def run_example() -> list[Event]:
    app = build_app()
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="greeter",
                session_id="public-concepts-demo",
                messages=[Message.text("user", "Say hello.")],
            )
        )
    ]


def main() -> None:
    events = asyncio.run(run_example())
    for event in events:
        if event.type in {EventType.MODEL_TEXT_DELTA, EventType.SESSION_COMPLETED}:
            print(event.type, event.payload)
    print(f"Recorded {len(events)} runtime events.")


if __name__ == "__main__":
    main()
