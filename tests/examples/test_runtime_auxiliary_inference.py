import asyncio

from examples.runtime_auxiliary_inference import LIMITS, MODEL, Summarize, run_demo

from cayu import AgentSpec, CayuApp, EventType, Message, ModelStreamEvent, RunRequest
from cayu.evals.testing import ScriptedModelProvider


def test_managed_auxiliary_example_counts_usage_without_an_extra_agent_step() -> None:
    assert asyncio.run(run_demo()) == {
        "provider_calls": 3,
        "ordinary_model_steps": 2,
        "total_tokens": 13,
        "auxiliary_attempts": 1,
        "auxiliary_purposes": ["tool.summary"],
        "auxiliary_outcomes": ["completed"],
    }


def test_request_factory_routes_managed_inference_without_consuming_positional_batches():
    seen = []

    def respond(request):
        if request.options.get("scripted", {}).get("max_output_tokens") == LIMITS.max_output_tokens:
            seen.append("auxiliary")
            assert not request.tools and not request.hosted_tools
            assert request.messages == [Message.text("user", "Long input")]
            result = ModelStreamEvent.text_delta("Short summary")
        elif any(message.role == "tool" for message in request.messages):
            seen.append("continuation")
            result = ModelStreamEvent.text_delta("Finished")
        else:
            seen.append("parent")
            result = ModelStreamEvent.tool_call(
                name="summarize", id="summary-call", arguments={"text": "Long input"}
            )
        return [
            result,
            ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
        ]

    async def run():
        provider = ScriptedModelProvider(response_factory=respond)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model=MODEL), tools=[Summarize()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="factory-auxiliary",
                    agent_name="assistant",
                    messages=[Message.text("user", "Summarize")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert seen == ["parent", "auxiliary", "continuation"]
        assert len(provider.requests) == 3
        usage = await app.get_session_usage("factory-auxiliary")
        assert usage.model_steps == 2 and usage.usage.total_tokens == 9
        attempts = [
            event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert len(attempts) == 1
        assert attempts[0].payload["auxiliary_outcome"] == "completed"

    asyncio.run(run())
