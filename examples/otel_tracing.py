from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    ChatCompletionsProvider,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    OpenTelemetryEventSink,
    ReadFileTool,
    RunRequest,
    WriteFileTool,
)

# Wire an OpenTelemetryEventSink so a cayu session is exported as a trace:
#   cayu.session  ──  chat <model>  ──  execute_tool <tool>
# This example prints spans to the console (no backend needed). To export to a
# real collector instead, swap ConsoleSpanExporter for an OTLP exporter. The sink
# is exporter-agnostic, so `cayu[otel]` installs only opentelemetry-api/-sdk; the
# OTLP path additionally needs `pip install opentelemetry-exporter-otlp-proto-grpc`.
# Then, for a local Jaeger (`docker run -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one`):
#
#   from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
#   provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317")))
#
# Requires `pip install cayu[otel]`. Runs against Google Gemini (free tier) via the
# OpenAI-compatible endpoint, like the other local-tools examples.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _build_tracer_provider():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return provider


async def main() -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this OpenTelemetry tracing example.")
        return

    model = os.environ.get("CAYU_GEMINI_MODEL", "gemini-2.5-flash")
    root = Path(__file__).resolve().parents[1] / ".examples-workspaces" / "otel-tracing"
    root.mkdir(parents=True, exist_ok=True)
    workspace = LocalWorkspace(root, workspace_id="otel-tracing-demo")
    runner = LocalRunner(root, inherit_env=False)

    tracer_provider = _build_tracer_provider()
    app = CayuApp(
        event_sinks=[OpenTelemetryEventSink(tracer=tracer_provider.get_tracer("cayu"))],
    )
    app.register_provider(
        ChatCompletionsProvider(
            name="gemini",
            api_key_env="GEMINI_API_KEY",
            base_url=GEMINI_BASE_URL,
            document_encoding="image_url",
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-dev", metadata={"kind": "local"}),
            workspace=workspace,
            runner=runner,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "You are testing Cayu local tools. Use exactly one tool call per "
                "assistant turn. Keep the final answer short."
            ),
        ),
        tools=[WriteFileTool(), ReadFileTool(), ListFilesTool(), ExecCommandTool()],
    )

    request = RunRequest(
        agent_name="assistant",
        session_id="demo_otel_tracing",
        messages=[
            Message.text(
                "user",
                (
                    "Create notes/result.txt with the text 'otel ok', then run a "
                    f"process command that prints {sys.executable!r}."
                ),
            )
        ],
    )
    async for _ in app.run(request):
        pass

    # Flush so the spans print before the process exits.
    tracer_provider.force_flush()
    print("Trace exported above (session -> model steps -> tool calls).")


if __name__ == "__main__":
    asyncio.run(main())
