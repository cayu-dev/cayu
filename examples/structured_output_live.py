from __future__ import annotations

import asyncio
import os

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    Message,
    OpenAIProvider,
    RunRequest,
    StructuredOutputSpec,
)

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice": {
            "type": "object",
            "properties": {
                "number": {"type": "string"},
                "vendor": {"type": "string"},
                "status": {"type": "string", "enum": ["paid", "unpaid", "unknown"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "line_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                        "required": ["description", "amount"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                },
            },
            "required": ["number", "vendor", "status", "confidence", "line_items"],
            "additionalProperties": False,
        }
    },
    "required": ["invoice"],
    "additionalProperties": False,
}


async def main() -> None:
    provider_name = _provider_name()
    model = _model(provider_name)

    app = CayuApp()
    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY or choose CAYU_PROVIDER=anthropic.")
            return
        app.register_provider(OpenAIProvider(), default=True)
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Set ANTHROPIC_API_KEY or choose CAYU_PROVIDER=openai.")
            return
        app.register_provider(AnthropicProvider(), default=True)

    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "Extract invoice facts. Use the structured-output tool when the "
                "final answer is ready. Do not return final JSON as plain text."
            ),
        )
    )

    print("provider", provider_name)
    print("model", model)

    request = RunRequest(
        agent_name="assistant",
        session_id=f"demo_{provider_name}_structured_output",
        messages=[
            Message.text(
                "user",
                (
                    "Invoice INV-1042 from Acme Cloud is marked PAID. "
                    "It has one line item: Managed hosting for 125.50 USD. "
                    "Return the invoice status with confidence 0.91."
                ),
            )
        ],
        structured_output=StructuredOutputSpec(
            name="invoice_status",
            json_schema=INVOICE_SCHEMA,
            max_retries=1,
            repair_prompt=(
                "Call the structured-output tool again with an `output` object "
                "that exactly matches the invoice schema."
            ),
        ),
    )

    async for event in app.run(request):
        print(
            event.type,
            event.environment_name or "-",
            event.tool_name or "-",
            event.payload,
        )


def _provider_name() -> str:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
        if requested in {"openai", "anthropic"}:
            return requested
        raise RuntimeError("CAYU_PROVIDER must be openai or anthropic.")
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _model(provider_name: str) -> str:
    if provider_name == "openai":
        return os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.5")
    return os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")


if __name__ == "__main__":
    asyncio.run(main())
