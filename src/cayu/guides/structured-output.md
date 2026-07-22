# Structured output without credentials

Use `StructuredOutputSpec` on `RunRequest` when the runtime must validate a JSON
value and repair invalid model output. The provider-neutral tool strategy is the
default. A successful tool-strategy run normally has an empty `final_text`; read
`outcome.structured_output.output` instead.

This complete example uses public imports only and exercises an invalid first
attempt followed by a valid repair:

```python
import asyncio

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    RunRequest,
    ScriptedModelProvider,
    StructuredOutputSpec,
    run_to_completion,
    scripted_structured_output,
)

provider = ScriptedModelProvider(
    [
        scripted_structured_output({"wrong": "shape"}),
        scripted_structured_output({"invoice_total": 42.5}),
    ]
)
app = CayuApp(enable_logging=False)
app.register_provider(provider, default=True)
app.register_agent(AgentSpec(name="invoice_analyst", model="scripted-model"))

request = RunRequest(
    agent_name="invoice_analyst",
    messages=[Message.text("user", "Extract the invoice total")],
    max_steps=2,
    structured_output=StructuredOutputSpec(
        name="invoice",
        max_retries=1,
        json_schema={
            "type": "object",
            "properties": {"invoice_total": {"type": "number"}},
            "required": ["invoice_total"],
            "additionalProperties": False,
        },
    ),
)
outcome = asyncio.run(run_to_completion(app, request))

assert outcome.ok
assert outcome.final_text == ""
assert outcome.structured_output is not None
assert outcome.structured_output.output == {"invoice_total": 42.5}
assert outcome.structured_output.attempt == 2
```

`scripted_structured_output` owns Cayu's reserved submission-tool name, argument
envelope, and model finish reason. Tests and evals should not reproduce that
wire protocol.

The full event stream remains available on `outcome.events`. A successful
validation emits `structured_output.validated` with `name`, `step`, `attempt`,
`max_retries`, `valid`, `errors`, and the redacted validated `output`. The typed
`StructuredOutputResult` wrapper distinguishes valid JSON `null` from a run
that produced no validated structured output.
