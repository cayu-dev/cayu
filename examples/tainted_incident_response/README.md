# Tainted Incident Response

Part of Cayu's [advanced runtime example suite](../ADVANCED_RUNTIME_EXAMPLES.md).
See [Advanced runtime strategies](../../docs/advanced-runtime-examples.md) for
measured observations and proof boundaries.

The source agent reads a hostile prompt-injected ticket. A generic session fork
inherits the durable taint derived from that tool event. The quarantine policy
blocks a protected credential rotation, permits a sanitizer, and transfers only
an inert, provenance-bearing artifact into a new clean session.
The quarantine agent registers credential rotation only so the example can
exercise the runtime gate and assert its durable `tool.call.blocked` event. It
does not register notification or other outbound tools. The application is
reconstructed after the fork to prove the taint boundary survives `CayuApp`
reconstruction around the same store.

```bash
uv run python -m examples.tainted_incident_response.app
# Gemini
GEMINI_API_KEY=... uv run python -m examples.tainted_incident_response.app --mode live --provider gemini
# OpenAI
OPENAI_API_KEY=... uv run python -m examples.tainted_incident_response.app --mode live --provider openai
# Claude
ANTHROPIC_API_KEY=... uv run python -m examples.tainted_incident_response.app --mode live --provider anthropic
```

The safety assertion is the capability boundary and durable taint state—not
whether the model voluntarily follows a warning in its prompt.
