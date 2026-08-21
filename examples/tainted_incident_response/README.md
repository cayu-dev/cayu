# Tainted Incident Response

Part of Cayu's [advanced runtime example suite](../ADVANCED_RUNTIME_EXAMPLES.md).
See [Advanced runtime strategies](../../docs/advanced-runtime-examples.md) for
measured observations and proof boundaries.

The source agent reads a hostile prompt-injected ticket. A generic session fork
inherits the durable taint derived from that tool event. The quarantine policy
blocks a protected credential rotation, permits a sanitizer, and transfers only
an inert, provenance-bearing artifact into a new clean session.
The source registration includes credential rotation inside its durable
capability ceiling, but a static source tool view keeps that operation hidden
from the model. The quarantine agent can therefore expose the inherited
capability after the explicitly authorized profile transition without widening
the fork's ceiling. Its taint policy blocks the attempted rotation and produces
the durable `tool.call.blocked` event asserted by the example. The quarantine
agent does not register notification or other outbound tools. The application
is reconstructed after the fork to prove the taint and capability boundaries
survive `CayuApp` reconstruction around the same store.
Its final safety assertions read the inherited taint, blocked policy decision,
source event identity, recovery summary, and receipt identities from the public
bounded `runtime_evidence(app, request)` projection. Raw events are retained
only while driving the live retry/control flow, not as the scenario's evidence
aggregation contract.

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

## Independent safety contracts

Reading the stable ticket and sanitizing in memory declare `ToolEffect.NONE`;
sending a notification and rotating credentials declare `ToolEffect.EXTERNAL`
because replay can repeat durable mutation. These declarations do not grant
authority: `TaintAwareToolPolicy` independently enforces it. Run
`cayu guide tool-effects` for the canonical decision table.
