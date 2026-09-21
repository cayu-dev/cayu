# TypeSafe / Jev (experimental)

`cayu.experimental.typesafe.TypeSafeProvider` integrates TypeSafe's native
System One API with Cayu model sessions, normalized deadlines, events and
usage accounting. It requires no new dependency beyond Cayu's HTTP client.

```python
from cayu import AgentSpec, CayuApp, Message, RunRequest, SQLiteSessionStore
from cayu.experimental.typesafe import TypeSafeProvider

app = CayuApp(session_store=SQLiteSessionStore("decisions.sqlite3"))
app.register_provider(TypeSafeProvider(), default=True)
app.register_agent(AgentSpec(
    name="fact-check",
    model="jev-latest",
    provider_options={"typesafe": {"questions": {
        "truth": {"type": "noul", "instructions": "Is this statement true?"},
    }}},
))
# Within an async function:
# async for event in app.run(RunRequest(
#     agent_name="fact-check",
#     messages=[Message.text("user", "Venus has no natural moons.")],
# )):
#     ...
```

Set `TYPESAFE_API_KEY` in the server environment. It is never a model option.
The endpoint is fixed to `https://api.typesafe.ai/v1/systemone`; redirects are
disabled. Transport failures and provider errors emit sanitized error events.

All questions in a request share the text state. Native `choice` questions take
a mapping of option names to descriptions; `score` questions take an ordered
list of descriptions; `noul` returns the probability of true. Responses are
validated against requested keys, question types and probability ranges.
The answer is serialized as JSON in the assistant message. Provider-reported
input/output tokens are passed through without fabricated usage.

This is a decision provider, not a chat/tool-use model. It rejects executable
tools, images, tool history, hosted tools, thinking and JSON-schema output
controls. Import it explicitly from `cayu.experimental.typesafe`; experimental
configuration and response shapes may change between releases.

Durable Cayu events preserve completed results. The HTTP API does not establish
remote idempotency or reconnect: a request interrupted before durable completion
may be billed again if retried. Runtime durability is not exactly-once vendor I/O.

Native API contract: https://docs.typesafe.ai/introduction/quickstart
