# OpenAI hosted Tool Search validation

This credential-free fixture runs the complete server-executed Tool Search
vertical through the public Cayu runtime and the real `OpenAIProvider` adapter.
A deterministic transport returns the exact adjacent OpenAI
`tool_search_call` / `tool_search_output` pair and a function call in one
response. Cayu validates the loaded definition against its registered
catalogue and session ceiling, atomically creates a branch-local grant, runs
the ordinary application tool once, and replays the provider state with its
tool result on the final request.

```bash
PYTHONPATH=src python -m examples.openai_hosted_tool_search.scenario
```

The report proves adapter composition, replay ordering, and Cayu's durable
authority boundary. It does not contact OpenAI, require credentials, measure a
prompt-cache hit, benchmark a model, or establish support for a production
model id. Production applications must separately verify and list each exact
model id in `OpenAIProvider.hosted_tool_search_models`. OpenAI defines when
deferred definitions enter model context and how cached-input billing is
reported; applications should verify those provider-side properties with an
opt-in live contract test for their chosen model.
