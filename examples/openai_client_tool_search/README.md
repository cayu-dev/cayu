# OpenAI client Tool Search validation

This credential-free fixture runs the complete native discovery vertical
through the public Cayu runtime and the real `OpenAIProvider` adapter. A
deterministic transport returns one OpenAI `tool_search_call`, observes Cayu's
matching `tool_search_output`, calls the loaded application function directly,
and finishes after Cayu publishes the ordinary function result.

```bash
PYTHONPATH=src python -m examples.openai_client_tool_search.scenario
```

The report proves that the registered application schema is absent from every
top-level provider tool array, becomes available only through the search
output, and executes exactly once through Cayu. It does not contact OpenAI,
require credentials, benchmark a model, or establish support for a real model
id. Production applications must list each separately verified exact model id
in `OpenAIProvider.client_tool_search_models`.
