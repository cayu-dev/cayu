# Responses output-index collision investigation

A completed message at output index 7 followed by a function-call addition at
index 7 remains a protocol error (`function_call_output_index_type_mismatch`).
The parser must not renumber either item or accept an incompatible identity.

The bounded native-adapter diagnostic now includes completed non-function items
in its registration and identity relations. `provider_protocol_stream_item_types`
is a JSON array of `[ordinal, incoming_item_type, registered_item_type]` rows,
aligned with `provider_protocol_stream_trace`. Both retain at most 16 entries.
Types use a closed vocabulary; absent and unknown types become `missing` and
`other`. Raw item/response identities, arguments, prompts, source content, and
provider payloads are absent. The existing trace retains item and response
identity relations rather than their raw values. Projection revalidates types,
bounds, and row alignment before persistence or export.

## Synthetic evidence

`tests/core/test_openai_output_index_collision.py` covers the stated shape,
two HTTP/SSE attempts with durable SQLite diagnostics, and local HTTP boundary
captures. The loopback fixture records the upstream sequence before a controlled
intermediary and the intermediary output before parsing. It replaces identities
with shared local aliases and retains only event type, index, item type, item
alias, and response alias. It records no original incident wire data.

| Injection boundary | Upstream message-done index | Upstream function-added index | Intermediary function-added index |
| --- | --- | --- | --- |
| Upstream | 7 | 7 | 7 |
| Intermediary | 7 | 8 | 7 |

The first fixture already conflicts upstream. In the second, the first divergence
is the intermediary's index rewrite; item and response aliases otherwise agree.
Both are rejected. Existing function/search ordering suites cover valid
interleaving, completion, usage, and bounded retry behavior.

## Verification

Run with this checkout's `src` and root first in `PYTHONPATH`:

```sh
python -m pytest -q tests/core/test_openai_output_index_collision.py tests/core/test_openai_function_ordering.py tests/core/test_openai_search_ordering.py
```

Loopback sockets require local socket permission. These synthetic controls
verify rejection and diagnostic behavior; they do not attribute the cause of
an unobserved provider or intermediary failure.
