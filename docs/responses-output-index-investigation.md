# Responses output-index collision investigation

Tracking: #1591; related ordering investigation: #1495. These are separate
failure shapes; no common cause has been established.

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

## Reproduction baseline and limits

Baseline: Runtime `eae5abdfd6ec5d166ef011b655cad2ae7adaec80`, version 0.4.0,
including the provider lifecycle centralization. Environment: macOS 15.7.7 arm64,
Python 3.14.3, httpx 0.28.1, httpcore 1.0.9, anyio 4.13.0, pytest 9.0.3.
The PR records the exact tested implementation commit. The native Responses
adapter uses httpx; no OpenAI SDK or live provider/model version was exercised.

Run with this checkout's `src` and root first in `PYTHONPATH`:

```sh
python -m pytest -q tests/core/test_openai_output_index_collision.py tests/core/test_openai_function_ordering.py tests/core/test_openai_search_ordering.py
```

The focused suite passes 110 tests. Loopback sockets require local socket
permission. No paid probe, deployed intermediary, original upstream capture,
Linux/Windows run, or full qualification was performed. The historical
upstream/intermediary versions and first divergence remain unknown. These
controls improve diagnosis; they do not attribute or resolve the original
incident, and #1591 remains an open investigation.
