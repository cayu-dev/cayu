# Hosted-search source discriminator diagnostics

Issue #1448 adds evidence for unsupported Responses hosted-search source types.
It does not establish the historical offending type or expand supported variants.

## Contract and retained evidence audit

The official [Responses Python schema](https://developers.openai.com/api/reference/python/resources/responses.md)
was retrieved during this change. `ActionSearchSource.type` is `Literal["url"]`
and `url` is a string. The generated Python SDK agrees:
[`response_function_web_search.py`](https://github.com/openai/openai-python/blob/be928151372e4b62adb4a1571cda52ad759b38be/src/openai/types/responses/response_function_web_search.py).
No additional supported source variant was confirmed. Cayu keeps its existing
compatibility behavior for an omitted discriminator, normalizing it to `url`.
Explicit null and other values remain unsupported. Sources are never silently
dropped or coerced into invented URLs.

The issue and retained [GAIA #114](https://github.com/cayu-tech/cayu-gaia/issues/114)
comments identify Runtime `f2247e91c6095e5e95f81682caaf67c5a5ce40f5`, GAIA
`0bed305ba390b1fe499d0b1fc067dbe9d8cff2c4`, and `https://codex-lb.cayu.ai`.
They record event 6918's stable protocol reason, event 6919's retry and the
successful completion at event 6925, but omit the actual discriminator and index.
The referenced remote SQLite database is not available in this local workspace;
local smoke notes contain no recovered discriminator. The historical value is
**unknown**. No raw remote response or proxy implementation was available for
this change. No paid probe was run. Hermetic transport tests verify the configured
proxy URL is `/v1/responses`; they do not verify live proxy behavior.
The evidence establishes an unresolved incompatibility, not a confirmed adapter
gap or provider defect. Synthetic fixtures (including `api`) are boundary tests,
not reproductions or assertions about the historical value. Any later support
change requires a confirmed schema and exact pinned proxy-path evidence.

## Diagnostic contract

The stable reason remains `web_search_action_sources_type_is_unsupported`, stage
`hosted_tool`, field `output[].action.sources[].type`. Additional scalar fields:

| Field suffix after `provider_protocol_` | Meaning |
| --- | --- |
| `source_index` | Zero-based index, 0 through 99, within the failing source list |
| `source_type_kind` | `null`, `boolean`, `number`, `string`, `array`, `object`, or `non_json` |
| `source_supported_types` | `url` |
| `source_type_value_status` | `retained` or `omitted` |
| `source_type_value` | Optional canonical diagnostic label |

Only exact built-in strings matching the small diagnostic vocabulary `url`,
`api`, `file`, `image`, `document`, `text` can supply a label. This vocabulary is
not a support enum. Unknown enum strings, arbitrary text, URLs, credentials,
control characters, string subclasses, and strings longer than 32 characters
are omitted, without truncation, hashing, stringification, or retaining raw values.
Objects/arrays provide only their JSON kind. Syntax-only validation cannot prove
that an arbitrary enum-shaped string is not a secret, so unknown labels fail
closed. Labels matching configured API/header credential values are also omitted
at the provider boundary. Runtime's existing workload-secret redaction still
applies. Source bodies, titles, URLs, and complete payloads never enter these
additional diagnostics or exception messages.

The typed exception retains bounded evidence during completed-response parsing.
Streaming item completion and terminal-response parsing share the same validator.
Safe background exception wrapping preserves diagnostics; retrieval and reconnect
failures retain them on `provider.operation.recovery_required` events. Normal
stream failures retain them on `model.error`, including durable SQLite readback.
A successful retry does not erase the original failure. Unknown-outcome retry
limits and hosted-tool effect accounting remain in force. Background recovery
of an already identified operation remains malformed/manual recovery and never
starts another provider request merely because decoding failed.
