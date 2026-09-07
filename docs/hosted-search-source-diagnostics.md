# Hosted-search source discriminator diagnostics

## Supported evidence

Hosted search accepts URL sources (`WebSearchSource`) and named API sources
(`WebSearchAPISource`). `WebSearchAction.sources` preserves both as distinct
variants. API sources have a bounded nonblank `name` (up to 1,024 characters,
portable durable text without surrounding whitespace), not a URL. The name is
retained exactly; it is not an enum, a URL, or a citation target.

A bounded direct Responses API comparison on September 7, 2026 returned this
structure for weather and finance searches. A compatible HTTP path also received
and forwarded these source fields unchanged:

```json
{"type": "api", "name": "redacted"}
```

`redacted` is an explicit replacement, not an observed source-name value. The
comparison did not retain exact names. This support follows observed wire fields;
it does not claim that every published provider schema declares this variant.

URL sources retain their existing constructor and validation. An omitted source
discriminator still means `url`; explicit null is not omission. API sources are
never converted to URL sources, assigned fabricated URLs, or discarded.

Both item-completion and completed-response parsing use the same normalizer.
The normalized action enters hosted-tool events and typed `HostedToolCallPart`
transcript evidence through `WebSearchAction`. Session serialization preserves
the variant and name; provider continuation state retains the normalized source
object for replay. The server schema and generated client expose both variants.
Consumers that need a link must check `source.type == "url"` before reading `url`.

Unsupported discriminators raise `OpenAIUnsupportedSearchSourceError`, a subclass
of `OpenAIProtocolError`. Streaming exposes
`provider_error_type=unsupported_capability`, `retryable=false`, and stops with
`retry_disposition=explicit_nonretryable`. Repeating a hosted search cannot repair
an unsupported decoding contract. Malformed API names use the static diagnostic
`web_search_action_sources_name_is_invalid`, stage `hosted_tool`, field
`output[].action.sources[].name`, and the existing bounded protocol retry policy.
No source name or body is included in these diagnostics.

The observed-shape fixture covers completed parsing, streaming item completion,
terminal-only streams, mixed URL/API sources, durable readback, and provider-state
replay. Synthetic malformed and unknown variants cover safe diagnostics and
bounded retries. HTTP source compatibility does not qualify WebSocket ordering,
concurrency, or idle behavior.

## Diagnostic contract

The stable reason remains `web_search_action_sources_type_is_unsupported`, stage
`hosted_tool`, field `output[].action.sources[].type`. Additional scalar fields:

| Field suffix after `provider_protocol_` | Meaning |
| --- | --- |
| `source_index` | Zero-based index, 0 through 99, within the failing source list |
| `source_type_kind` | `null`, `boolean`, `number`, `string`, `array`, `object`, or `non_json` |
| `source_supported_types` | `url,api` |
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
applies. Source names, bodies, titles, URLs, and complete payloads never enter these
additional diagnostics or exception messages.

The typed exception retains bounded evidence during completed-response parsing.
Streaming item completion and terminal-response parsing share the same validator.
Safe background exception wrapping preserves diagnostics; retrieval and reconnect
failures retain them on `provider.operation.recovery_required` events. Normal
stream failures retain them on `model.error`, including durable SQLite readback.
Unsupported source discriminators are terminal; other protocol failures retain
the existing bounded unknown-outcome policy. Hosted-tool effect accounting
remains in force. Background recovery
of an already identified operation remains malformed/manual recovery and never
starts another provider request merely because decoding failed.
