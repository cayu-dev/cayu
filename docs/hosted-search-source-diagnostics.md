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

Issue #1486 records the exact discriminator `api`, but no complete observed
source object or emitting upstream boundary. Its URL-bearing example is a
synthetic rejection fixture, not evidence that an API source has a URL.
The [web-search guide](https://developers.openai.com/api/docs/guides/tools-web-search#sources)
mentions third-party feeds in sources, but does not establish their wire schema
or tie them to the observed discriminator. The remaining fields and the origin
of this variant are unresolved; no live provider probe was run.

Runtime exposes `OpenAIUnsupportedSearchSourceError` (a subclass of
`OpenAIProtocolError`) for unsupported discriminators, including explicit null
and non-string values. Streaming translates this into
`provider_error_type=unsupported_capability`, `retryable=false`, while preserving
the existing bounded protocol diagnostic. The attempt stops with
`retry_disposition=explicit_nonretryable`; a second hosted search cannot repair
an unsupported decoding contract. URL and omitted-type behavior is unchanged.

Supporting a non-URL variant still requires a bounded sanitized **observed**
fixture from the emitting boundary: field names/types and safe enum values,
with credentials, text, prompts, service identity, and full responses removed.
Qualify any upstream normalization and Runtime decoding with that same fixture.
Until then, do not coerce sources into URLs, discard them, or claim complete
evidence. This change supplies the terminal capability outcome; it does not
resolve the upstream schema investigation or declare `api` supported.

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
Unsupported source discriminators are terminal; other protocol failures retain
the existing bounded unknown-outcome policy. Hosted-tool effect accounting
remains in force. Background recovery
of an already identified operation remains malformed/manual recovery and never
starts another provider request merely because decoding failed.
