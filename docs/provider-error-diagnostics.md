# Provider-error diagnostics

Normal Cayu events deliberately omit raw provider error bodies and arbitrary
request IDs. Such fields can echo credentials or customer input. Error identity,
status and retry classification remain available where safely recognized.
An error event inside an HTTP-200 SSE response is still a provider failure.

For an authorized investigation, opt into a separate private capture scope:

```python
from cayu.providers import capture_provider_errors
from cayu.vaults import SecretRedactor

# write_private_record is application-owned: restricted storage, no public logs.
# Register all workload secrets that could be echoed by the provider.
with capture_provider_errors(
    write_private_record,
    redactor=SecretRedactor(workload_secret_values),
) as capture:
    async for event in app.run(request):
        handle_event(event)

if capture.sink_failures or capture.records_dropped:
    report_incomplete_diagnostics()  # Do not claim that no provider error occurred.
```

The synchronous sink receives detached JSON-compatible records only on errors.
It must complete promptly. Cayu never sends these records to a model, appends
them to a session, or includes them in support bundles. Applications own storage,
access controls, cleanup, and retention. Use an owner-only directory/files (for
example POSIX `0700`/`0600`), or equivalently restricted storage. Capture is off
by default. An empty `SecretRedactor` is an explicit choice for workloads without
registered secrets; it is not automatic anonymization of customer data.

## What is preserved

Bundled HTTP transports capture OpenAI-shaped HTTP error bodies and SSE `error`
and `response.failed` events before public sanitization. The record contains:

- A local UUID `capture_id`, optionally supplied by the caller to correlate a
  private prepared-request artifact. It is **not** the upstream request ID and
  is not sent to the provider. Prefer one scope per prepared request when exact
  request-level correlation is required.
- The actual HTTP status separately from an explicit error-body status.
- All valid explicit body statuses in `error_status_codes`, keyed by their fixed
  paths: `status_code`, `error.status_code`, `response.status_code`, and
  `response.error.status_code`. Only paths belonging to the observed envelope
  are included. `error_status_conflict` states whether they disagree;
  `error_status_code` is present only when at least one body status exists and
  all agree. The transport HTTP status is not part of this body-status comparison.
- The upstream `x-request-id`, or the body's request ID when no header exists,
  with its source stated. No arbitrary response headers are retained.
- Bounded message, parameter, error type and code, plus a state for each field:
  present, absent, invalid, redacted, truncated, or omitted because oversized.
- A fixed transport exception category when no API error body is available.
  Raw transport exception text, URLs, headers and tracebacks are not captured.
  Mid-stream failures also retain the already-received HTTP status and redacted
  request ID, without reading more of the response. Pre-header failures do not
  invent either value. Both report `body_state="unavailable"`.

Request headers are used only as additional redaction secrets, including the
bare authorization credential. No request bodies or successful output are
captured. Redaction runs before truncation. Messages are bounded to 4 KiB,
parameters/request IDs to 256 bytes, and type/code to 128 bytes. Fields above
64 Ki characters are omitted whole; malformed Unicode fields are explicitly
omitted instead of being repaired after redaction. HTTP body decoding respects a 64 KiB bound.
There are at most 32 records per scope. Sink failures and dropped records are
counted without exposing sink exception text or replacing provider exceptions.

Custom transports can call `capture.record_error(...)` explicitly. They must
provide request credentials for redaction and accurate source metadata. Other
provider-specific streaming error shapes are not automatically interpreted.
For nested envelopes, pass `status_fields` with explicit status values at the
fixed paths above. Only integers from 100 through 599 are retained; booleans,
strings and arbitrary keys are omitted. Without `status_fields`, the supplied
error's `status_code` is recorded at that path. These fields are private evidence,
not classification or retry inputs.

Scopes are task-local and nestable. Inherited child work must finish before its
scope exits. Do not leave an async generator unclosed across scope boundaries.
No network retry, additional request, header injection, or recovery permission
is introduced by capture. Classification continues to use the provider's typed
status/identity, not diagnostic text.

## Interpreting compaction failures

A compactor's `model.completed` record accounts for an attempted model call; it
does not alone mean success. Inspect `purpose`, `compaction_outcome`, error
fields, and `context.compaction.failed`. A rejected compactor call must not be
counted as a successful actor response. Private capture can retain the provider
explanation while ordinary durable events keep the privacy-safe failure state.

Capture records what the provider actually supplies. An absent explanation or
request ID remains explicitly absent; it cannot reconstruct discarded historical
details or reveal the provider's internal root cause.

## Durable rejection diagnostics

Bundled OpenAI (including subscription), Chat Completions, Anthropic, and Vertex
HTTP failures now carry a bounded safe projection in normal `model.error`
events. Their recognized streaming error envelopes use the same projection.
For example, a synthetic HTTP 400 with `Unsupported parameter: 'temperature'.`
retains:

```json
{
  "provider_rejection_reason": "unsupported_parameter",
  "provider_rejection_parameter": "temperature",
  "provider_rejection_explanation": "Remove this parameter; the endpoint or model does not support it."
}
```

These are untrusted diagnostic claims. Runtime generates the explanatory sentence
from a finite vocabulary; it never copies provider prose. A recognized flat
`code` and `param` pair can supply `unsupported_parameter`,
`missing_required_parameter`, `invalid_value`, or `positive_integer_required`.
Without such a pair, only complete, exact messages match: `Unsupported parameter:
'<param>'.`, `Missing required parameter: '<param>'.`, `<param>: Input should be
a valid integer`, and `<param>: must be a positive integer`. There is no substring
matching, arbitrary numeric constraint retention, or extraction of echoed values.

Parameters must exactly match one of: `temperature`, `top_p`, `max_tokens`,
`max_output_tokens`, `max_completion_tokens`, `messages`, `input`, `model`, `tools`,
`tool_choice`, `response_format`, `stream`, `stop`, `seed`, `reasoning_effort`,
`previous_response_id`, or `metadata`. Unknown codes and parameter paths are not
copied. Known credentials overlapping these labels suppress the diagnostic.
This fixed-vocabulary policy also works before credential resolution; it does not
assume replacement of known API keys can sanitize arbitrary customer data.

`provider_rejection_unavailable_reason` distinguishes `absent_body` and
`absent_details` from `body_unavailable` (including unread, encoded, stalled, or
transport-omitted bodies), `body_too_large`, `non_json_body`, `malformed_body`,
`unrecognized_details`, `unsafe_parameter`, and `credential_overlap`. HTTP
projection decodes at most 64 KiB; message matching examines at most 512
characters. Nested arbitrary details are ignored. Output has at most five fixed
fields and no unbounded values. Malformed JSON and nesting failures leave HTTP
status and retry classification intact.

`provider_rejection_request_id_state` is `absent`, `omitted_untrusted`, or
`unavailable`. No raw, truncated, or hashed upstream identifier is made durable:
its format cannot prove it is not a credential or customer identifier. Use the
existing session ID, model attempt ID, and workflow child failure's session and
terminal-event references to locate the durable failure. For upstream/gateway
correlation, use the explicitly authorized private capture above, with one scope
per request and an application-owned association to these local identities.

Inspect retained details with:

```sh
cayu session events SESSION_ID --sqlite data/cayu.db --include-payload 10000
```

The CLI omits payloads unless requested. These diagnostic fields do not change
HTTP status, retry disposition, recovery authority, or dispatch count. A
non-retryable 400 still stops after one dispatch. They cannot recover explanations
from historical events that were already sanitized, and a recognized claim does
not establish whether the provider or a gateway caused the rejection.
