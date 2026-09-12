# Private provider-error diagnostics

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
