# Monitor invocation health and command outcomes

A `tool.call.completed` event says the tool fulfilled its invocation contract. It
can report an exit code of 1 without failing the invocation. Expected probes such
as `grep` with no matches are normal uses of this contract. Timeout and cancellation
retain the tool's existing lifecycle semantics; reporting introduces no retries,
score changes, or command-policy changes.

Eval trials expose `operation_outcomes` in both normal and portable corpus JSON.
Workflow trials aggregate their retained descendants. Normal and corpus HTML
reports display the separate counts. Older results without this optional field
render as `not_reported`; zero observations are not proof of successful work.

For live monitoring or exported trajectory inspection:

```python
from cayu.evals.operation_outcomes import (
    summarize_operation_outcomes,
    trajectory_operation_outcomes,
)

observations = summarize_operation_outcomes(events, evidence_complete=False)
print(observations.counts.invocation_failed)
print(observations.counts.command_nonzero_exit)
print(observations.counts.command_timed_out)
print(observations.counts.command_cancelled)

# Includes the retained child tree, and can be used with load_trajectory(...).
observations = trajectory_operation_outcomes(trajectory)
```

`evidence_state=observed` means counts describe the captured evidence, not that work
succeeded. `incomplete` means capture coverage or legacy execution pairing is
incomplete; counts are observations, not complete totals. `not_reported` means no
summary source was available. Missing exit codes and uncompleted starts have
`not_reported` outcomes. Missing timeout/cancellation flags remain null in evidence
rows; an observed zero exit does not fill those flags with false.

Command count categories are exclusive: timeout, cancellation, then zero/nonzero
exit, otherwise not reported. Each row retains the reported exit code and flags,
including a nonzero code accompanying a timeout. The runner contract has no separate
signal field; negative exit codes are retained without guessing a platform-specific
signal. Invocation cancellation uses Runtime's explicit interrupted terminal marker.

Runner `execution_id` is shared by the started and completed events for each actual
execution. Separate executions (including retries inside a tool) count separately.
The summary prefers runner evidence over the enclosing tool result, correlating by
session, tool round, invocation idempotency key, and tool-call ID. Repeated event IDs
and repeated completion observations for the same execution do not inflate counts.
Legacy runner completions without execution IDs use event identity; legacy start
pairing cannot establish full coverage. Replaying a retained trajectory recomputes
observations without redispatching commands.

The original event/result is unchanged. Each summary retains up to 256 content-free
observation references with session/event/call/execution IDs. Counts cover all supplied
evidence; `omitted_evidence_count` describes references omitted from the bounded list.
Retrieve source events from the session store or retain the trajectory with
`retain_trajectory=True` and `write_trajectory_json`. IDs longer than 1024 characters
are represented by a `sha256:` fingerprint (optional call/execution linkage is omitted
when oversized); match the fingerprint against the original ID. HTML displays counts;
JSON carries the references. These records are diagnostics, not correctness assertions.

## HTTP outcomes

Runtime does not parse stdout for HTTP status. An exit-zero script printing `403`
is still an observed zero exit, with HTTP `not_reported`. It could have printed a
fixture, handled an expected response, or made several requests. Command runners
cannot determine the script's application outcome.

Built-in web tools' validated `WebAccessEvidence` (`structured.access`) and the
existing `error="http_status", status_code=...` result contract supply status
evidence where available. Missing HTTP evidence, including successful tools that
do not publish a response status, stays `not_reported`.

A custom tool can explicitly supply one typed HTTP response outcome:

```python
from cayu import ToolResult
from cayu.evals.operation_outcomes import HttpOperationOutcomeV1

result = ToolResult(
    content="Request finished",
    structured={
        "operation_outcome": HttpOperationOutcomeV1(status_code=403).model_dump(mode="json"),
    },
)
```

This opt-in contract counts statuses 400–599 as HTTP errors and statuses below 400
as responses, without claiming application success. Invalid typed evidence stays
unknown. The protocol field and strict status range prevent arbitrary result fields
or output text from being interpreted as HTTP evidence. Tools handling multiple
requests should publish separate invocations if each response needs a separate
count; this contract describes one supplied outcome per invocation.
