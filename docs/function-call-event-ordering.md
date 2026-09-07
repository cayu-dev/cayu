# Function-call event ordering investigation

Issue #1498's historical
`function_call_arguments_done_arrived_before_output_item_added` diagnostic does
not establish provider, proxy, or Runtime ownership. It establishes only that
there was no pending function at the received output index. No incident wire
fixture is available here. This investigation uses credential-free synthetic
fixtures, not workload data or paid requests.

## Findings

The native adapter registers functions in `pending_function_calls` by
`output_index`. Argument deltas append to that entry. `arguments.done` validates
and removes the entry, emits a normalized tool-call event, and saves complete
function evidence in `fallback_output_items`. `output_item.done` checks that
evidence; the response terminal reconciles it. Runtime executes client tools
only after accepting the model attempt, not on the parser's intermediate
`TOOL_CALL` event.

Controls executed against base `03558cac` establish three separate findings:

- A second `arguments.done` is rejected with the same "before registration"
  reason as a true orphan because the first completion already removed the
  pending entry. The original incident could involve either state; the historical
  reason alone cannot distinguish them.
- An internally consistent completion/terminal pair can replace previously
  streamed arguments or the registered function name. The base parser emits the
  replacement call and accepts model completion. These are adapter validation
  defects, independent of the incident's missing registration.
- A function stream with omitted item `status` is rejected during reconciliation,
  despite the completion validator permitting that omission. The
  [official function-calling streaming example](https://developers.openai.com/api/docs/guides/function-calling#streaming)
  uses registration, deltas, argument completion, and output-item completion in
  that order, with item status omitted. This valid illustrative sequence exposes
  an adapter inconsistency; it does not explain the incident's different reason.

The patch distinguishes consumed registration from an orphan, rejects changes to
known function/response identities and accumulated arguments, rejects reuse of a
function's index for another item type, and checks that nonempty terminal output
retains previously completed calls. Reconciliation treats omitted status as
compatible with completed status only after the existing completion validator
has accepted the item. Explicit nonterminal statuses remain invalid.

## Synthetic transitions

`tests/core/test_openai_function_ordering.py` runs direct native-parser controls
and byte-chunked SSE through `HttpxOpenAITransport`, Runtime dispatch/retries,
and SQLite durable event readback. The SSE framing helper is shared with #1495.
Let `A(i)` register a function, `d(i)` append arguments, `G(i)` complete
arguments, `D(i)` complete the output item, and `T` complete the response.

| Sequence or mutation | Verified behavior after the patch |
| --- | --- |
| A(0), d(0), G(0), D(0), T | Pending, accumulating, complete evidence, matching item, matching terminal; exactly one call executes |
| Same sequence with omitted item status | Accepted; status omission cannot contradict synthesized completion |
| A(0), A(1), d(1), d(0), G(1), G(0), D(1), D(0), T | Independent indices and identities; each call executes once |
| G(0), or A(0), G(1) | Existing orphan reason; no complete call can execute |
| A(0), G(0), G(0) | `function_call_arguments_done_was_repeated` |
| A(0), G(0), d(0) | `function_call_arguments_delta_arrived_after_arguments_done` |
| A(0), G(0), A(0) | Repeated registration; completed evidence cannot be overwritten |
| Changed item/response/call identity or name | Bounded identity diagnostic; no tool execution |
| Same item/call identity at a different index | `function_call_identity_was_reused` |
| Function index reused by a message, or the reverse | `function_call_output_index_type_mismatch` |
| G(0) disagrees with nonempty accumulated arguments | `function_call_arguments_done_conflicts_with_streamed_arguments` |
| D(0) or T contradicts completed arguments/identity | Existing item/terminal conflict diagnostic; no tool execution |
| Nonempty terminal output omits a completed call | `terminal_response_omitted_completed_function_call_evidence` |
| A(0), d(0), T or EOF | Unfinished-call or early-stream-end rejection; partial arguments do not execute |
| Empty JSON object arguments | Accepted as `{}` |
| Empty string arguments with no accumulated object | Rejected as missing arguments; no object is invented |
| No deltas, or an empty delta, then complete JSON in G(0) | Accepted from registered identity and complete arguments |
| Identical repeated D(0) | Existing idempotent reconciliation; no second tool call |
| Empty/omitted terminal output after complete streamed evidence | Existing fallback behavior retained |
| T followed by more transport input | Accepted completed response closes the stream without consuming the tail |
| Incomplete terminal followed by G(0) | Direct parser rejects post-terminal mutation; native Runtime does not execute the partial call |
| Abandoned partial attempt, then valid retry with the same IDs | Fresh registration/trace state; one execution from the accepted attempt |

Missing optional event identities retain existing compatibility; supplied item
IDs cannot be blank/null substitutes for a known identity. Response identity
comparison uses the identity established by `response.created`. Fully specified
terminal-only function outputs remain supported by existing response
reconciliation. An orphan argument event is never promoted into such authority.

## Structural evidence and remaining ownership work

Function protocol errors now reuse the bounded trace format from
[the hosted-search investigation](hosted-search-event-ordering.md). The scalar
fields are `provider_protocol_stream_boundary`,
`provider_protocol_stream_trace`, and
`provider_protocol_stream_trace_truncated`. Each of at most 16 rows contains:

1. Received ordinal, saturated at 1,000,000, local to the parser invocation.
2. Allowlisted event type (including function delta/completion); otherwise `other`.
3. Output index, or -1 if absent, invalid, or over 1,000,000.
4. Function state immediately before processing: `absent`, `pending`, or
   `completed`. Here `completed` means validated arguments are in the existing
   fallback map; it does **not** prove that `output_item.done` has arrived.
5. Item identity relation to registration/evidence at that index.
6. Response identity relation to the registered response.

Relations are `missing`, `invalid`, `unregistered`, `matches`, or `differs`.
Serialization is under 4 KiB and revalidates all values before durable/public
projection. The trace retains no raw identities, hashes, names, arguments,
headers, endpoints, or exception text. It adds constant storage and uses the
existing completed-item map. A truncated window is explicitly incomplete
history, not proof that registration never arrived. Hosted-search errors retain
their search trace when both tool types occur.

The native SSE regressions verify distinct durable model-attempt IDs, shared
step identity, attempts 1/2, exact failure reasons, public/durable trace equality,
and exhaustion of the existing two unknown-attempt allowance. Registered test
tools record zero executions on rejected model attempts. Retry caps and retry
classification are unchanged. There is no new deferred assembly or buffering.

Fault ownership for the original incident remains unproven. A new occurrence
needs bounded structural order/index/identity-relation evidence at upstream
output, any proxy ingress/egress, and native adapter ingress for the same
externally correlated attempt. Locate where the sequence first diverges. The
adapter trace alone cannot distinguish upstream ordering from a proxy dropping
an event, nor recover registration outside its retained window. No raw streams,
argument contents, or workload identifiers are needed in the issue or PR.
