# OpenAI terminal HTTP ownership

The Responses parser acknowledges a terminal event only after validating its
identity, output assembly, usage, and terminal classification. The shared HTTP
transport then drains readily available trailing data, across chunk boundaries,
with a fixed **50 ms grace** starting at the next byte read. The grace is not
refreshed by heartbeats, empty chunks, or additional data. Response-context ownership closes the
HTTP body; close failures and caller cancellation retain their existing behavior.

This avoids waiting for transport EOF after a complete protocol response. The
OpenAI API documents [`response.completed`](https://developers.openai.com/api/reference/resources/responses/streaming-events#response.completed)
as a completed model response. A server or intermediary keeping the connection
open must not turn that validated completion into a semantic-idle timeout.

The acknowledgment is an internal attribute on the exact Cayu-owned decoded SSE
envelope, not a JSON field. It is independent of runtime deadline context, so it
also works with direct parser/transport use. Transparent transport wrappers must
preserve the yielded envelope to preserve this acknowledgment, just as they must
preserve trusted response metadata. Copied dictionaries and custom iterators keep
their existing EOF behavior. Raw transport consumers do not implicitly accept a
terminal merely because they received a matching event type.

## What is and is not validated

- Foreground API and subscription streams accept validated `response.completed`
  and `response.incomplete`; an incomplete result remains incomplete.
- Background/reconnect parsing also acknowledges validated failed, cancelled,
  and expired terminal states. These remain failures, not successful completions.
- Readily available trailing events still pass through the existing strict parser,
  including events in separate HTTP chunks.
  Conflicting identities, repeated terminals, malformed SSE and extra output fail.
- The drain does not wait indefinitely for EOF. A violation sent only after the
  grace expires is outside the consumed response; this is not a promise to inspect
  arbitrary data sent after completion. Normal EOF returns immediately.
- A real read error is not treated as graceful expiry. A transport that suppresses
  cancellation does not manufacture success: the existing stream deadline and
  retained read/close ownership still apply. The 50 ms grace is not a replacement
  for those outer safety bounds.
- When the grace cancels a pending read, EOF is accepted only if that read ends
  in cancellation with no authenticated cleanup-failure evidence. A transport's own timeout
  remains an error even if the drain timer has expired. A failed nested close
  remains a non-retryable cleanup error; genuine caller cancellation retains its
  cancellation diagnostics instead of becoming EOF.
- The response is still closed. This is not proof of remote cancellation or
  settlement and does not change retry authority, effect accounting or deadlines.
- Other provider protocols and custom event iterators are unchanged.

## Regression controls

`tests/core/test_openai_terminal_http.py` covers API/subscription completion,
incomplete results, split frames, buffered conflicts, invalid and forged terminal
claims, reconnect terminal states, close failures, caller cancellation, and a
negative control disabling acknowledgment to reproduce the old terminal-then-idle
failure. Native runtime controls use HTTP mocks and a loopback socket that never
sends EOF. They execute one synthetic tool exactly once, finish the next model
turn, observe local socket closure before provider-wide shutdown, and reopen the
SQLite event history without redispatch.

Negative runtime controls inject a read timeout or failed nested close during
drain cancellation. They require session failure, zero proposed-tool executions,
no model retry, preserved usage, and the same event history after SQLite reopen.
Paired parser controls cover clean expiry, transport timeouts with and without an
explicit cancellation cause, nested cleanup failure, and real caller cancellation.

These controls reproduce and repair a possible failure mechanism. They do not
identify which remote server or intermediary withheld EOF in a historical trace.
