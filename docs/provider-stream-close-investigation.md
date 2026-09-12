# Provider stream-close exception investigation

Provider stream-close diagnostics distinguish the local close result from
remote cancellation and settlement. A `close_exception` classified as
`RuntimeError` does not by itself establish generator reentrancy or a
remote-resource leak.

## Diagnostic additions

The version-1 cleanup envelope accepts optional, revalidated fields:

| Field | Retained evidence |
| --- | --- |
| `cleanup_exception_message` | Exact recognized content-free message, otherwise `redacted` |
| `cleanup_cause_type` | Closed classification of the direct cause, or implicit context if no explicit cause exists |
| `cleanup_cause_message` | Recognized message or `redacted` |
| `cleanup_local_stack` | JSON array of at most eight `[provider_module_filename, line_number]` locations from at most 32 traceback frames |

Messages are limited to an explicit vocabulary for generator read/close
reentrancy, closed event loops, and send-after-close. Unknown messages are never
truncated into public diagnostics. Exception formatting hooks are not called.
Stack capture retains only allowlisted Cayu provider filenames and line numbers,
with no absolute paths, source lines, function arguments, locals, or extension
filenames. Exception/cause objects and tracebacks are not retained by the new
fields. Remote cancellation and settlement remain `unknown`.

## Controls and findings

The current HTTP lifecycle controls exercise child deadlines during streaming,
single/multiple children, bounded/unbounded parents, parent cancellation,
semantic idle deadlines, and delayed/failed/noncooperative closure. The new
loopback test cancels the consumer again while its actual HTTP stream close is
held open. It proves one local close owner, continued retention until release,
caller cancellation, and eventual local/socket closure without upgrading remote
settlement. It also passes against unchanged main.

A deliberately reentrant Python async-generator close separately establishes the
actual `aclose(): asynchronous generator is already running` message and verifies
its sanitized classification. This is an injected diagnostic control, not a
reproduction of the original provider failure. Secret-bearing and hostile
exceptions, malformed diagnostic fields, SQLite recovery/export, and owned
cleanup handoffs are covered by the focused diagnostics and credential-boundary
tests. No cleanup scheduling or cancellation behavior is changed by this PR.

With this checkout's `src` and root first in `PYTHONPATH`, run:

```sh
python -m pytest -q tests/core/test_provider_cleanup_diagnostics.py tests/core/test_provider_credential_boundary.py tests/core/test_http_deadline_cleanup.py tests/core/test_http_semantic_cleanup.py
```

Local HTTP tests require socket permission. Passing synthetic controls does not
identify the cause of an unobserved provider failure or prove remote settlement.

## Interrupted read versus close failure

Joining an interrupted read can fail even when the subsequent `aclose()` returns
successfully. Cleanup diagnostics now retain `cleanup_failure_phase` (`read`,
`close`, or `read_and_close`) and separate `cleanup_read_exception_type/message`
and `cleanup_close_exception_type/message` fields. These use the same closed
exception classification and exact message allowlist as the original envelope.
A settled read-only failure has `cleanup_reason=read_exception`; two failures
have `cleanup_reason=read_and_close_exception`, even when the combined exception
is a group. The original exception propagation and cancellation behavior remain
unchanged.

`stream_close_state=confirmed` means the local close hook returned successfully,
including when joining the read failed. No close hook leaves closure unconfirmed;
a retained close still running remains pending. Remote cancellation and settlement
remain unknown. New fields are optional and revalidated on persistence/export,
so older version-1 records remain readable. Published pending snapshots do not
change when the retained close later finishes.

The regression controls distinguish failed reads with successful closure,
close-only failures, simultaneous failures, and suppressed read cancellation.
They also exercise repeated cancellation while closure is pending. These reproduce
the diagnostic ambiguity; they do not attribute the historical transport failure.

## Native child-deadline ownership controls

The current ownership chain separates protocol state from task ownership:

| Boundary | Owner and observation |
| --- | --- |
| Protocol lifecycle | `_stream_lifecycle.py` validates transitions; it does not own tasks |
| Pending provider read | The provider deadline controller and `_ProviderDeadlineAwaitOwnership` retain the dispatched read until its result is consumed |
| Ordered read/close | `aclosing_provider_stream` starts `_close_after_provider_read`; an interrupted read is joined before the close hook is invoked |
| Retained cleanup | A reserved cleanup owner or the existing deadline-read owner retains the close task and consumes its outcome; repeated caller cancellation does not transfer ownership to the caller |
| Native terminal publication | The model/session execution boundary snapshots sanitized cancellation diagnostics into the exact session epoch and terminal event |
| Tool stream closure | `ToolRoundRun.run` owns nested call and interruption streams with `aclosing`; the session boundary drains deadline cleanup without yielding new events to an expired caller |

`tests/faults/test_native_child_deadline_cleanup.py` uses parsed loopback HTTP
requests and drained first frames as dispatch barriers. It covers a child expiry
before semantic output and during output, with either one successful sibling or
two expiring children. It checks native deadline identity, the durable terminal,
SQLite replay without dispatch, local socket EOF before provider-wide shutdown,
and the loop exception handler after owned finalizers are drained.

The tool-round closure control also cancels again while the nested interruption
stream is finalizing. Cancellation remains authoritative, with exactly two
cancellation requests and observed nested teardown. Retained-read/close controls
inspect the saved terminal both before and after releasing local cleanup. The
immutable pending diagnostic is unchanged; it is an interruption-time snapshot,
not a live resource-state query or evidence of permanently leaked work.

These controls do not reproduce the historical `async generator ignored
GeneratorExit` / `no running event loop` chain. Resolving that incident still
requires one correlated local observation containing the session/epoch and tool
round identities, the owner driving each generator, the suspension point when
close/cancellation arrives, cancellation counts, and the finalizer outcome before
the owning loop exits. Keep these observations content-free and bounded; unrelated
pending-cleanup records or a successful synthetic close cannot supply this chain.
