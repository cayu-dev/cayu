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
