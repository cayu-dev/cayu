# Provider stream-close exception investigation

Tracking: #1592. The original evidence establishes `close_exception`,
`RuntimeError`, and unconfirmed local closure. It does not establish generator
reentrancy or a remote-resource leak. Earlier child-deadline work (#1468/#1471)
and cleanup-cancelled work (#1487/#1489) remain separate.

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

Baseline: Runtime `eae5abdfd6ec5d166ef011b655cad2ae7adaec80`, version 0.4.0,
including lifecycle centralization `8f3deea64766458223bccbfee294a5a3fe01986d`.
Environment: macOS 15.7.7 arm64, Python 3.14.3, httpx 0.28.1, httpcore 1.0.9,
anyio 4.13.0, pytest 9.0.3. The PR records the exact tested implementation commit.
No live provider/model version or OpenAI SDK was exercised.

With this checkout's `src` and root first in `PYTHONPATH`, run:

```sh
python -m pytest -q tests/core/test_provider_cleanup_diagnostics.py tests/core/test_provider_credential_boundary.py tests/core/test_http_deadline_cleanup.py tests/core/test_http_semantic_cleanup.py
```

Local HTTP tests require socket permission. The historical exception message,
stack, and transport context are still unavailable, so its cause remains
unattributed and #1592 is not closed by passing controls. No paid probe, campaign
rerun, Linux/Windows test, full qualification, or remote-settlement proof was
performed.
