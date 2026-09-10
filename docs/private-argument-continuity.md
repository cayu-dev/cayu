# Private tool-argument continuity

Tools can retain redacted **original model-authored inputs** for subsequent model
requests independently of whether those inputs may appear in terminal events and
the transcript. `RememberKnowledgeTool` enables this behavior. Other tools opt in
by overriding `Tool.retain_arguments_for_model` to return `True`.

The existing `_publish_arguments` policy still controls audit/transcript
publication. Enabling retention does not enable publication. Tools which already
publish their arguments do not need a private copy.

```python
class PrivateInputTool(Tool):
    @property
    def _publish_arguments(self) -> bool:
        return False

    @property
    def retain_arguments_for_model(self) -> bool:
        return True
```

## What is retained

Only original inputs, after secret redaction. Hook-enriched effective arguments,
resolved credentials and privileged store metadata are not substituted for those
inputs. Attempted inputs do not imply tool success: denied, failed and pending
knowledge writes retain their authentic results and lifecycle status. No knowledge
read, embedding call or activation is performed to reconstruct a call.

Retention is explicitly **bounded best effort**:

- At most 4 KiB of canonical argument JSON per call, 8 KiB per round and 32 calls
  per round. Oversized calls are omitted, not truncated into misleading arguments.
- At most 16 retained rounds, each including at most 16 KiB of association and
  integrity data: at most 256 KiB plus the small JSON container per session.
- Oldest records are evicted when a retained round is committed. Oversized
  association records are omitted. Cache admission never changes a successful
  external tool effect into an argument-size failure.
- Targeted-tool gateway/grant envelopes are currently withheld. Their resolved
  inner execution arguments are not the original model envelope. Direct ordinary
  calls are supported; this feature does not weaken targeted-tool authority.

## Durability and privacy boundaries

The sealed argument batch is committed in the same native transaction as the
assistant/result transcript pair and pending-round deletion. Private state uses a
reserved session-operation key; ordinary operation reads, writes and initialization
cannot access that namespace. There is no second transcript or historical backfill.

In-memory, SQLite and PostgreSQL use the same bounds and validation. A replay after
lost acknowledgement returns the existing publication rather than appending another
private record. Invalid sealed evidence rolls back without publishing private state.
Records bind the session incarnation, publication identity/request digest, exact
public tool-call representation, execution profile and knowledge-access scope.
Private random nonces keep public publication digests from being useful as
dictionary checks against low-entropy omitted text.

Same-profile resume can reconstruct eligible arguments after reopening the store.
A changed profile or knowledge-access scope withholds them. Forks do not inherit
private operation records. Session deletion removes them with the other operation
records. Unsupported custom stores fail closed before tool dispatch when a call
has a retained candidate; calls excluded from retention do not require this
capability. A custom store must implement this atomic contract before declaring
`supports_private_argument_continuity = True`.

Context policies and compactors receive the public projection, not these private
arguments. After selection, the runtime overlays only still-present, exactly
matching calls. It does not resurrect compacted-away calls or inject private text
into persisted summaries. Fully materialized requests pass through the existing
request-footprint, context-pressure and dispatch-admission pipeline. The bounded
overflow recovery path applies the same materialization rules.

For OpenAI native replay, the model-only projection also restores the matching
omitted function-call arguments in provider state. Opaque reasoning items and
provider item identities remain unchanged; the persisted audit retains its empty
arguments. Matching requires the same selected message, call ID and tool name.

There are no extra knowledge queries. Disabled retention returns immediately;
enabled retention with no eligible calls performs no private-store read. Eligible
calls use one bounded read, not one lookup per call. Unchanged context messages
are not deep-copied again, and the write path never copies a growing session-wide
operation-record map just to append private arguments.

This is an audience boundary, **not database encryption**. Database administrators
and explicitly authorized model-request processors can access this material. A
model may repeat it in a later answer. Applications must treat their providers and
request-processing extensions as authorized processors, not public audit sinks.
Pending recovery checkpoints already contain private execution arguments and now
also contain the provisional retained candidate. Full recovery snapshots of a
pending session are privileged state, not public audit exports. The completed
private cache is excluded from session exports; public events and transcript
projections never gain these retained arguments.

## Verification and scope of the claim

`tests/core/test_tool_argument_continuity.py` exercises independent policies,
native publication fault injection, store reopening, fork isolation, deletion,
export exclusion, bounds, authority changes, late secret discovery and context
selection. Existing secret, publication, profile and compaction suites remain
regression gates. `benchmarks/argument_continuity.py` measures overlay overhead
without provider calls; it is not an evaluation of model reasoning or memory quality.

Preserving readable arguments is a deterministic runtime improvement. It does
**not** prove that argument omission caused repeated remembering, or that retention
reduces repetition or improves task success. Those are separate controlled model
evaluation questions.
