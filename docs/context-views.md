# Participant sessions and historical context views

This capability has three distinct owner boundaries:

1. `CayuApp.create_participant_session()` creates an inert session bound to an
   authenticated, active `ParticipantRef`. It resolves and freezes the
   participant's lifecycle/configuration revisions and execution-profile
   identity, commits the binding and a typed receipt in the SessionStore, and
   performs no provider, model, tool, or environment work.
2. `CayuApp.execute_participant_session()` is the explicit authenticated
   activation entrance. It rechecks the exact participant/session incarnation
   and creation request, then reuses the ordinary session engine. Retries use
   the same execution key and durable invocation receipt; a changed key,
   participant, incarnation, or request is rejected. Creation never activates
   a session, and execution does not create recipient sessions or publish a
   view implicitly.
   Continuation uses `app.resume(request, context=administration_context)`:
   bound sessions require current participant administration and an active exact
   participant incarnation before recovery/admission. Omitting the context is
   rejected for bound sessions; ordinary unbound sessions are unchanged. Internal
   continuation callers without delegated participant authority also fail closed.
   The same `context=` argument is required for bound-session
   `resolve_tool_approval`, `resolve_user_input`, `compact_session`,
   `recover_tool_approval`, `recover_user_input`, `recover_tool_round`, and
   `reconcile_tool_effect`, and `resolve_provider_operation` calls. Manual recovery
   and receipt replay retain their
   exact durable operation identity; neither stored evidence nor reconstruction
   substitutes for the caller's current participant administration authority.
   These entrances check current authority before admitting mutations or
   executing tools, models, or a compactor. Recovery does not implicitly grant
   participant execution authority.
   A queued `on_idle` message may start a successor interaction only after
   rechecking the executing caller's current participant administration and
   lifecycle. Queue access alone and the predecessor's profile do not grant
   execution authority. If that check fails, the queued input remains retained;
   an authorized later continuation can consume it. Work already admitted in
   the current interaction is not retroactively cancelled by this check.
   Explicit provider-operation failure/fallback resolution carries the same
   context through its recovery continuation; a persisted disposition is not
   a substitute for current participant authority.
3. `CayuApp.publish_completed_context_view()` projects an authoritative,
   completed model-turn checkpoint into an immutable historical
   `ContextViewManifest`. `select_context_view()` atomically selects and pins
   one eligible view; ownership transitions explicitly adopt, transfer, or
   release that pin.

Publication captures the session, binding, completed checkpoint, completion
event, historical profile, and bounded transcript in one native store snapshot.
For a tool-bearing turn it also validates the exact ordinary tool-round or
approval/input closure receipt and its durable transcript/event evidence.
An unresolved or partially completed round is not publishable; a completed
round retains both its assistant calls and grouped tool results. The parent
may continue after capture without changing the retained historical boundary.

The configured SessionStore must advertise the native context-view and
participant-binding capability. Unsupported stores fail closed; callers must
not set a capability flag on a wrapper that does not implement the native
records and transaction checks.

## Creation and replay

```python
creation = ParticipantSessionCreationRequest(
    request=RunRequest(agent_name="reviewer", messages=[]),
    creation_key="stable-retry-key",
)
session, receipt = await app.create_participant_session(
    creation,
    participant=participant,
    context=administration_context,
)
```

The creation key is reused after cancellation, timeout, or a lost
acknowledgement. The exact request—including an explicitly requested session
ID or explicit absence, initial-input commitment, participant incarnation,
creator/configuration commitments, and resolved execution profile—is compared
before a replay is returned. A changed tuple is a conflict and cannot create a
second binding. The binding is historical identity evidence, not a reusable
execution permit; the explicit execution entrance and every later publication,
selection, and readback recheck current participant authority. A disabled or
replaced participant cannot activate the old binding.

## Publication and readback

Publication requires a source session bound to the authenticated participant,
an authoritative completed-turn checkpoint, a contiguous retained transcript
frontier, and a complete tool round. Partial rounds, unsupported resource
references, unavailable execution-profile ancestry, unsafe registered
extensions, and non-canonical projections are rejected before publication.
The manifest contains detached bounded JSON, commitments, historical profile and
causal-budget ancestry, extension presence/absence records, and an explicit
compaction relationship. It is historical data, never an executable session or
authority object.

Native stores capture the source binding, session incarnation, completed-turn
checkpoint, exact completion event and bounded transcript slice in one coherent
read snapshot. Historical profile attribution follows that completion's
fingerprint and durable invocation evidence, never an unrelated later profile
adoption. Extension callbacks run on the detached snapshot outside the store
transaction. The parent may advance after capture; final publication checks the
same live source incarnation under the deletion lock before inserting a new
manifest. Exact publication-key replay remains available without recapturing the
mutable source.

Creation retains a bounded historical definition projection containing the
original agent prompt, resolved initial system prompt, agent-definition
commitment and initial execution-profile commitment. Publication reads that
immutable binding, not today's agent registration. The manifest also records
the completed model event identity. Historical prompts are data, not current
instructions or permission grants; secret-bearing ancestry is rejected.
Prompt values are checked before JSON escaping, as well as at the final
serialized boundary, so quoted or multiline credentials cannot evade rejection.

Registered extension producers receive an immutable `ContextViewProjectionSource`
with the exact session incarnation, completion event, interaction/boundary,
transcript frontiers, retained messages, compaction relationship and historical
ancestry. A producer must deterministically resolve its historical state for
that boundary, never substitute current application state. It returns either
`None` (explicit absence) or `ContextViewExtensionProjection` containing canonical
JSON and the supplied source commitment. A mismatched commitment rejects the
entire publication; reconstruction checks every extension against the retained
manifest boundary. Each producer receives its own detached source object.

Extension JSON is a positive historical-record schema, not arbitrary application
JSON. A record may contain string fields `label`, `text`, `note`, `summary`,
`provider_name`, and `model`; a JSON scalar `value`; and an `items` array of
records using this same schema. Unknown fields and object-valued scalar fields
are rejected, including nested tools, provider stages, receipts and resources.
Historical text can describe an operation but never authorizes one. Publication
and durable reconstruction enforce the same grammar; absence remains explicit.

```python
def project_history(source: ContextViewProjectionSource):
    return ContextViewExtensionProjection(
        source_commitment=source.commitment,
        projection_json='{"note":"historical-only"}',
    )
```

```python
manifest = await app.publish_completed_context_view(
    ContextViewPublicationRequest(
        source_session_id=session.id,
        source_session_instance_id=session.instance_id,
        view_id="view-1",
        interaction_id="interaction-1",
        boundary_id="boundary-1",
        projection_schema="whole-turn.v1",
        publication_key="publication-retry-key",
    ),
    participant=participant,
    context=administration_context,
)
readback = await app.read_context_view(
    manifest.view_id,
    source_session_id=session.id,
    participant=participant,
    context=read_context,
)
assert readback.historical_only is True
```

Readback returns a bounded immutable manifest and optional lifecycle evidence;
it does not return a `Session`, provider registration, tool grant, workspace,
lease, ticket, reservation, or other executable object. Historical ownership
evidence remains inspectable to an already-authorized reader after disablement,
but disabled or replaced participants cannot acquire new views, transfer pins,
publish, or execute through the old receipt.

Retention transfer does not revoke already-authorized historical inspection.
Read authorization resolves exact participant-incarnation evidence independently
of the requested lifecycle-event page size. Persistent lifecycle readback checks
the event against its indexed identity, owner, state, pin and revision before
using it as evidence or exposing it to a reader.

## Selection and retention

`ContextViewSelectionRequest` supports exact, latest-eligible, and minimum
transcript-freshness selection. Projection schema and registered extension-set
commitment are part of eligibility.
The extension-set commitment identifies the sorted registered names and schema
versions only; content, presence/absence and source-boundary commitments remain
bound separately inside each manifest. An unchanged schema therefore permits
latest/minimum selection of later turns even when extension content changes.
Selection returns an immutable receipt and creates the initial retention pin;
it does not make a destination owner.
Adoption, transfer, and release use the exact selection key, view and pin
commitments, expected state/revision, and a stable operation key. Replaying an
identical operation returns the original receipt; changing any decision-bearing
field is a conflict. Expiry applies only to a still-selected pin and publishes
durable lifecycle evidence. Adopted or transferred pins are never silently
expired.

Release discharges existing retention rather than acquiring authority. An
authenticated administrator for the exact current owner can release after
disablement or retirement, and replay the same release receipt. Adoption and
transfer still require active participants; disabling an owner does not release
its pin implicitly.

All count, byte, and lifetime limits are finite and store-enforced. Source
closure and compaction are admitted by the SessionStore transaction owner and
fail closed when active pins—or orphaned active pin records—cannot be proven
safe. External artifact/workspace pinning is a separate resource-owner
contract; storing an ID alone never qualifies a resource reference.

Resource-free `whole-turn.v1` views with complete uncompacted material are retained
independently of the source transcript. Compaction verifies their commitments,
frontiers and retained message count under the mutation transaction and may
proceed with an active pin. Missing, corrupt or unsupported material remains
fenced. Compaction does not release the pin: deletion remains blocked until exact
release/transfer settlement. SQLite publication admits replay and quotas inside
one write transaction, including independent-process callers.

New selection validates the live source incarnation inside the pin transaction;
historical exact receipt replay does not create a new pin. SQL selection bounds
candidate identities before loading the selected manifest. Lifecycle accounting
reserves one terminal event for every unsettled pin within the per-view ceiling.
Adoption/transfer requires additional capacity; release/expiry consumes the
reserved capacity, so reaching the quota cannot strand settlement. PostgreSQL
serializes these bounded lifecycle transactions with a shared advisory lock,
held alongside the source-session lock through selection and lazy expiry.

Memory, SQLite, and PostgreSQL implementations use typed native records and
versioned compare-and-set transitions. SQLite/PostgreSQL reconstruction must
be qualified in a fresh process; PostgreSQL tests require either
`CAYU_TEST_POSTGRES_DSN` or the repository's Docker-backed fixture.
