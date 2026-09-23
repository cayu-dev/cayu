# Recipient sessions

Recipient creation is a separate boundary from root participant-session
creation. `Application.create_recipient_session` accepts a typed
`RecipientSessionCreationRequest` and creates an inert, recipient-owned
session. It does not run a provider, tool, model, worker, or environment.

There are two explicit modes:

- `fresh` uses the bounded `RunRequest` supplied by the recipient. It cannot
  carry a selected context view.
- `fork` requires an adopted or transferred `ContextViewSelectionReceipt`.
  The selected manifest is read as historical material, and its exact detached
  messages are used. A busy source session is never reloaded and a failed fork
  is never silently changed into a fresh request.

The request has a caller-chosen creation key of at most 246 UTF-8 bytes for
exact replay, reserving ten bytes for the internal `recipient:` namespace.
Oversized keys are rejected when constructing the request. The key,
recipient, mode, selected-view identity, and accepted resource-transfer
receipts are included in the durable creation commitment. When transfers are
present, the caller must also provide the qualified local resource owner; the
application reads each exact accepted receipt back through that owner before
creating the child. Replaying the same request returns the same inert session;
changing those fields conflicts before a new session is created.
Each transfer must also carry the owner-issued preparation receipt for the exact
transfer command. Caller-shaped transfer or preparation data is rejected before
the child transaction.
Preparation responsibility must name the exact recipient participant and
incarnation; sharing an application owner with another participant is insufficient.

Historical local-file attachments are supported in retained `FilePart` and
`ToolResultPart.artifacts` material. The source publishes with exact acquisition
receipts in `ContextViewPublicationRequest.resource_receipts` and its qualified
`resource_owner`. Publication authenticates those receipts and holds their
retention fence through the manifest transaction. FORK then requires accepted
destination transfers for those same acquisition receipts, in addition to
coverage for any attachments in new input. Unsupported resource reference
families remain refused. Neither a manifest nor an attachment identifier grants
acquisition or release authority.
Publication also authenticates the source's static environment against its
completed execution profile, which binds the qualified local artifact store's
physical identity. An identically named artifact in another store is not the
same historical material. If the historical environment cannot be authenticated
after restart, new publication refuses; existing exact publication readback does
not launch or reconstruct the source environment.
Before creation, the resolved recipient environment must be static and expose
the same qualified physical local artifact store as the destination resource
owner. Referenced artifacts must have environment scope matching that environment.
Session-scoped files, mismatched stores/scopes, and opaque factory environments
are refused: retention transfer alone does not import a file into a new store
or change its access scope. No environment or provider is launched by this check.
Complete file references must match retained size and content type, and repeated
references must agree on resolution identity. Only per-read `source_artifact_id`
provenance may differ, matching ordinary runtime attachment resolution.

The returned `RecipientSessionCreationReceipt` binds the child session and
incarnation to the participant-session receipt and the recipient creation
tuple. Creation remains subject to the configured SessionStore's durable
participant-session support; stores that do not provide that boundary fail
closed.

Recipient creation is not activation or orchestration. A later, separately
authorized operation is required to execute the child.

`settle_recipient_resource_handoff` verifies the stored child receipt and its
session incarnation before asking each source owner to release its pin against
the destination's exact acceptance. The source and destination owners remain
responsible for their own resource journals; the application does not read or
modify those journals. Repeating this handoff uses the original transfer
identities, without creating another child or reacquiring resources.
If handoff is cancelled or loses its acknowledgement, the committed child and
accepted transfer remain indexed; repeating the same handoff is the recovery
operation and does not dispatch or repin anything.
Source cleanup uses the owner-internal accepted-handoff path, so expiration or
revocation of the original acquisition mandate cannot strand an already
accepted destination transfer.

## Durable creation admission and future targets

Recipient creation first retains its complete target as an unadmitted pending
intent in the SessionStore, then registers a collaboration permit before the
child transaction. Only exact authenticated registration upgrades the pending
record's `responsibility_registered` evidence. Preparing an intent alone never
permits child creation. Participant disablement and permit registration are ordered by the
collaboration owner: disablement first denies new creation; registration first
retains the admitted responsibility in the disablement frontier. Finishing that
already admitted inert creation does not reactivate the participant or grant
permission to execute the resulting session. Exact creation readback settles the
responsibility after acknowledgement loss.

The shared `SessionCreationTarget` binds the complete registered permit,
receiving owner, creation key, requested public ID (including explicit absence),
request and material commitments, and execution-identity commitment. The permit
includes the namespace generation, initiator, participant incarnation, lifecycle
and configuration revisions, admission generation, and source and settlement
operation identities. A target or receipt supplied by a caller is not authority
to register, exclude, or create a child.

The full pending target is durable before crossing the two-store boundary. If
registration loses its acknowledgement or the process stops between stores,
recovery can discover the complete target without the original request handle.
It retains uncertainty until exact reconciliation or an explicit whole-target
exclusion. Such exclusion also fences registration that completes late: neither
preparation nor authenticated registration may revive an excluded target. The
collaboration obligation can then settle against authenticated exclusion evidence
if its registration committed; absence of a registration acknowledgement never
means that no registration occurred.

Memory, SQLite, and PostgreSQL retain a separate creation-decision record with
`pending`, `created`, or `excluded` state. Creating the session atomically changes
its exact pending target to `created`, recording the store-minted session ID,
incarnation, and creation-receipt commitment. An explicit whole-creation exclusion
that wins first prevents this transition. If creation wins, exclusion returns
the created decision rather than relabeling it. Decisions survive live-session
deletion; another operation using the same public ID is a different target.
Memory retains these decisions for its store lifetime; SQLite and PostgreSQL
retain them across store reopening and process restart.

`read_session_creation_decision(expected)` distinguishes exact match, exact
not-found, conflict, and unavailable. It compares the complete expected target;
a missing live session is not exclusion evidence. Registered owners can recover
pending targets with `list_pending_session_creations(owner, cursor=..., limit=...)`;
pages are bounded to at most 64 decisions and have an opaque continuation cursor.
Here pending means pending recovery responsibility: the index includes terminal
`created` and `excluded` decisions until their collaboration settlement has been
positively acknowledged. `settlement_acknowledged` is separate from creation
state. A lost acknowledgement on either store leaves the full target discoverable;
recovery replays exact source settlement before acknowledging it in SessionStore.
Acknowledgement removes it from discovery but retains its exact terminal tombstone.
For an explicitly excluded target whose permit may never have registered,
recovery uses the collaboration owner's durable permit-exclusion operation.
It atomically fences a late registration or settles the registration winner;
an absent registration alone is never treated as settlement evidence.
Cancellation does not itself change a pending target into an exclusion. Recovery
must read the exact decision before retrying or explicitly excluding that target.

### Integration with delivery owners

These typed contracts are available from `cayu.sessions.creation_fence` for
registered receiving-owner integrations; they are deliberately not a top-level
application grant API. The registration/exclusion methods require private
runtime provenance following collaboration responsibility admission. An
unsupported SessionStore refuses those mutations and reports exact readback as
unavailable; wrappers must explicitly preserve the full target and the atomic
creation boundary instead of falling back to ordinary session creation.

Excluding one delivery obligation is **not** whole-creation cancellation. A
delivery owner retains its own exact obligation/attempt decision against the
future target; creation must preserve that separate record, and another delivery
or unrelated creation must remain unaffected. Peer append, withdrawal, and queue
decisions belong to the delivery owner, not recipient creation.

Both owners must share the creation transaction boundary. Memory uses the
SessionStore mutation lock and its retained target map. SQLite uses one write
transaction and the shared exact-target read helper. PostgreSQL takes the
canonical creation-operation advisory transaction lock before delivery-operation
and destination-row locks; the shared read helper runs on the same cursor.
The helpers live in `cayu.storage._creation_fence` and do not commit the caller's
transaction. A delivery integration must atomically inspect/bind the target and
record its own decision under this boundary; a read followed by an independently
committed append is insufficient. The target's public session ID alone never
authenticates a future recipient or a subsequently reused incarnation.
