# Policy-controlled human review

`CayuApp.inspect_human_review(session_id, context=HumanReviewContext(...))` reads
one authoritative pending input or approval. Configure `human_review_policy` on
`CayuApp` to enable it. Without a policy, access is denied. Inspection does not
claim the pause, invoke a provider or tool, or change the checkpoint.

The policy is trusted application code. Its `authorize` method must independently
check the recipient, purpose, exact session and tenant for `inspect` and `decide`.
Authentication alone does not authorize disclosure. Session metadata is an input
to application authorization, not an automatically trusted tenant partition.
An inspection grant cannot answer, approve or recover an interaction. Direct SDK
callers are inside the trusted application boundary: they supply verified context
and own access to the app and stores. Do not expose those objects to clients.

`project` receives a detached typed `HumanReviewSource`, including every pending
call, its private arguments, and the dynamic question/options when applicable.
It returns `HumanReviewDisclosure` with a narrow selection of display fields.
Unknown and dynamic secret scopes withhold those fields by default. To disclose
in such a scope, the projector must explicitly return
`sensitive_content="application_attested"` after independently validating its
selected output against an application-owned disclosure rule. An argument dump
or a scan against currently known secrets cannot establish that attestation:
a vault, proxy, tool, or later sibling may discover a secret after the pause.
Known secrets are additionally rejected in selected labels and values.

Use `permitted`, `redacted`, and `unavailable` as distinct outcomes. Withheld
content is never an empty valid proposal. The view supplies bounded guidance and
whole-round scope: `eligible`, `denied`, or `withheld` for each call. Eligibility
is conditional on the existing execution admission checks. Ambiguous recovery
gates are unavailable for execution and explicitly identify blocked recovery.
No view promises that a gating call is the only call that can proceed.

Display fields are untrusted text, limited to 32 fields, 4,096 characters per
field, and 16 KiB before escaping. The response declares `html_escaped_text`:
render these fields as escaped text, never as Markdown, a template, a command,
or trusted operator instructions. Escape application-added labels and other UI
content too. Oversized or invalid projections yield typed unavailable outcomes;
policy errors do not echo their private inputs.

Pass `view.reference` as `review_reference` on `UserInputResponse` or
`ToolApprovalRequest`, using the view's exact session/action/round/call identities.
With a configured policy, normal resolution requires a reference. The runtime
checks separate decision permission before continuation, then compares the view
inside the existing atomic claim. The opaque HMAC binds the complete pending
execution content, session incarnation, policy version, recipient/purpose and
selected display material. Changed content, expiry, supersession, another winner,
and policy/key changes cannot authorize an action using an old view. Unavailable
or redacted approval views can support denial, not approval. Existing exact
resolution receipts implement retries, including lost acknowledgements; the
reference participates in the request digest, not a second decision ledger.

After a decision has been accepted, publication progress can change the pending
checkpoint. Inspection then returns an `unavailable` recovery view with a fresh
reference and fixed recovery guidance. This reference binds the current pending
content and resolution intent; it cannot replace the accepted answer or approval
to authorize more execution. Pass it as `review_reference` to
`ToolApprovalRecoveryRequest` when supplying an externally verified tool outcome.
Recovery cannot run unstarted approval siblings: retry the exact original
`ToolApprovalRequest` to continue them under its already accepted authority.

For `UserInputRecoveryRequest`, pass the fresh recovery view as `review_reference`
and preserve the accepted answer's original reference as `answer_review_reference`.
The latter participates only in the original answer identity, along with its
answer, metadata and other existing fields; it grants no recovery permission.
Existing answer metadata and `resolved_by` must still match the accepted answer.
The current recovery recipient must independently have `decide` permission.
Accepted answer/approval retries match their durable request digests inside the
atomic claim and recheck decision permission instead of reapproving changed
publication progress. The accepted approval intent also binds the immutable
approved content, so changed tool arguments still fail closed. Accepted recovery
retries similarly retain their exact request identity. Keep the original requests
for these retries.

Use a private random `binding_key` of at least 32 bytes, shared by authorized
workers. Keep it in application configuration, never session metadata. Keep the
policy deterministic and side-effect free and change `version` for every change
to authorization, projection or sensitive-content rules. A new process with the
same configuration reconstructs the same view from SQLite or PostgreSQL. Lost
keys invalidate outstanding views; re-inspect rather than trying to recover a
key from stored arguments. Inconsistent or completed pauses return unavailable
without a reference. An outstanding resolution claim yields only the recovery
view described above; it does not bypass an active owner's run fence.
Inspection is a snapshot; its atomic decision binding handles races
after the response has been returned.

The protected server path is `GET /api/sessions/{session_id}/human-review?purpose=...`.
It always requires authentication, including on a local-development server. It
constructs recipient/tenant from the verified `AuthContext` and then applies the
application policy. It sets `Cache-Control: no-store`. Resolution and recovery
routes require matching authenticated provenance and independent decision
permission when the policy is configured. Existing deployment authentication and
store isolation still apply. A tenant string is not a substitute for scoped
storage or application authorization.

Review responses have **no ordinary publication contract**. Do not place them in
logs, events, hooks, transcripts, provider requests, exports, generic caches or
analytics. `HumanReviewPolicy.audit` receives only bounded action/status strings;
use it for access auditing without questions, arguments or private fingerprints.
It must be fast and safe inside an atomic claim. Ordinary runtime projections
continue to quarantine pre-execution arguments.

## Small application policy

```python
from cayu import (
    HumanReviewDisclosure, HumanReviewField, HumanReviewPolicy,
)

class DeliveryReview(HumanReviewPolicy):
    version = "delivery-v1"

    def __init__(self, binding_key, session_owner):
        self._binding_key = binding_key
        self.session_owner = session_owner  # trusted application-owned mapping

    @property
    def binding_key(self):
        return self._binding_key

    def authorize(self, context, *, session_id, session_metadata, action):
        owner = self.session_owner.get(session_id)
        return (
            owner == (context.tenant, context.recipient)
            and context.purpose == "delivery"
            and action in {"inspect", "decide"}
        )

    def project(self, context, source):
        if source.kind == "user_input":
            permitted = {
                "Which delivery window should I use: morning or afternoon?",
                "Should the delivery arrive on Monday or Tuesday?",
            }
            if source.question not in permitted:
                return HumanReviewDisclosure(status="redacted")
            text = source.question
        else:
            # Validate every call; no sibling disappears from the review scope.
            if any(
                args != {"window": "morning"}
                for args in source.arguments_by_call.values()
            ):
                return HumanReviewDisclosure(status="redacted")
            text = "Schedule the listed deliveries in the morning."
        return HumanReviewDisclosure(
            status="permitted",
            sensitive_content="application_attested",
            fields=(HumanReviewField(label="Delivery", text=text),),
        )
```

For an input view, show the permitted field and call
`app.resolve_user_input(UserInputResponse(session_id=view.session_id,
input_id=view.interaction_id, answer="morning", review_reference=view.reference))`.
For approval, call `app.resolve_tool_approval(ToolApprovalRequest(
session_id=view.session_id, approval_id=view.interaction_id,
tool_round_id=view.tool_round_id, tool_call_id=view.tool_call_id,
decision=ToolApprovalDecision.APPROVE, review_reference=view.reference))`.
Consume these async event streams. If content is redacted, show the fixed guidance
and either contact the application owner or explicitly deny the approval using
`ToolApprovalDecision.DENY`; never manufacture an empty executable proposal.
