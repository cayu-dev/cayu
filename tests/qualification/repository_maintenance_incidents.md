# Repository-maintenance incident exercise

Use an authorized disposable repository and a reviewed deployment. Do not inject
process loss into unrelated work. Record application/Runtime/image/configuration
identities, incident times, observations, decisions, receipts, interventions and
cleanup evidence privately. This procedure is guidance, not evidence that the
production exercise has passed. A local scripted-provider test cannot certify
remote provider or Docker cleanup.

## Diagnose before acting

Use the operator credential, not a product token, with exactly one tenant selector.
Keep the bearer header in a private curl configuration file rather than command
arguments. `RUN_ID` is the accepted business identifier, not a guessed session ID.

```sh
curl --config "$OPERATOR_CURL_CONFIG" --get \
  --data-urlencode "tenant=$TENANT" "$API/operator/runs/$RUN_ID"
curl --config "$OPERATOR_CURL_CONFIG" --get \
  --data-urlencode "tenant=$TENANT" "$API/operator/runs/$RUN_ID/tasks"
curl --config "$OPERATOR_CURL_CONFIG" --get \
  --data-urlencode "tenant=$TENANT" "$API/operator/runs/$RUN_ID/delivery"
```

The lookup's `allocated_references` identifies the four reserved tasks and exposed
coding/workflow sessions. Allocation does not prove existence: 404 before dispatch
is not failed recovery. Task observations distinguish absent, conflicting,
unavailable and recorded data. An expired recorded lease does not prove that a
worker or external operation stopped. `cancellation_marker=requested` is a fence,
not permission to reset the task. Task-only cleanup is `not_inspected`.

Read native task/session evidence through `/internal/cayu/api/tasks/{task_id}`
and `/internal/cayu/api/sessions/{session_id}` using those returned identifiers
and operator authentication. The delivery view reports historical phase evidence;
task completion, native delivery success and cleanup are different facts.
Do not silently classify absent/unknown evidence as successful cleanup.

## Coding worker loss

1. During the authorized trial, stop the exact coding process after dispatch.
   Record its exit and independently inspect any owned runner/container and
   remote model operation. Host process death alone does not prove they stopped.
2. Observe the original reservation and task, including cancellation/lease state.
   A replacement worker must not dispatch a new model merely because the old
   lease expired. Preserve source, artifacts, broker state and original IDs.
3. Plan through the same registered application's native recovery entrance.
   Use the exposed coding session ID from lookup. Keep output in a private
   directory outside the application image and repository source.

```sh
docker compose -f compose.yaml run --rm --no-deps api \
  cayu recovery plan --session "$CODING_SESSION_ID" --limit 1 \
  > "$PRIVATE_EVIDENCE/plan.json"
```

Check command success and inspect the complete plan: registration compatibility,
current model/tool stage, blockers, current ownership and allowed actions. Do not
edit the saved plan or manufacture a missing action. For an unresolved model,
`model_mark_interrupted` is a deliberate non-success disposition, not proof of
remote cancellation. Select it only when present and appropriate to the incident.
Save a JSON array of `RecoveryDecision` objects with the exact `item_id` and chosen
`action`; tool-effect decisions also require their operator evidence message.

```sh
docker compose -f compose.yaml run --rm --no-deps \
  --volume "$PRIVATE_EVIDENCE:/operator-evidence:ro" api \
  cayu recovery execute /operator-evidence/plan.json \
  --decisions /operator-evidence/decisions.json \
  --execution-id "$RECOVERY_EXECUTION_ID" \
  > "$PRIVATE_EVIDENCE/execution.json"
```

The private evidence directory must already exist, be absolute and be readable by
the application UID; restrict access to authorized operators. Keep one stable
execution ID and the same files for acknowledgement-loss replay. Inspect every
item's receipt/status, not just CLI exit zero. A stale plan requires fresh
inspection and an explicitly reviewed decision; it must not grant an old action.

Registered session recovery, settled product reconstruction, outer task settlement
and external resource cleanup are separate boundaries. This application does not
ship a general task-reset or unattended recovery worker. In particular, a native
task cancellation receipt is not a completed coding result. If the production
validator cannot positively establish quiescence and exact effect evidence, keep
the task fenced, retain the resources and report the incident as unresolved.
Do not copy the local test harness's synthetic cleanup evidence into production.

## Restart at pending approval

Stop/restart after Git preparation is durable and before approval. Observe the same
task and `GET /operator/runs/{id}/git/approval?tenant={tenant}`. Review the exact
request, prepared tree and destination. Only then POST exactly
`request_fingerprint`, `prepared_tree` and a stable `approval_id` to that route.
The actor comes from operator authentication, never a submitted actor override.
A stale tree/request must return conflict without queuing a replacement effect.
Retry the exact approval after response loss; do not invent a new approval ID.
HTTP 202 is queue acceptance, not a push or PR result.

GitHub requires separate consent at
`/operator/runs/{id}/github/approval?tenant={tenant}`. GET the reviewed request,
then POST exactly its `request_fingerprint` and a stable `approval_id`. Git consent
does not authorize a PR. A destination/revision change needs renewed authority.

## Lost publication acknowledgement and unavailable evidence

Inject acknowledgement loss only after the controlled remote accepts publication.
Keep the original worker and its task authority while native reconciliation is
pending. Observe delivery state without selecting a new dispatch. A conflicting
remote observation, unavailable store, absent receipt or unresolved cleanup must
remain distinct from success. A recorded next-poll time is not a new-worker lease.
The existing connector owner may follow its bounded native poll permission;
operator task reset or a second connector process is not equivalent.

After settlement, inspect `/operator/runs/{id}/result?tenant={tenant}` and its
acceptance/cost links. Verify the exact commit and PR head, required hosted checks,
review and cleanup evidence. An HTTP error is not partial verified completion;
unpriced/unknown charges remain visible. Never merge as part of this exercise.

## Exit criteria and retained evidence

Use the deployment guide's ordered stop procedure. Record positive session/task
settlement, owned runner disposal or explicitly retained recovery state, durable
artifact retention, and process cleanup for each incident. An unknown cleanup
outcome fails production qualification even when the system correctly fences it.
Do not use private SQL, hand-edited checkpoints, `down -v`, prune or replacement
IDs to make a trial look recovered. Record every intervention and remaining gap.
The complete production incident exercise, including any unresolved task-settlement
validator requirement, must be demonstrated separately from these instructions.
