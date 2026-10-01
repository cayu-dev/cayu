# Model policy for new sessions

Model policy is explicitly enabled application configuration, not an inference
credential feature. It changes defaults for **new sessions** only. Selection is:
an explicit `RunRequest.target`, then the locally installed policy default, then
the registered `AgentSpec`. Existing sessions retain their recorded model.
Prepared durable subagents retain their selected provider, model and policy
provenance across queue delay and worker restart; later policy updates do not
reselect their target.

Import the integration from `cayu.model_policy`. Create one
`ModelPolicyController` per local agent, with its local provider name, exact
Cloud scope and enrolled incarnation. Pass their `ModelPolicy` to
`CayuApp(model_policy=...)`. Scope identifies the Cloud Agent separately from the
local agent registration. Duplicate local-agent or Cloud-scope mappings are
rejected.

For direct API use, call `await app.start_model_policy()` before creating
defaulted sessions and `await app.stop_model_policy()` during shutdown. The
standard project entrypoint and HTTP server (`create_server` and `mount_cayu`)
own these calls automatically.
With a server application lifespan, application resources start before policy
workers and close after policy workers and server work have stopped.
Direct applications can use `async with app.model_policy_lifespan():` to preserve
both an application failure and any shutdown failure.
Stores and manually supplied channels belong to the application and must be
closed after the workers stop. No network connection is opened by construction.

## Generated projects

The generated Gateway project supports `CAYU_MODEL_POLICY_CONFIG`, an explicit
path to a deployment-owned JSON file. Without it, inference credentials do not
enable policy adoption or management access. The file contains:

```json
{
  "bindings": [{
    "agent_name": "assistant",
    "provider_name": "cayu_gateway",
    "origin": "https://cloud.example",
    "credential_env": "MY_POLICY_CREDENTIAL",
    "incarnation_id": "enrolled-incarnation",
    "incarnation_epoch": 1,
    "scope": {
      "organization_id": "organization",
      "cloud_agent_id": "cloud-agent",
      "application_id": "application",
      "instance_id": "instance",
      "integration_id": "integration",
      "credential_family_id": "policy-family",
      "inference_key_id": "inference-key"
    }
  }]
}
```

Supply a separately delegated policy credential in the named environment
variable. It needs current `policy:read` and `policy:report` authorization for
that exact enrollment; readback additionally requires `policy:report-read`.
The configuration neither enrolls an instance nor grants inference credentials
management permissions. Changing a credential requires reconstructing the
channel; environment rotation does not silently inherit its pending authority.

`open_application_stores(..., model_policy=True)` provides `model_policy_store`
alongside the other stores. SQLite and PostgreSQL use the same selected database;
PostgreSQL shares the application pool. Apply storage migrations before using a
PostgreSQL deployment. In-memory applications can use `InMemoryModelPolicyStore`.
Each binding has one leased writer. A crashed owner stays fenced until its
60-second lease expires; a second process cannot reuse its unexpired authority.

## Adoption, failure and local control

Workers attempt a fetch at startup and every 20 seconds (configurable up to 30).
Snapshots live for 60 seconds, but Runtime admits installation only within the
original 55-second adoption deadline. Exact snapshot replay does not extend it.
Publication retains the previous installation until timely durable completion
has been positively observed and recorded. An unconfirmed publication is rolled
back on recovery; a lost confirmation acknowledgement is reconciled from storage.
Same-boot freshness is persisted for cache recovery; platforms without a
verifiable boot identity must fetch a fresh snapshot after process restart.
Already installed defaults do not expire merely because management is offline.
Healthy snapshot and report maintenance does not disable the installed default.
After a storage outage, workers can reacquire an expired lease only if no other
live owner holds it; unresolved storage outcomes remain fenced until readback.
Explicit inference still goes through ordinary live provider authorization and
Runtime budgets; cached policy never grants spending authority.

The selected default must be eligible in the snapshot, present in `get_models()`,
and accepted by the provider's model preflight. Absent, ineligible, unknown, and
unsupported defaults produce authenticated refusal reports. Refusal leaves the
last installed default unchanged. Cloud reporting follows the atomic local
installation/pending-report commit. Lost acknowledgements retry that exact
report, and a restarted owner restores it before admitting defaulted work.

`await controller.override_default(model="...")` and
`await controller.withdraw()` work without a management channel. Both pause
automatic adoption; withdrawal retains the current local model. Call
`await controller.resume_adoption()` explicitly to enable adoption again.
These operations also retain their pending reports for later delivery.

The journal retains at most 256 decisions and 128 pending reports, pruning only
acknowledged prefixes. It refuses further changes if pending work prevents safe
pruning; it never drops an unacknowledged installation to make space.

Policy-selected runs record bounded `model_policy` evidence on the initial
`interaction.started` and `session.started` events. This identifies the exact
installation used by that session; it is not proof that other sessions switched,
that a replica is still alive, or that the model remains currently authorized.

Participant-session creation retains policy evidence in its native creation
binding. Delayed execution, including after restart, uses that creation-time
selection rather than a newer default. Participant sessions created without a
policy selection do not acquire policy provenance retroactively. Other execution-
profile compatibility checks still apply.
