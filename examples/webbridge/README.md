# WebBridge application recipes

`research.py` implements `browse -> extract -> verify` over an explicit hosted
`WebBridge`. Search and fetch still run through ordinary `web_search` and
`web_fetch` tools. Every fetched page is bound to the canonical search-result
URL, retains its final URL and truncation evidence, remains untrusted, and has
an independent failure record so one denied or malformed page does not erase
the others.

```python
from cayu import (
    Environment, EnvironmentSpec, ExaWebAdapter,
    ExecutionProfileBehaviorIdentity, SecretRef, ToolContext, WebBridge,
)
from examples.webbridge.research import browse_extract_verify, register_researcher

hosted_environment_identity = ExecutionProfileBehaviorIdentity(
    name="hosted-environment", behavior_version="1", implementation_version="2026-08-20",
)
bridge = WebBridge.hosted(
    adapter=ExaWebAdapter(api_key_ref=SecretRef(name="exa_api_key")),
    execution_profile_identity=ExecutionProfileBehaviorIdentity(
        name="exa-web", behavior_version="1", implementation_version="2026-08-20",
    ),
)
app.register_environment(
    Environment(
        EnvironmentSpec(
            name="hosted", execution_profile_identity=hosted_environment_identity,
        ),
        proxy=credential_proxy,
    ),
    default=True,
)
register_researcher(app, bridge, model="your-model")
evidence = await browse_extract_verify(
    bridge,
    ToolContext(session_id="research", proxy=credential_proxy),
    "Cayu runtime contracts",
)
```

`credential_proxy` must declare the bridge authority through the synchronous,
side-effect-free `supports_webbridge_credential_authority(...)` seam. Built-in
`PassthroughProxy` and `AllowlistProxy` implementations do so without resolving
the referenced credential; custom proxies fail registration until they provide
an equivalent declaration.
Restart-safe hosted sessions require stable identities for both the opaque
adapter and the selected environment; the bridge identity does not substitute
for `EnvironmentSpec.execution_profile_identity`.

`daily_check.py` deliberately separates recurrence from durable execution:

1. a platform cron invokes `external_cron_tick(...)` once per desired day and
   persists any explicitly selected `environment_name` in the task input;
2. the tick derives a deterministic task ID and reconciles an identical task;
3. any Cayu task worker runs `daily_check_worker(...)`; its startup sweep
   settles terminal ownerless work, while an abandoned attached session remains
   fenced for the application's normal session recovery or operator control
   plane instead of being acted on from a stale task-list lease observation;
4. the registered `daily_web_checker` agent calls the profile's ordinary web
   tools, and its session/task result is durable;
5. application-owned code loads that terminal record with
   `load_durable_daily_result(...)` and may publish a notification from it.

Cayu does not currently ship a recurring scheduler, and web tools do not own
recurrence or wake a sleeping worker. Use platform cron, Kubernetes CronJob,
GitHub Actions, or another external scheduler to make the tick. Durable
future-eligibility and self-wake scheduling are separate future capabilities.

For sandboxed execution, construct the bridge directly from the same
factory/store pair that the app registers. No session resource is allocated at
setup; the factory declares pre-create admission and the exact pinned workload,
then the tools revalidate the materialized runner on every dispatch.

```python
from cayu import (
    AgentSpec, ApprovedEgressDestination, BrowserEgressPolicy, EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    LocalArtifactStore, VirtualEgressEnvironmentFactory, WebBridge,
)

artifacts = LocalArtifactStore(".cayu/browser-artifacts")
policy = BrowserEgressPolicy(
    name="public-docs",
    allowed_hosts=["docs.example.com"],
    allowed_path_prefixes=["/"],
)
browser_environment_identity = ExecutionProfileBehaviorIdentity(
    name="browser-egress", behavior_version="1", implementation_version="2026-08-20",
)
factory = VirtualEgressEnvironmentFactory(
    policies={policy.name: policy},
    approved_destinations=[ApprovedEgressDestination(
        destination="docs.example.com", policy_name=policy.name,
    )],
    runner_kind="docker",
    image="cayu-browser-fetch:7-playwright-1.62.0",
    artifact_store=artifacts,
    execution_profile_identity=browser_environment_identity,
)
bridge = WebBridge.sandboxed_browser(
    environment=factory,
    browser_image="cayu-browser-fetch:7-playwright-1.62.0",
)
app.register_environment_factory(
    EnvironmentSpec(
        name="browser", execution_profile_identity=browser_environment_identity,
    ),
    factory,
    artifact_store=artifacts,
    default=True,
)
bridge.register_agent(
    app,
    AgentSpec(name="browser-researcher", model="your-model"),
)
```
