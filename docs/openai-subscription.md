# OpenAI subscription authentication

Cayu can experimentally run an agent against the Codex backend using the
developer's own ChatGPT subscription. This is useful for local agent testing
when separate per-token API billing is not affordable.

## Sign in

```bash
cayu auth openai login
```

The normal flow starts a localhost callback on port `1455` and opens OpenAI's
authorization page. If a browser cannot run on the same machine, use the
device-code flow:

```bash
cayu auth openai login --headless
```

Inspect or remove the local sign-in without printing token material:

```bash
cayu auth openai status
cayu auth openai logout
```

Credentials are stored in `~/.cayu/auth.json` with mode `0600`. Set
`CAYU_HOME` to move the Cayu home directory. Do not copy this file into a
project, container image, shared server, log, or support request. Cayu keeps a
separate refresh-token chain instead of reading or modifying Codex CLI's token
store.

Cayu serializes credential changes across threads and processes. The last
writer that completes the locked transaction is authoritative. On supported
local POSIX filesystems, Cayu acknowledges a login, refresh, or logout only
after synchronizing the private replacement file and the containing directory.
A nested auth-store path that Cayu must create is built from private,
owner-controlled directory components, and each directory entry is synchronized
again before every credential mutation so an interrupted earlier creation
cannot weaken a later acknowledgement.
Before exchanging a rotating refresh token, Cayu uses the host's native
full-allocation primitive to reserve and synchronize the worst-case encoded
credential size in the actual staging file. The allocation remains live until
the complete rotated record has been written, so a capacity failure stops before
the provider call instead of stranding a consumed refresh token. If optional
size compaction fails, Cayu can publish the complete record with bounded trailing
JSON whitespace rather than discard it. A waiter also revalidates the pinned
auth directory after acquiring its process lock, immediately before publishing
rotated credentials, and before reporting success.
A failure after atomic publication can therefore leave a complete newer
credential record in place while still reporting that durability was not
confirmed; a later load adopts that complete record instead of repeating a
refresh. Hidden staging files left by a terminated process are ignored.

Credential writes fail closed when the host or filesystem cannot provide the
required atomic replacement and directory-synchronization primitives. Existing
credentials remain readable on such a host, but Cayu does not silently describe
a best-effort rotation as power-loss durable. This local-filesystem protocol
does not promise durability for remote or distributed filesystems whose server
acknowledgements do not honor the host's synchronization requests.
Coordination covers Cayu writers using the same auth-store path. External
same-user mutation of ancestor namespace entries while a credential transaction
is in flight is outside this contract; publication remains pinned to the opened
auth directory and will not be redirected to a replacement directory.

## Register the provider

```python
from cayu import AgentSpec, CayuApp, OpenAISubscriptionProvider

app = CayuApp()
app.register_provider(OpenAISubscriptionProvider(), default=True)
app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))
```

The experimental adapter also supports the same typed native hosted-search
authority as `OpenAIProvider` while the Codex backend continues to accept it:

```python
from cayu import OpenAIWebSearch

app.register_agent(
    AgentSpec(name="researcher", model="gpt-5.6-luna"),
    hosted_tools=[OpenAIWebSearch(search_context_size="low")],
)
```

This does not make the backend a public OpenAI contract. Upstream rejection is
a capability failure, not permission to spoof a first-party identity or silently
disable search. Opt-in live verification uses
`CAYU_OPENAI_HOSTED_WEB_SEARCH_SUBSCRIPTION_LIVE=1`; it records only bounded
structural evidence and never prints credentials, account identity, or response
prose.

## Execution security boundary

The subscription access token, refresh token, and account identifier are
**model-provider credentials**. They remain in the trusted Cayu process and
authorize only `OpenAISubscriptionProvider` requests. Registering this provider
does not add those values to a `Vault`, `SecretResolver`, runner environment,
tool context, workspace, virtual-egress grant, manifest, or artifact. Configure
workload credentials separately and explicitly when an agent integration needs
authority of its own.

For untrusted model-authored commands, use an isolated remote runner with
explicit guest environment and mount inputs, such as a hardened E2B or
Microsandbox deployment that has passed Cayu's provider-credential isolation
probe. Do not copy `~/.cayu/auth.json`, `CAYU_HOME`, your host home, or provider
key variables into the sandbox or image.

`LocalRunner` is for trusted local execution. Its default minimized child
environment does not inherit arbitrary provider variables, but it is not a
filesystem security boundary: the command still runs as your host user and may
read paths that user can access. `LocalRunner(inherit_env=True)` explicitly
copies the full host environment and can expose provider credentials. Never use
that opt-in as an isolation claim for untrusted code.

Model availability belongs to the user's subscription and may change. A model
accepted by the OpenAI Platform API is not necessarily available through the
subscription backend. Generated projects select `gpt-5.4` in subscription mode;
set `CAYU_MODEL` if the account offers a different model.

The adapter honors Codex's typed `end_turn` completion signal. When Codex
completes a response with `end_turn=false`, Cayu durably records that model step
and any visible commentary, then requests the next model step inside the same
run and interaction. It does not add a synthetic user message. Normal Cayu
step, budget, interruption, cancellation, and recovery limits remain in force.

## Support and policy boundary

This is not an OpenAI API key and does not turn a ChatGPT subscription into
general OpenAI Platform credit. The adapter supports Cayu model streaming and
tool calls through the Codex Responses endpoint; it does not provide
embeddings or the Platform input-token counting endpoint.

OpenAI documents ChatGPT sign-in for its Codex clients and the Codex SDK, but
does not currently document the raw Codex backend as a general third-party
provider API. Therefore this Cayu integration is explicitly experimental. It
may stop working, be rate-limited, or be rejected upstream without notice.
Review the current [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)
and [Codex plan documentation](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan)
before enabling it.

Cayu always sends `originator: cayu` and a Cayu user agent. It does not send a
Codex first-party identity or retry around an access-control rejection. A
`403`, unsupported-originator response, or vendor policy change is a stop
condition—not a reason to spoof headers. Use `OpenAIProvider` with a Platform
API key or another officially supported provider if that happens.

> **Intended-use boundary:** This path is intended for a subscription holder's
> own local development and evaluation. It is not intended for production,
> customer-facing or multi-user services, credential sharing, resale, or
> bypassing plan limits. For production, use the OpenAI Platform API or another
> officially supported provider.

Do not collect end-user ChatGPT credentials or expose subscription-backed
access as a service without written authorization from OpenAI.

This provider ships in Cayu core by an explicit repository placement decision:
it is an authentication and local-development mode for Cayu's foundational
OpenAI Responses adapter, including the shared CLI and project scaffold. It is
not a general exception to the policy that new third-party integrations belong
in standalone packages.
