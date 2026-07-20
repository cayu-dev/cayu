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

## Register the provider

```python
from cayu import AgentSpec, CayuApp, OpenAISubscriptionProvider

app = CayuApp()
app.register_provider(OpenAISubscriptionProvider(), default=True)
app.register_agent(AgentSpec(name="assistant", model="gpt-5.4"))
```

Model availability belongs to the user's subscription and may change. A model
accepted by the OpenAI Platform API is not necessarily available through the
subscription backend. Generated projects select `gpt-5.4` in subscription mode;
set `CAYU_MODEL` if the account offers a different model.

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
