# Cayu providers

Cayu focuses on official OpenAI, Anthropic, Google, AWS, and Vertex AI
endpoints. Services that expose OpenAI Chat Completions also work through
`ChatCompletionsProvider`.

Provider selection is explicit. `CAYU_PROVIDER` is only a scaffold convenience
for `openai`, `anthropic`, and `openai-subscription`; it is not the complete Cayu
provider surface. Credentials authenticate a provider but never select one.

## Primary integrations

| Service | Cayu provider | Setup |
| --- | --- | --- |
| OpenAI Platform | `OpenAIProvider()` | `OPENAI_API_KEY`; use an OpenAI model ID |
| Anthropic API | `AnthropicProvider()` | `ANTHROPIC_API_KEY`; use an Anthropic model ID |
| Google AI Studio | `ChatCompletionsProvider(name="google", api_key_env="GEMINI_API_KEY", base_url="https://generativelanguage.googleapis.com/v1beta/openai")` | Use a Gemini API model ID |
| Amazon Bedrock | `BedrockProvider(region_name=...)` | Install `cayu[aws]`; use AWS credentials and a Bedrock model or inference-profile ID |
| Anthropic on Vertex AI | `VertexProvider(project_id=..., region=...)` | Install `cayu[vertex]`; use Google credentials and a Vertex Claude model ID |
| OpenAI subscription | `OpenAISubscriptionProvider()` | Run `cayu auth openai login`; local development and evaluation only |

Google AI Studio automatically uses Gemini usage accounting. For Gemini through
another OpenAI-compatible Vertex or gateway endpoint, pass
`usage_dialect=UsageDialect.GEMINI` explicitly.

## Recoverable OpenAI background responses

Long OpenAI Responses calls can opt into provider-owned background execution:

```python
from cayu import AgentSpec, CayuApp
from cayu.providers import OpenAIProvider

app = CayuApp()
app.register_provider(OpenAIProvider(background=True), default=True)
app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"))
```

This is a provider-registration choice, not a per-request OpenAI option.
Cayu sends `background: true`, `stream: true`, and `store: true`, records the
OpenAI response ID before accepting later output, and can retrieve or resume
that same response after worker loss. `ModelRequest.options["openai"]` cannot
override these fields. The default `OpenAIProvider()` remains synchronous.

Enable this only after reviewing the deployment tradeoffs:

- OpenAI background responses have higher time to first token than synchronous
  responses.
- A non-ZDR project stores the response at OpenAI so it can be retrieved and
  resumed. OpenAI documents a 30-day Responses application-state retention
  period and says data for `store=true` responses is retained for at least 30
  days. OpenAI forces `store=false` under Zero Data Retention, but background
  mode still stores response data on disk for roughly ten minutes for polling.
  Confirm that behavior satisfies the application's retention policy before
  enabling it.
- Cayu currently enables this mode only for the global `api.openai.com` base
  URL and rejects region-specific OpenAI domains. OpenAI separately documents
  that `background=true` is unavailable on its EU regional route. Validate the
  account, model, project policy, processing location, and storage location for
  the intended production deployment.
- OpenAI does not document exact recovery of a lost create acknowledgement from
  Cayu's idempotency key. If the response ID was never made durable, Cayu reports
  `ambiguous_submission` and does not submit the request again automatically.

This capability reconnects one provider operation. It is distinct from
server-side conversation chaining (`previous_response_id`) and from Cayu's
durable transcript. `OpenAISubscriptionProvider`, `ChatCompletionsProvider`,
Anthropic, Bedrock, Vertex, and custom adapters do not gain this capability.
See OpenAI's [data controls](https://developers.openai.com/api/docs/guides/your-data)
and [background mode](https://developers.openai.com/api/docs/guides/background)
guides for the current provider policy.

## Compatible Chat Completions

OpenRouter, Fireworks, Baseten Model APIs, OpenCode Go, and other compatible
endpoints work through Cayu even though they are not scaffold choices. Register
the generic adapter and route the agent to its name:

```python
from cayu import AgentSpec, CayuApp, ChatCompletionsProvider

provider = ChatCompletionsProvider(
    name="fireworks",
    api_key_env="FIREWORKS_API_KEY",
    base_url="https://api.fireworks.ai/inference/v1",
)
app = CayuApp()
app.register_provider(provider, default=True)
app.register_agent(
    AgentSpec(
        name="assistant",
        model="accounts/fireworks/models/YOUR_MODEL_ID",
        provider_name="fireworks",
        system_prompt="Help the user.",
    )
)
```

Change the provider name, base URL, API-key environment variable, and model ID
together:

| Service | Base URL and credential | Model ID |
| --- | --- | --- |
| OpenRouter | `https://openrouter.ai/api/v1`; `OPENROUTER_API_KEY` | Provider slug such as `vendor/model` |
| Fireworks | `https://api.fireworks.ai/inference/v1`; `FIREWORKS_API_KEY` | `accounts/fireworks/models/...` |
| Baseten Model APIs | `https://inference.baseten.co/v1`; `BASETEN_API_KEY` | Baseten catalog model ID |
| OpenCode Go | `https://opencode.ai/zen/go/v1`; `OPENCODE_API_KEY` | Raw API ID such as `grok-4.5`, never `opencode-go/...` |
| Together AI | `https://api.together.ai/v1`; `TOGETHER_API_KEY` | Together catalog model ID |
| Mistral AI | `https://api.mistral.ai/v1`; `MISTRAL_API_KEY` | Mistral catalog model ID |
| Ollama | Commonly `http://127.0.0.1:11434/v1`; placeholder key | Pulled model name |
| vLLM | Commonly `http://127.0.0.1:8000/v1`; server key or placeholder | Served model name |

For local HTTP endpoints such as Ollama or vLLM, pass `allow_http=True`. OpenCode
Go models span multiple protocols: use `ChatCompletionsProvider` for its Chat
Completions models and the matching Cayu protocol adapter for its other models.
Always use the raw API model ID.

Set `AgentSpec.provider_name` when routing explicitly, or register one provider
with `default=True`. Keep credentials out of `AgentSpec`, model IDs, and source
files. `cayu check` reports missing or ambiguous provider routes without calling
a provider.
