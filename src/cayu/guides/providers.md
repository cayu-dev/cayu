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
