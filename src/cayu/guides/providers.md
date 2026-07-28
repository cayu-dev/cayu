# Explicit provider and model selection

Provider intent must be configuration, not a guess based on which credential
happens to be present. Credentials authenticate a selected provider.

A generated application accepts `CAYU_PROVIDER=openai`, `anthropic`, or
`openai-subscription`; `cayu new --provider ...` can bake in the default.
`CAYU_MODEL` overrides the compatible model selected with that provider. With
no provider selection, inspection, checks, tests, and evals remain available,
while live `run.py` execution fails with a setup message.

## OpenAI

```python
provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
app.register_provider(provider, default=True)
agent = AgentSpec(name="assistant", model="gpt-5.6-luna", provider_name="openai")
```

## Anthropic

```python
provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
app.register_provider(provider, default=True)
agent = AgentSpec(
    name="assistant",
    model="claude-sonnet-4-6",
    provider_name="anthropic",
)
```

## OpenAI subscription

Run `cayu auth openai login`, select `openai-subscription`, and use
`OpenAISubscriptionProvider`. This experimental path is for the subscription
holder's own local development and evaluation, not production or multi-user
services.

Register multiple providers under distinct names when an application truly
routes across them. Set `AgentSpec.provider_name` for an explicit per-agent
route, or register one provider with `default=True`. Model-pattern routing must
resolve to exactly one provider; `cayu check` reports missing or ambiguous
routes without calling a provider.

## Gemini through an OpenAI-compatible endpoint

Gemini thinking models can report visible candidate output in
`completion_tokens` while `total_tokens` also includes hidden, billable thinking
tokens. `ChatCompletionsProvider` automatically selects `UsageDialect.GEMINI`
for Google's AI Studio OpenAI-compatible endpoint.

Vertex AI and gateway endpoints require an explicit commercial usage contract.
Vertex uses the same OpenAI-compatible endpoint family for Gemini and
non-Gemini models, so its hostname alone cannot select accounting semantics
safely:

```python
from cayu import ChatCompletionsProvider, UsageDialect

provider = ChatCompletionsProvider(
    name="vertex-gemini",
    api_key_env="GEMINI_API_KEY",
    base_url=(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/"
        "PROJECT_ID/locations/us-central1/endpoints/openapi"
    ),
    usage_dialect=UsageDialect.GEMINI,
)
```

The Gemini dialect attributes the positive difference between total tokens and
reported prompt plus visible completion tokens to reasoning output, billing it
once as output. Ordinary OpenAI-compatible endpoints retain
`UsageDialect.OPENAI` and unexplained total mismatches fail closed.

Provider subclasses can continue declaring `usage_dialect` as a class
attribute. An explicit constructor value takes precedence over that declaration
and automatic endpoint detection.
