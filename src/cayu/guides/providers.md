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
