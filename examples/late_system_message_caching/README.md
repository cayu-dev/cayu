# Late system message caching

This example measures whether prompt caching survives an agent that appends a
changing system message after the conversation history on every call. The τ
agent does this with its transient controller and memory context.

One six-turn conversation with about 11k tokens of stable early context runs in
two variants:

- `static`: a fixed leading system prompt only.
- `late_system`: the same conversation, plus a trailing system message that
  changes every turn and isn't kept in history.

Each turn records provider-reported input, cache-read, cache-write and uncached
input tokens, and a cost from `cayu.default_price_book()`. The example drives the
provider with `ModelRequest` directly, as a custom agent bridge does. Providers
are constructed with defaults, so the numbers show what an application gets
without tuning cache settings.

## Deterministic mode

This mode makes no API calls. Real `OpenAIProvider` and `AnthropicProvider`
instances serialize every turn through recording transports. The check asserts:

- the late-system variant's serialized prefix is append-only across turns;
- OpenAI sends the trailing message as a `developer` item, with one stable
  `prompt_cache_key`;
- the default Anthropic cache markers sit on the system prompt and the stable
  history, never on the changing `<system>` note.

Token counts in this mode are simulated from the serialized payloads, and the
result labels them that way.

```bash
uv run python -m examples.late_system_message_caching.app
```

## Live mode

```bash
OPENAI_API_KEY=... uv run python -m examples.late_system_message_caching.app \
  --mode live --provider openai --trials 3
ANTHROPIC_API_KEY=... uv run python -m examples.late_system_message_caching.app \
  --mode live --provider anthropic --trials 3
```

The models default to the price-book IDs `gpt-6-luna` and `claude-haiku-4-5`.
`CAYU_LATE_SYSTEM_CACHE_MODEL` overrides the model. Live mode passes when turns
2–6 of the late-system variant each read at least half of their input from the
provider cache. A trial costs about $0.005 on OpenAI and about $0.05 on
Anthropic. Results are written to
`<root>/.cayu-example-results/late_system_message_caching/`.

The dated before/after numbers, and how "before" was measured on `origin/main`,
are in [late-system-message-caching-results.md](../../docs/late-system-message-caching-results.md).
