# Late system message caching: measured results

Measured on 2026-10-08 with the
[late-system-message caching example](../examples/late_system_message_caching/),
before and after the change that keeps later system messages in place and turns
Anthropic prompt caching on by default.

## Scenario

One conversation of 6 turns with about 11.7k (OpenAI) or 13.3k (Anthropic)
input tokens of stable early context: a system prompt, 259 policy notes, and
one short question per turn. There are two variants:

- `static`: only the fixed leading system prompt.
- `late_system`: the same conversation, plus a system message after the history
  on every call. It changes every turn and isn't kept in history, like the τ
  agent's transient controller context.

Each conversation starts with a unique nonce, so turn 1 can't read a cache
written by another trial or by the other variant. Token counts are the
provider-reported usage, normalized by `normalize_usage_metrics`. Costs come from
`cayu.default_price_book()` (version 2026-10-01), using the price-book model IDs
`gpt-6-luna` ($0.10/M input, $0.01/M cache read, $0.125/M cache write) and
`claude-haiku-4-5` ($1/M input, $0.10/M cache read, $1.25/M 5-minute cache write).

"Before" is `origin/main` at 36c575516. "After" is the
`fix/system-message-prefix-caching` branch. Each configuration ran 3
trials. Both use default provider construction (`OpenAIProvider()`,
`AnthropicProvider()`).

## Results

"Turns 2–6 cached" is the share of turns 2–6 input tokens that the provider
reported as cache reads. The range covers the 3 trials.

| Provider | Variant | Before: turns 2–6 cached | After: turns 2–6 cached | Before: cost per conversation | After: cost per conversation |
| --- | --- | --- | --- | --- | --- |
| OpenAI `gpt-6-luna` | `static` | 99.7% | 99.7% | $0.0021–0.0022 | $0.0022 |
| OpenAI `gpt-6-luna` | `late_system` | **0.0%** | **99.5%** | $0.0090 | $0.0022 |
| Anthropic `claude-haiku-4-5` | `static` | **0.0%** | **99.6%** | $0.0811–0.0813 | $0.0243–0.0244 |
| Anthropic `claude-haiku-4-5` | `late_system` | **0.0%** | **99.4%** | $0.0810–0.0814 | $0.0245–0.0248 |

What changed:

- **OpenAI, `late_system`.** Before, Cayu moved the trailing system message into
  `instructions`, the first thing in the request. Every turn therefore had a new
  prefix and read nothing from cache. It even reported a fresh cache write for
  the whole prompt on each turn. After, the message is sent in place as a
  `developer` item, and turns 2+ read 99.5% of their input from cache. That is
  within 0.2 points of the static case.
- **Anthropic, both variants.** Before, `AnthropicProvider()` sent no
  `cache_control` markers, so nothing was cached even without a late system
  message. After, the default `CachePolicy()` marks the system prompt and the
  history before the newest message. The `<system>` note stays outside the
  cached prefix. Cost per 6-turn conversation fell by about 70%: one cache write
  on turn 1 (1.25x), then cache reads at 0.1x. The saving grows with longer
  conversations.

### Per-turn detail, `late_system`, trial 1

OpenAI before (main):

| Turn | Input | Cache read | Cache write | Uncached | Cost |
| --- | --- | --- | --- | --- | --- |
| 1 | 11,760 | 0 | 11,757 | 3 | $0.00150 |
| 2 | 11,802 | 0 | 11,799 | 3 | $0.00150 |
| 3 | 11,844 | 0 | 11,841 | 3 | $0.00149 |
| 4 | 11,873 | 0 | 11,870 | 3 | $0.00150 |
| 5 | 11,906 | 0 | 11,903 | 3 | $0.00149 |
| 6 | 11,937 | 0 | 11,934 | 3 | $0.00151 |

OpenAI after:

| Turn | Input | Cache read | Cache write | Uncached | Cost |
| --- | --- | --- | --- | --- | --- |
| 1 | 11,763 | 0 | 11,733 | 30 | $0.00148 |
| 2 | 11,805 | 11,733 | 42 | 30 | $0.00015 |
| 3 | 11,847 | 11,775 | 42 | 30 | $0.00014 |
| 4 | 11,876 | 11,817 | 29 | 30 | $0.00014 |
| 5 | 11,907 | 11,846 | 31 | 30 | $0.00014 |
| 6 | 11,939 | 11,877 | 32 | 30 | $0.00015 |

Anthropic before (main):

| Turn | Input | Cache read | Cache write | Uncached | Cost |
| --- | --- | --- | --- | --- | --- |
| 1 | 13,314 | 0 | 0 | 13,314 | $0.01345 |
| 2 | 13,356 | 0 | 0 | 13,356 | $0.01352 |
| 3 | 13,401 | 0 | 0 | 13,401 | $0.01357 |
| 4 | 13,447 | 0 | 0 | 13,447 | $0.01361 |
| 5 | 13,492 | 0 | 0 | 13,492 | $0.01360 |
| 6 | 13,529 | 0 | 0 | 13,529 | $0.01369 |

Anthropic after:

| Turn | Input | Cache read | Cache write | Uncached | Cost |
| --- | --- | --- | --- | --- | --- |
| 1 | 13,319 | 0 | 13,281 | 38 | $0.01677 |
| 2 | 13,360 | 13,281 | 41 | 38 | $0.00155 |
| 3 | 13,399 | 13,322 | 39 | 38 | $0.00153 |
| 4 | 13,434 | 13,361 | 35 | 38 | $0.00155 |
| 5 | 13,473 | 13,396 | 39 | 38 | $0.00150 |
| 6 | 13,503 | 13,435 | 30 | 38 | $0.00158 |

Turn 1 after the change costs more on Anthropic ($0.0168 instead of $0.0135)
because it writes the cache. Every later turn costs about a ninth as much.

## Limits

- One workload shape, one model per provider, 3 trials each. The numbers show
  the mechanism and its size for a long stable prefix. They aren't a general
  savings claim. A short one-off request pays the Anthropic cache-write premium
  and saves nothing.
- Bedrock and Vertex now place the same markers by default (Bedrock only for
  documented cacheable Claude families). They weren't measured live here.
- The example drives the provider with `ModelRequest`, as a custom agent bridge
  does. Runtime-built requests also carry a per-lineage `prompt_cache_key` on
  OpenAI. On this branch the example sets the same field itself.

## Recall and tool exposure (`recall_exposure`)

A later investigation found three more prefix-cache breakers on the τ³ banking agent:
automatic recall removed the previous turn's memory block from its user message,
exposure changes rewrote the tools array, and integer bounds on `number` fields
rendered as `100` or `100.0` depending on the upstream replica. The
`recall_exposure` variant runs the same six questions through `CayuApp` with
automatic recall, an exposure policy that swaps one tool at turn 3 and hides
another from turn 5, `tool_exposure_mode="stable_catalogue"`, and `number` bounds
in the tool schemas. The policy notes are part of turn 1's user message, behind
that turn's memory block.

These numbers are from deterministic mode, so the token counts are simulated
from the serialized payloads, not reported by a provider. "Before" runs the same
variant with `retain_earlier_recall=False` and the default
`tool_exposure_mode="filtered_tools"`. Anthropic's default markers write turn 1's
notes on turn 2, so turns 3–6 are the comparable share there.

| Provider (simulated) | Before: turns 3–6 cached | After: turns 3–6 cached | Before: cost | After: cost |
| --- | --- | --- | --- | --- |
| OpenAI `gpt-6-luna` | 48.2% | 96.6% | $0.0055 | $0.0022 |
| Anthropic `claude-haiku-4-5` | 48.4% | 94.3% | $0.0635 | $0.0381 |

The live measurement for this variant hasn't been run yet. Run it with the live
commands below; it passes when turns 3–6 each read at least half of their input
from the cache.

## Reproduce

Deterministic mode makes no API calls. It asserts that the serialized prefix is
append-only and that the default markers are on the stable history:

```bash
uv run python -m examples.late_system_message_caching.app
```

Live, on this branch:

```bash
OPENAI_API_KEY=... uv run python -m examples.late_system_message_caching.app \
  --mode live --provider openai --trials 3
ANTHROPIC_API_KEY=... uv run python -m examples.late_system_message_caching.app \
  --mode live --provider anthropic --trials 3
```

For "before", the same example directory was copied into a detached worktree of
`origin/main`, then run from that worktree with main's `src` first on
`PYTHONPATH`. The live assertion fails there by design:

```bash
git worktree add --detach /tmp/cayu-main origin/main
cp -R examples/late_system_message_caching /tmp/cayu-main/examples/
cd /tmp/cayu-main
# Use the branch checkout's environment; main's src comes first on the path.
PYTHONPATH=$PWD/src:$PWD /path/to/branch/.venv/bin/python \
  -m examples.late_system_message_caching.app --mode live --provider openai --trials 3
```

Each trial writes its full per-turn JSON to
`.cayu-example-results/late_system_message_caching/<run_id>.json` under
`--root`. The run IDs for these results:

- OpenAI before: 449b226a, 1d573a1c, b6829100
- OpenAI after: 50ae74eb, cf54849f, f03546c9
- Anthropic before: fd679192, 060771ea, c6d97dfc
- Anthropic after: 1c7c3e34, ee662687, a494e70d

All 12 trials together cost about $0.70.
