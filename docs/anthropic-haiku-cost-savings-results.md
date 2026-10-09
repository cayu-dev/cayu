# Live Anthropic Haiku cost-savings results

Date: 2026-07-13

**Cayu cut measured Claude Haiku 4.5 cost by 66% for prompt-cache compaction
across three paired live trials. The July research-branch comparison recorded
here is invalid as savings evidence; see the correction below.**

These are provider-reported live measurements, not deterministic fixtures. All
six July trials used `claude-haiku-4-5-20251001` through the direct Anthropic
API and passed the example-specific assertions that existed at the time.

> **Correction (2026-10-09): the research-branch result below is not valid savings evidence.**
> In July the source session compacted with `TranscriptDigestCompactor(max_summary_chars=2_000)`.
> At that time the digest compactor cut its summary to 2,000 characters but still claimed to
> cover the whole 15,242-character research context, so the compacted forks silently lost
> about 13k characters of source. Part of the measured saving came from dropped context, not
> from compaction. dfc85d174 (2026-07-22) made the compactor claim only messages that fit.
> After that change the same example compacted nothing and its forks were no smaller.
> The example now compacts with `PromptCacheCompactor`, which covers the full source;
> live reruns put compacted branches at about 0.6x of baseline input. The prompt-cache
> compaction result is unaffected: a 2026-10-09 rerun on `claude-haiku-4-5` measured
> 58–68% savings per trial.

These July artifacts predate `PairedCostQualityComparisonReport` v1. Their
measurements remain historical evidence under the denominators below; current
example runs emit the typed public report and must not relabel an old artifact
as v1 without rerunning it.

## Results

| Workload | Verified trials | Paired baseline | Cayu candidate | Aggregate savings | Trial median | Trial range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt-cache compaction | 3/3 | $0.025633 | $0.00867080 | **66.17%** | **67.04%** | 63.63%–67.82% |
| Compacted research branches (invalid; see correction) | 3/3 | $0.125362 | $0.064486 | ~~48.56%~~ | ~~52.11%~~ | ~~37.27%–53.31%~~ |
| Combined paired calls (invalid; includes the research branches) | 6/6 | $0.150995 | $0.0731568 | ~~51.55%~~ | — | ~~37.27%–67.82%~~ |

### Prompt-cache compaction

Each trial captured one real `CompactionRequest`, then ran:

- the cache-aware `PromptCacheCompactor` candidate, preserving the exact warm
  request prefix and reading about 7,590 cached tokens; and
- a bounded `ModelCompactor` control from the same compactable source with the
  same model and thinking configuration but no cache read.

The comparison prices only those paired first-compaction attempts. Candidate
session work and the comparison-only control remain separately attributed.
The denominator treats the earlier warm-up and cache-write premium as common
deployment setup. The comparison-only bounded control runs through a separate
recorder and does not replay that setup; whole-session token evidence keeps the
candidate's setup spend visible.

| Run ID | Cache read | Uncached input | Cache write | Candidate output | Control output | Candidate | Bounded control | Savings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `6a88545c7e2a` | 7,590 | 78 | 69 | 365 | 272 | $0.00274825 | $0.008539 | **67.82%** |
| `a2f251ee117a` | 7,591 | 78 | 63 | 431 | 257 | $0.00307085 | $0.008443 | **63.63%** |
| `1162b9f8c23c` | 7,587 | 78 | 68 | 386 | 296 | $0.00285170 | $0.008651 | **67.04%** |

Every trial also completed a second bounded incremental compaction and recovered
`CACHE_RETENTION_OK` from the original tool result. The final implementation
propagates the caller's compaction instruction into every later bounded cycle;
this prevents retention requirements from disappearing after the first cached
checkpoint.

The retained live artifact was produced by the code committed as
`947ae38c4eabfcccb66ab2a778a9ffb875c37f3b`. It also carries the whole
candidate-session and benchmark-harness token denominators. It predates the
final code that prices those denominators from every durable completion, and
its aggregate omitted the per-attempt cache categories needed to reconstruct
exact dollar totals.
Those historical totals therefore fail closed as unavailable instead of being
priced as ordinary input.

| Run ID | Candidate session tokens (in/out) | Candidate steps | Full harness tokens (in/out) | Harness steps | Whole-session/harness cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| `6a88545c7e2a` | 25,838 / 1,049 | 7 | 33,017 / 1,321 | 8 | Unavailable; cache split not retained |
| `a2f251ee117a` | 25,920 / 1,215 | 7 | 33,078 / 1,472 | 8 | Unavailable; cache split not retained |
| `1162b9f8c23c` | 25,788 / 1,180 | 7 | 32,959 / 1,476 | 8 | Unavailable; cache split not retained |

Current example runs now emit exact, retry-inclusive candidate-session and
full-harness costs when every provider attempt has priced completion usage;
otherwise those dollar totals fail closed as unpriced.

### Cache-aware research council

> See the correction at the top: these July branch numbers include context the digest
> compactor silently dropped and overstate the saving.

Each trial created three uncompacted and three compacted research branches from
the same resolved source checkpoint. The paired cost includes every provider
attempt inside those six branch sessions, including structured-output repairs
if they occur. Source preparation is common setup. Evaluator and repair are
separately attributed downstream sessions that were not run against a paired
control, so this branch-call denominator excludes them and is not presented as
a whole-workflow savings result.

| Run ID | Baseline input | Candidate input | Baseline output | Candidate output | Baseline | Candidate | Savings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `83f294fdf0374e00bbce3041c7b3bd1d` | 25,143 | 13,830 | 3,690 | 1,409 | $0.043593 | $0.020875 | **52.11%** |
| `dd7dbdf242ca418d9a12907517eb0e58` | 26,028 | 13,839 | 1,568 | 1,481 | $0.033868 | $0.021244 | **37.27%** |
| `0427f71064b54943a46080e070ad589c` | 26,991 | 14,022 | 4,182 | 1,669 | $0.047901 | $0.022367 | **53.31%** |

All trials passed the shared causal-budget, checkpoint persistence, identical
source/prompt, smaller provider input, fork lineage, strategy diversity,
evaluator-critique, and critique-repair assertions.

This is a paired **branch-call** result, not a whole-workflow savings claim.
Source preparation is common setup. Evaluation and repair are separately
attributed downstream sessions that were not run against a paired control, so
their cost is not subtracted from the baseline. Each final trial recorded 11
whole-workflow model completions; the paired baseline and candidate each used
three branch model steps, while the additional retry occurred downstream and
remains visible outside this denominator.

For cost governance, the same evidence also reports the retry-inclusive Cayu
workflow and the entire benchmark harness. The Cayu workflow includes source
preparation, the three compacted branches, evaluation, and repair. The harness
adds the three comparison-only uncompacted branches. These totals are spend
reports—not additional savings denominators—because there is no paired
whole-workflow control.

| Run ID | Cayu workflow tokens (in/out) | Cayu steps | Cayu workflow cost | Full harness tokens (in/out) | Harness steps | Full harness cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `83f294fdf0374e00bbce3041c7b3bd1d` | 54,535 / 4,985 | 8 | $0.079460 | 79,678 / 8,675 | 11 | $0.123053 |
| `dd7dbdf242ca418d9a12907517eb0e58` | 56,915 / 6,604 | 8 | $0.089935 | 82,943 / 8,172 | 11 | $0.123803 |
| `0427f71064b54943a46080e070ad589c` | 58,923 / 7,287 | 8 | $0.095358 | 85,914 / 11,469 | 11 | $0.143259 |

## Pricing provenance

The estimates use Anthropic's direct Claude API standard rates as of
2026-07-13:

- base input: $1.00 per million tokens;
- output: $5.00 per million tokens;
- 5-minute cache write: $1.25 per million tokens; and
- cache hit or refresh: $0.10 per million tokens.

Source: [Anthropic Claude Platform pricing](https://platform.claude.com/docs/en/about-claude/pricing).
The caller-supplied price book was versioned
`anthropic-claude-api-pricing-2026-07-13`; every result records its version,
generation time, source URL, as-of date, provider/model match, and currency.
The exact checked-in price book is
[`docs/evidence/anthropic-haiku-4-5-pricing-2026-07-13.json`](evidence/anthropic-haiku-4-5-pricing-2026-07-13.json).

The sanitized machine-readable trial envelope is
[`docs/evidence/anthropic-haiku-cost-savings-2026-07-13.json`](evidence/anthropic-haiku-cost-savings-2026-07-13.json).
It records every run ID and assertion name; paired input, output, cache, cost,
attempt, and missing-usage evidence; exact prompt-compaction reasoning tokens;
prompt whole-session and harness token denominators; retry-inclusive research
workflow and benchmark-harness totals; and the resolved pricing tier.

## Reproduce

Use the checked-in provenance-bearing price book, or create one for another price
snapshot, then run:

```bash
ANTHROPIC_API_KEY=... \
CAYU_ANTHROPIC_MODEL=claude-haiku-4-5-20251001 \
CAYU_PROMPT_CACHE_PRICE_BOOK=docs/evidence/anthropic-haiku-4-5-pricing-2026-07-13.json \
uv run python -m examples.prompt_cache_compaction.app \
  --mode live --provider anthropic --trials 3

ANTHROPIC_API_KEY=... \
CAYU_ANTHROPIC_MODEL=claude-haiku-4-5-20251001 \
CAYU_RESEARCH_COUNCIL_PRICE_BOOK=docs/evidence/anthropic-haiku-4-5-pricing-2026-07-13.json \
uv run python -m examples.cache_aware_research_council.app \
  --mode live --provider anthropic --trials 3
```

These percentages describe the measured prompt shapes, output lengths, cache
state, and pricing snapshot above. The enterprise claim is stronger than a
universal percentage: Cayu gives teams an executable way to reproduce the
savings on their own workloads while keeping cache writes, retries, repairs,
quality failures, and unknown pricing visible instead of treating them as free.
