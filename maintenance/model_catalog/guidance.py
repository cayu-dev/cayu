"""Workspace operating guidance for the verification environment — the domain knowledge of HOW
to read AI-provider pricing pages, kept separate from the agent's identity/system prompt and
delivered via cayu's Environment `workspace_instructions` (rendered into the session's system
message as a labeled workspace section). This is the "AGENTS.md for the pricing-page domain": it
travels with the environment, not the agent, so the agent stays a generic "verify a model" agent.
"""

WORKSPACE_GUIDANCE = (
    "EXTRACT WHAT THE PROVIDER ACTUALLY EXPOSES — pricing structures differ, so fill the fields "
    "that apply and leave the rest null:\n"
    "  - input/output per 1M tokens, and the cached-read price (cache hit on input). Most have these.\n"
    "  - CACHE WRITES with a TTL choice are provider-specific (Anthropic): the price to write to the "
    "default ~5-minute cache vs the extended 1-hour cache (the cost of keeping the cache alive "
    "longer). Report both. If the provider caches automatically with only a cached-read discount and "
    "no write charge, leave the cache_write fields null. When a provider publishes a cache-write "
    "rate that changes with context size (including current OpenAI tiers), put it on each "
    "context_tiers entry; do not flatten it into the Anthropic TTL fields.\n"
    "  - BATCH/async API prices (often ~50% off) if offered, else null.\n"
    "  - CONTEXT TIERS only if rates change by input size (e.g. Google ≤200k vs >200k, OpenAI >272k): "
    "return one entry per band. For flat pricing leave context_tiers null.\n"
    "  - context_window = advertised max INPUT context as a single round number (200000, 1000000); "
    "never add max-output, never sum two numbers; null if not clearly stated. Pricing tables often "
    "do NOT show the context window at all — if it isn't explicitly stated, leave it null (never "
    "infer it from a tier label, a price, or a '>N tokens' threshold).\n"
    "\nWATCH FOR PAGE-LAYOUT TRAPS (these cause wrong batch/tier values):\n"
    "  - 'Standard / Batch / Flex / Priority' appearing together are TABS (radio buttons), not "
    "columns. Only ONE (Standard) is active, so EVERY number you can read belongs to Standard. Do "
    "NOT treat the 2nd or 3rd group of numbers in a model's row as Batch or Flex.\n"
    "  - When a model's row has repeated Input/Cached/Output groups, that is a context-size split "
    "('short context' vs 'long context', or ≤N vs >N tokens): first group = base prices, later "
    "group(s) = larger-context TIERS → context_tiers. It is NEVER batch.\n"
    "  - Only fill batch_* if the page has a price EXPLICITLY labeled 'Batch' for THIS model that is "
    "distinct from standard. If batch is behind an unselected tab, or you are unsure whether a number "
    "is batch vs a context tier, set batch to null. A wrong batch is worse than a null one.\n"
    "\nAMAZON BEDROCK: preserve the committed exact inference-profile scope, source-region allowlist, "
    "service tier, and cache-write TTLs. Verify rates only for that exact combination. Global and "
    "geographic profiles are distinct products; the request's source region selects the price. "
    "Standard is represented by the API value 'default'. Do not infer an application-profile ARN's "
    "underlying model, invent Reserved/Provisioned rates, or import Batch/Priority/Flex rates into a "
    "Standard row. If official evidence does not identify the exact scope/tier/rate, leave the "
    "committed row unchanged and flag it for human review."
)
