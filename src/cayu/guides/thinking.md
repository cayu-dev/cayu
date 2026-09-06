# Thinking effort compatibility

Use public `ThinkingConfig(effort="max")` on `AgentSpec.thinking`,
`RunDefaults.thinking`, `RunRequest.thinking`, `ResumeRequest.thinking`, or
`StepRunOptions.thinking`. Request/step overrides take precedence over agent
settings, which take precedence over application defaults. These values survive
JSON serialization and durable approval/dispatch recovery without normalization.

The exact vocabulary is `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`.
`xlow` remains invalid: the audit found no verified native `xlow` contract in these
adapters. It is not translated to `minimal` or `low`. Effort names are not a
cross-provider scale; `xhigh`, `high`, and `max` are distinct requests.

`enabled=True` enables the configuration, not necessarily model reasoning.
`effort="none"` explicitly requests native non-reasoning behavior; `minimal`
still permits reasoning. Both require `enabled=True`. `enabled=False` retains
its historical best-effort behavior (Anthropic disables, OpenAI/compatible Chat
Completions omit the typed control). Combining disabled thinking with any effort,
or combining effort with `max_tokens`, fails validation. Omitting effort preserves
existing defaults and legacy budget behavior. Typed effort overwrites raw effort;
unrelated raw siblings such as `reasoning.summary` and `output_config.format`
remain intact. No adapter downgrades or drops an explicit typed effort.

Provider-contract audit: **2026-09-05**. “Declared” below means documented wire
support, not tested backend availability or evidence of internal reasoning work.

| Adapter / transport | Typed effort contract |
| --- | --- |
| OpenAI Responses (streaming, buffered construction, background create) | Declared `reasoning.effort`; exact model restrictions below. |
| OpenAI Chat Completions | Declared `reasoning_effort`; same model effort restrictions, with Responses-only models rejected. |
| Compatible Responses and Chat Completions (including OpenRouter) | Forward the exact wire value. Unknown model/deployment/router compatibility is unresolved; acceptance must be established at the selected backend. |
| OpenAI subscription Responses | Same Responses wire construction and local model checks; subscription entitlement and backend acceptance are unverified independently of Platform support. |
| Anthropic Messages and Vertex Anthropic Messages | `thinking.type=adaptive` plus exact `output_config.effort`; model restrictions below. Vertex model/region availability remains backend-dependent. |
| Google AI Studio via Chat Completions | Exact `reasoning_effort`, subject to the Google rows below. This adapter does not emit native Gemini `thinking_level`. |
| Bedrock Converse | No implemented typed effort mapping; any explicit effort fails locally. Native `additionalModelRequestFields` do not establish typed effort support. |
| Custom providers | Their adapter owns compatibility and validation; this matrix makes no support claim. |

The following known model rules apply to exact IDs, `-latest` aliases, and dated
snapshots (`-YYYY-MM-DD`, `-YYYYMMDD`, or Vertex `@YYYYMMDD`). These are bounded
incompatibility rules, **not a model allowlist**. Unknown identifiers are accepted
locally within the adapter vocabulary and sent unchanged. Recognized IDs retain
these restrictions even through a compatible gateway; arbitrary deployment IDs
and vendor-prefixed router slugs do not establish native-model identity.

| Model IDs | Allowed explicit effort |
| --- | --- |
| `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | `minimal`, `low`, `medium`, `high` |
| `gpt-5-pro` | `high`; Responses only |
| `gpt-5.1` | `none`, `low`, `medium`, `high` |
| `gpt-5.1-codex` / `gpt-5.1-codex-max` | `low`, `medium`, `high` / additionally `xhigh`; Responses only |
| `gpt-5.2`, `gpt-5.4`, `gpt-5.5` | `none`, `low`, `medium`, `high`, `xhigh` |
| `gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` | `none`, `low`, `medium`, `high`, `xhigh`, `max` |
| `gpt-6-astra` | `low`, `medium`, `high`, `xhigh`, `max` |
| `o1`, `o3`, `o3-mini`, `o4-mini` | `low`, `medium`, `high` |
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` | No typed effort |
| Claude Opus 4.6, Sonnet 4.6, Mythos Preview | `low`, `medium`, `high`, `max` |
| Claude Opus 4.7/4.8/5, Sonnet 5, Fable 5/5.1, Mythos 5/5.1 | `low`, `medium`, `high`, `xhigh`, `max` |
| Claude Opus 4/4.1/4.5, Sonnet 4/4.5, Haiku 4.5, 3.7 Sonnet, 3.5 Sonnet/Haiku | No adaptive typed effort. Opus 4.5's manual-thinking effort contract is not the adaptive mode selected by `ThinkingConfig.effort`. |
| Unknown Anthropic model | `low`, `medium`, `high`, `xhigh`, `max` forwarded; model acceptance unresolved. `none` and `minimal` fail locally. |
| `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro` | `minimal`, `low`, `medium`, `high` |
| `gemini-2.5-flash`, `gemini-2.5-flash-lite` | `none`, `minimal`, `low`, `medium`, `high` |

Google documents its own compatibility mappings: `minimal` becomes `low` on
3.1 Pro, while 2.5 maps `minimal`/`low`, `medium`, `high` to budgets of 1,024,
8,192, 24,576. Cayu forwards the effort string; those translations occur at
Google, not in Runtime. Do not combine it with Google's raw thinking level/budget.

Local incompatibility raises a bounded `ValueError` before authentication or model
network dispatch, with allowed values or an adapter-selection remedy. It includes
no model input, credentials, URLs, or request bodies. Upstream rejection continues
through the provider's credential-safe API error path; it never triggers an effort
fallback. A successful fingerprint or payload build proves local validity and
requested/emitted settings only. Keep requested identity, emitted request identity,
and backend-reported identity separate.

The GAIA `gpt-5.6-luna` / `max` compatible Responses path has credential-free
transport coverage. Operational support through `codex-lb.cayu.ai` requires a
**separately authorized bounded probe** confirming the emitted request and a
successful backend response. Neither mocked responses nor launch metadata proves
backend acceptance; even a successful real response does not prove internal
reasoning allocation. No paid probe or evaluation is part of the unit tests.

Audit sources: OpenAI [Chat Completions schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning),
[GPT-5](https://developers.openai.com/api/docs/models/gpt-5),
[GPT-5 Pro](https://developers.openai.com/api/docs/models/gpt-5-pro),
[GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1),
[Codex-Max](https://developers.openai.com/api/docs/models/gpt-5.1-codex-max),
[GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2),
[GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4),
[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5),
[Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[model catalog](https://developers.openai.com/api/docs/models);
Anthropic [effort](https://platform.claude.com/docs/en/build-with-claude/effort) and
[thinking modes](https://platform.claude.com/docs/en/about-claude/models/extended-thinking-models);
Google [OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai).

