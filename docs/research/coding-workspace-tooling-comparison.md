# Coding workspace tooling: Cayu vs. Pi, Claude Code, and Codex

Date: 2026-07-27
Status: primary-source comparison for the Cayu coding-workspace tool contract

## Scope and method

This report compares the coding-file and repository-inspection tools relevant to
the coding-workspace issue:

- reading and paging;
- exact or patch-based editing;
- create, overwrite, and delete semantics;
- glob and content search;
- repository-change inspection;
- output bounding;
- parallel execution and mutation ordering; and
- portability across local and remote workspaces.

It does not attempt to rank every product feature such as browser automation,
subagents, task management, or provider integrations.

The sources are revision-pinned primary-source snapshots:

| System | Source | Revision |
|---|---|---|
| Cayu | [public source](https://github.com/cayu-dev/cayu/tree/009f65281a05b37feb8231fc6e7f5bf0ecc60d62) | `009f65281a05b37feb8231fc6e7f5bf0ecc60d62` |
| Pi coding agent | [public source](https://github.com/badlogic/pi-mono/tree/a5afc3f171e422e08a2ccc342827719f9952f38a) | `a5afc3f171e422e08a2ccc342827719f9952f38a`, package `0.81.1` |
| Claude Code | [official package](https://www.npmjs.com/package/@anthropic-ai/claude-code/v/2.1.88) | extracted package `2.1.88`; file-level snapshot evidence is not redistributable |
| Codex | [public source](https://github.com/openai/codex/tree/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9) | `6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9` |

Pi's package identifies itself as a coding agent with read, Bash, edit, and
write tools, and reports version `0.81.1`.
[`pi-mono/packages/coding-agent/package.json:1-5`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/package.json)
Claude Code's package reports version `2.1.88`.
`claude-code/package.json:1-16`

## Executive conclusion

Cayu is behind Pi and Claude Code on first-party coding ergonomics today because
it has no edit tool and its current write tool is a blind whole-file
replacement. It is also behind all three agents in the practical editing loop:
Codex has `apply_patch`, Pi has multi-edit exact replacement, and Claude Code has
stateful read-before-edit/write validation.

The coding-workspace issue is directionally correct, but its current digest language is not
sufficient to make the stale-write guarantee true. A tool that reads a file,
checks its SHA-256 digest in Python, and later calls `write_bytes` still has a
time-of-check/time-of-use race. The conditional check and mutation must be one
workspace operation, with a documented concurrency domain.

If Cayu adds that workspace-level conditional-mutation seam, it can be safer
than all three references for shared or remote workspaces:

- Pi serializes mutations to the same path only inside one process and has no
  stale-content precondition.
- Claude Code has the strongest current guard, but the guard is session-local
  `mtime`/content state followed by a separate filesystem write.
- Codex uses patch context as an implicit precondition, but matching is fuzzy,
  deletion has no content precondition, and multi-file patches may partially
  commit.

A dedicated bounded `GitChangesTool` would also be a meaningful Cayu
differentiator. None of the three references exposes a standalone model-visible
tool with structured, pageable repository status and diff evidence. Claude Code
attaches structured patches and optional Git diffs to individual mutations, and
Codex emits a turn-level diff event for UI consumers, but their agents still use
shell commands for repository-wide inspection.

## Comparison matrix

| Dimension | Cayu today | Pi 0.81.1 | Claude Code 2.1.88 | Codex snapshot |
|---|---|---|---|---|
| Dedicated coding surface | `read_file`, `write_file`, `list_files`, `search_text`, `exec_command` | `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls` | Broad surface: Bash, Glob, Grep, Read, Edit, Write, NotebookEdit, web, tasks, agents, and more | Primarily shell, `apply_patch`, image viewing, MCP, and runtime utilities |
| Read paging | Byte-prefix only; no offset or digest | Line offset/limit, but reads the entire file before slicing | Line offset/limit with true ranged read, total lines, and unchanged-read dedup | No dedicated read; uses bounded shell commands |
| Edit shape | None | Multiple non-overlapping replacements in one file | One replacement or replace-all per call | Multi-hunk, multi-file add/update/move/delete patch |
| Matching | N/A | Exact first, then fuzzy Unicode/whitespace normalization | Exact-ish lookup with quote normalization | Exact context first, then whitespace and Unicode normalization |
| Stale-write guard | None; write blindly overwrites | None; same-process per-path queue only | Prior read required; `mtime` plus content fallback checked again immediately before write | Current patch context is an implicit precondition; no digest/revision |
| File-level atomicity | N/A | Computes all edits then performs one write | Synchronous recheck and one write, but no cross-process CAS | Each file is derived then written; the whole multi-file patch can partially commit |
| Create/overwrite | One undifferentiated write | One undifferentiated create-or-overwrite write | New file allowed; existing file requires prior read | `Add File` may overwrite an existing file |
| Delete | No tool | No tool | No dedicated tool | `apply_patch` delete, without expected content |
| Glob/search | Contained `list_files`; robust pageable `search_text` | Bounded `find`/`grep`; no deterministic offset paging | Rich Glob/Grep; Grep has modes and offset paging | No dedicated Glob/Grep; prompt directs use of `rg` and `rg --files` |
| Repository evidence | Shell only | Shell only | Shell for repository-wide status/diff; mutation results can carry patches/Git diff | Shell for status/diff; turn-diff event tracks `apply_patch` mutations for UI |
| Output control | Excellent in `search_text`; basic prefix/list bounds elsewhere; durable structured metadata | Shared 2,000-line/50-KiB bounds and per-tool details | Per-tool persistence, hard token cap, and aggregate parallel-result budget | Per-call shell token budgets plus context-level tool-output budget |
| Parallel scheduling | Explicit `parallel_safe`; unsafe calls are ordering barriers | Parallel by default; any sequential-marked tool makes the whole batch sequential; per-path mutation queue | Consecutive read-only calls run concurrently; mutations form serial barriers | Parallel-capable handlers share a read lock; other tools take an exclusive barrier |
| Workspace boundary | Workspace-relative and backend-neutral | Local absolute paths are accepted; operations are replaceable | Absolute local paths plus application permissions | Environment filesystem plus sandbox/approval policy |

## Cayu today

Cayu's exported coding-related tools are `ExecCommandTool`, `ReadFileTool`,
`WriteFileTool`, `ListFilesTool`, and `SearchTextTool`; there is no edit,
delete, or Git-changes tool.
`src/cayu/tools/__init__.py:3-32`
`src/cayu/tools/__init__.py:44-75`

### What is already strong

The `Workspace` abstraction is the most important architectural advantage in
this comparison. Tools address one portable interface rather than assuming the
host filesystem. The current contract has bounded reads, writes, deletes, and
glob-style listing.
`src/cayu/workspaces/base.py:195-238`

Tool-level path validation rejects absolute paths, Windows roots and drives, and
relative traversal before the backend sees the request.
`src/cayu/tools/files.py:1290-1317`
That is a tighter default workspace boundary than Pi or Claude Code's
model-visible file schemas, which accept absolute paths.

`ToolSpec` explicitly declares side-effect class and parallel safety.
`src/cayu/core/tools.py:103-110`
`src/cayu/core/tools.py:128-170`
The runtime turns unsafe calls into ordering barriers while batching consecutive
safe calls.
`src/cayu/runtime/_tool_round_executor.py:2296-2314`

`ToolResult` also separates a bounded model-facing summary from structured,
durable evidence for dashboards and workflows.
`src/cayu/core/tools.py:221-246`
That is a good foundation for digest, paging, replacement-count, diff, and Git
status metadata.

`SearchTextTool` is already competitive with Claude Code's Grep tool. It exposes
files/content/count modes, limit and offset paging, case control, and
registration-time output bounds.
`src/cayu/tools/search.py:86-131`
It invokes `rg` with process-form arguments, caps file size and line previews,
and excludes configured directories.
`src/cayu/tools/search.py:288-324`
Its structured result explicitly reports truncation reasons and `next_offset`.
`src/cayu/tools/search.py:413-434`

### What is missing

`ReadFileTool` returns only the leading `max_bytes` of a workspace text file.
It reports byte counts and truncation, but has no offset, continuation token, or
content revision.
`src/cayu/tools/files.py:271-320`

`WriteFileTool` has a size limit and is correctly marked non-parallel, but it
unconditionally calls `workspace.write_bytes`; it does not distinguish create
from overwrite and has no stale-content precondition.
`src/cayu/tools/files.py:998-1058`

`ListFilesTool` is already Cayu's Glob equivalent. It has deterministic bounded
results and structured truncation, so a separate Glob tool is unnecessary.
`src/cayu/tools/files.py:1061-1114`

## Pi coding agent

Pi's complete built-in file/shell inventory is `read`, `bash`, `edit`, `write`,
`grep`, `find`, and `ls`.
[`pi-mono/packages/coding-agent/src/core/tools/index.ts:71-94`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/index.ts)
Its default coding subset is narrower—read, Bash, edit, and write—while the
read-only subset includes grep, find, and ls.
[`pi-mono/packages/coding-agent/src/core/tools/index.ts:138-165`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/index.ts)

### Read and output bounds

Pi's read schema has a one-indexed line offset and an optional line limit.
[`pi-mono/packages/coding-agent/src/core/tools/read.ts:20-30`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/read.ts)
However, its default implementation reads the entire file into a buffer and
splits all lines before applying the requested range. This is model-context
paging, not backend I/O paging.
[`pi-mono/packages/coding-agent/src/core/tools/read.ts:264-288`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/read.ts)
Continuation instructions are explicit and useful.
[`pi-mono/packages/coding-agent/src/core/tools/read.ts:289-315`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/read.ts)

Pi centralizes its ordinary text limits at 2,000 lines and 50 KiB, and caps
individual grep lines at 500 characters.
[`pi-mono/packages/coding-agent/src/core/tools/truncate.ts:1-37`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/truncate.ts)
The returned details carry the reason, total/output lines and bytes, and
partial-line flags. This is good UI and telemetry material, although it is not a
portable file revision.

### Editing

Pi has the best direct analogue to the proposed Cayu `EditFileTool`: one call
can contain multiple disjoint replacements, all matched against the original
file. The tool reads once, validates and computes all changes, then writes once.
[`pi-mono/packages/coding-agent/src/core/tools/edit.ts:287-360`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/edit.ts)
It rejects ambiguous and overlapping edits.
[`pi-mono/packages/coding-agent/src/core/tools/edit-diff.ts:295-365`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/edit-diff.ts)

The word "exact" in Pi's prompt is not a strict byte-level contract. If exact
matching fails, Pi normalizes trailing whitespace, Unicode compatibility forms,
quotes, dashes, and spaces, then applies the replacement in that normalized
space.
[`pi-mono/packages/coding-agent/src/core/tools/edit-diff.ts:26-53`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/edit-diff.ts)
[`pi-mono/packages/coding-agent/src/core/tools/edit-diff.ts:200-243`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/edit-diff.ts)
This improves success rates but makes the operation less predictable. Cayu
should keep exact matching as the default. If fuzzy matching is ever added, it
should be an explicit match policy returned in structured evidence.

Pi serializes mutations to the same resolved file path while allowing different
files to mutate concurrently.
[`pi-mono/packages/coding-agent/src/core/tools/file-mutation-queue.ts:28-60`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/file-mutation-queue.ts)
That prevents two Pi calls in one process from interleaving, but it does not
protect against another process, another Pi process, or a user/linter changing
the file, and the edit request carries no expected digest or revision.

Pi's write tool explicitly creates or overwrites and uses the same per-path
queue, but it has no stale-content precondition.
[`pi-mono/packages/coding-agent/src/core/tools/write.ts:181-225`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/write.ts)

### Search and scheduling

Pi's `find` is a real Glob analogue. It defaults to 1,000 results, excludes
`node_modules` and `.git`, and applies the shared output cap.
[`pi-mono/packages/coding-agent/src/core/tools/find.ts:20-56`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/find.ts)
[`pi-mono/packages/coding-agent/src/core/tools/find.ts:155-209`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/find.ts)
Its grep tool limits match count, truncates each line, and bounds total output.
[`pi-mono/packages/coding-agent/src/core/tools/grep.ts:24-45`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/grep.ts)
[`pi-mono/packages/coding-agent/src/core/tools/grep.ts:270-355`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/core/tools/grep.ts)
Unlike Cayu and Claude Code Grep, Pi's public grep/find schemas do not expose a
deterministic result offset.

Pi runs tool calls in parallel by default. If any call in a model-produced batch
is marked sequential, the whole batch is executed sequentially.
[`pi-mono/packages/agent/src/types.ts:254-263`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/agent/src/types.ts)
[`pi-mono/packages/agent/src/agent-loop.ts:411-425`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/agent/src/agent-loop.ts)
Otherwise it prepares calls in order and executes them concurrently, returning
tool-result messages in source order.
[`pi-mono/packages/agent/src/agent-loop.ts:489-552`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/agent/src/agent-loop.ts)

Pi's path layer deliberately expands `~` and accepts absolute paths. It is a
coding application boundary, not a contained workspace contract.
[`pi-mono/packages/coding-agent/src/utils/paths.ts:57-85`](https://github.com/badlogic/pi-mono/blob/a5afc3f171e422e08a2ccc342827719f9952f38a/packages/coding-agent/src/utils/paths.ts)

## Claude Code

Claude Code has the broadest model-visible application surface in this
comparison: Bash, optional Glob/Grep, Read, Edit, Write, NotebookEdit, web,
planning, user-input, task, agent, MCP, and other tools.
`claude-code/src/tools.ts:185-250`

### Read

The Read schema exposes absolute path, line offset/limit, and PDF page ranges.
Its structured text output includes returned lines, start line, and total lines.
`claude-code/src/tools/FileReadTool/FileReadTool.ts:227-331`
Unlike Pi, the text path uses a ranged file reader. When the caller supplies a
line limit, the normal whole-file byte gate is not applied; a token limit still
guards the returned content.
`claude-code/src/tools/FileReadTool/FileReadTool.ts:1019-1037`
The default policies are a 256-KiB total-file gate and a 25,000-token output
cap.
`claude-code/src/tools/FileReadTool/limits.ts:1-18`

Claude Code also deduplicates a repeated identical range if the file's
modification time is unchanged, returning a small `file_unchanged` result
instead of injecting the same content again.
`claude-code/src/tools/FileReadTool/FileReadTool.ts:523-573`
That is a context-cost optimization worth considering above Cayu's portable
tool contract, but it should remain app/runtime state rather than being hidden
inside a backend-neutral file mutation precondition.

### Edit and write safety

Claude Code's public Edit call performs one replacement or replace-all, not a
multi-edit array.
`claude-code/src/tools/FileEditTool/types.ts:5-34`
It requires an existing file to have been read fully, rejects a partial cached
view, and checks modification time with a content fallback.
`claude-code/src/tools/FileEditTool/FileEditTool.ts:275-343`
Immediately before writing, it repeats the synchronous read/staleness check and
then performs the synchronous write without an `await` between them.
`claude-code/src/tools/FileEditTool/FileEditTool.ts:425-491`

That is the strongest current implementation among the references, but it is
not a strict cross-process compare-and-swap. Its precondition lives in a
session-local cache containing content, timestamp, offset, and limit.
`claude-code/src/utils/fileStateCache.ts:4-22`
An arbitrary external process can still change the path between the final
check and filesystem replacement. More importantly for Cayu, this hidden state
does not compose with durable sessions, multiple application workers, or
backend-neutral workspaces.

Claude Code's Write similarly permits creation but requires a prior read before
overwriting an existing file.
`claude-code/src/tools/FileWriteTool/FileWriteTool.ts:186-221`
It repeats its state check immediately before the write.
`claude-code/src/tools/FileWriteTool/FileWriteTool.ts:247-305`

Like Pi and Codex, Claude Code performs quote/whitespace-tolerant matching in
places. Cayu's first safety-oriented tool should not silently inherit this
behavior.

Mutation results are rich: Edit returns a structured patch and can include a
Git diff.
`claude-code/src/tools/FileEditTool/types.ts:62-83`
That supports good transcript UI, but it is evidence for an individual
mutation, not a repository-wide validation step.

### Search, bounds, and scheduling

Claude Code Grep is the richest dedicated search schema in the comparison. It
has content/files/count modes, context flags, a default 250-entry head limit,
and offset paging.
`claude-code/src/tools/GrepTool/GrepTool.ts:33-108`
It excludes version-control directories and uses ripgrep's 500-column cap.
`claude-code/src/tools/GrepTool/GrepTool.ts:329-359`
Glob is simpler: it returns at most 100 results with explicit truncation, but
its public schema has no offset.
`claude-code/src/tools/GlobTool/GlobTool.ts:26-51`
`claude-code/src/tools/GlobTool/GlobTool.ts:154-195`

Claude Code has the most mature aggregate result-budget policy: ordinary tool
results persist after 50,000 characters, there is a 100,000-token absolute
limit, and one parallel tool-result message is budgeted to 200,000 characters.
`claude-code/src/constants/toolLimits.ts:1-49`

Its scheduler partitions calls into consecutive concurrency-safe batches and
single mutating barriers, with a default maximum concurrency of 10.
`claude-code/src/services/tools/toolOrchestration.ts:8-29`
`claude-code/src/services/tools/toolOrchestration.ts:84-116`
The streaming executor applies the same shared-versus-exclusive rule as calls
arrive.
`claude-code/src/services/tools/StreamingToolExecutor.ts:34-39`
`claude-code/src/services/tools/StreamingToolExecutor.ts:126-150`

Claude Code has no standalone Delete or GitChanges tool in its base inventory.
Its own Git workflow directs the model to run `git status`, `git diff`, and
`git log` through Bash, in parallel where independent.
`claude-code/src/tools/BashTool/prompt.ts:81-109`

## Codex

Codex makes a different tradeoff. Its core tool planner exposes a shell,
`apply_patch`, selected runtime utilities, image viewing, MCP, and optional
collaboration/extension tools.
[`codex/codex-rs/core/src/tools/spec_plan.rs:564-605`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/spec_plan.rs)
[`codex/codex-rs/core/src/tools/spec_plan.rs:621-663`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/spec_plan.rs)
[`codex/codex-rs/core/src/tools/spec_plan.rs:687-768`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/spec_plan.rs)

There is no first-party Glob tool. Codex instructs the model to use `rg` for
text and `rg --files` for file discovery.
[`codex/codex-rs/core/gpt_5_2_prompt.md:244-253`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/gpt_5_2_prompt.md)
There is likewise no dedicated model-visible Read, Write, or GitChanges tool in
the normal core surface; shell commands fill those roles.

### `apply_patch`

`apply_patch` is the most expressive edit interface in this comparison. Its
grammar supports adding, deleting, updating, and moving multiple files in one
call.
[`codex/codex-rs/core/src/tools/handlers/apply_patch.lark:1-19`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/handlers/apply_patch.lark)
It is a grammar-constrained freeform tool rather than a JSON replacement schema.
[`codex/codex-rs/core/src/tools/handlers/apply_patch_spec.rs:5-26`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/handlers/apply_patch_spec.rs)

For updates, Codex reads the current file and locates the supplied context, so
patch context acts as an implicit current-content precondition.
[`codex/codex-rs/apply-patch/src/lib.rs:672-709`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/apply-patch/src/lib.rs)
The match is deliberately fuzzy: exact, trailing-whitespace-insensitive,
fully-trimmed, then Unicode-punctuation-normalized.
[`codex/codex-rs/apply-patch/src/seek_sequence.rs:1-109`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/apply-patch/src/seek_sequence.rs)

Two safety properties are weaker than the coding-workspace issue's target:

1. Delete carries no content or digest precondition; the grammar only names the
   path.
2. A multi-file patch is not transactionally atomic. Hunks are applied in
   sequence, and a failure explicitly carries the mutations that were already
   committed.

The failure type documents and exposes the committed delta.
[`codex/codex-rs/apply-patch/src/lib.rs:247-272`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/apply-patch/src/lib.rs)
The application loop writes or removes one hunk at a time and returns on a
later failure.
[`codex/codex-rs/apply-patch/src/lib.rs:359-414`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/apply-patch/src/lib.rs)
[`codex/codex-rs/apply-patch/src/lib.rs:416-565`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/apply-patch/src/lib.rs)
`Add File` also records overwritten content if the path already existed, so
"add" is not a strict create-if-missing operation.

Codex does maintain a turn-level net diff for committed `apply_patch`
mutations. It tracks exact deltas without rereading the workspace and emits a
`TurnDiff` event.
[`codex/codex-rs/core/src/turn_diff_tracker.rs:47-115`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/turn_diff_tracker.rs)
[`codex/codex-rs/core/src/tools/events.rs:602-629`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/events.rs)
That is excellent UI infrastructure, but it is not a model-visible bounded
repository-inspection tool and does not replace explicit validation with Git.

### Shell bounds and scheduling

Codex's unified exec schema lets each call set a model-output token budget and
returns structured exit code, original token count, elapsed time, and possibly
truncated output.
[`codex/codex-rs/core/src/tools/handlers/shell_spec.rs:21-110`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/handlers/shell_spec.rs)
[`codex/codex-rs/core/src/tools/handlers/shell_spec.rs:264-295`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/handlers/shell_spec.rs)
There is also a context-level tool-output token budget in configuration.
[`codex/codex-rs/core/src/config/mod.rs:856-864`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/config/mod.rs)

Shell execution opts into parallel calls.
[`codex/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs:80-103`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs)
Tool executors default to non-parallel.
[`codex/codex-rs/tools/src/tool_executor.rs:44-68`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/tools/src/tool_executor.rs)
At runtime, parallel-capable calls take a shared read lock and other tools take
an exclusive write lock, producing an ordering barrier.
[`codex/codex-rs/core/src/tools/parallel.rs:94-156`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/src/tools/parallel.rs)

Codex also invests heavily in prompt-level execution discipline: persist
through implementation, validate work, prefer `rg`, use `apply_patch`, and
parallelize independent reads.
[`codex/codex-rs/core/gpt_5_2_prompt.md:29-32`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/gpt_5_2_prompt.md)
[`codex/codex-rs/core/gpt_5_2_prompt.md:109-150`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/gpt_5_2_prompt.md)
[`codex/codex-rs/core/gpt_5_2_prompt.md:244-289`](https://github.com/openai/codex/blob/6e5a2d6b8d148a5554fdceb6f399ca45bd1c78d9/codex-rs/core/gpt_5_2_prompt.md)
That explains some of Codex's effectiveness despite its smaller dedicated
filesystem surface; it does not remove the value of safer runtime primitives
for less specialized agents.

## What Cayu should borrow

### From Pi

- Multiple disjoint replacements in one call.
- Match all edits against the original file.
- Reject ambiguity and overlap before any write.
- Compute once and write once.
- Return compact diff/truncation details.
- Serialize mutations to the same workspace path.

Do not copy Pi's implicit fuzzy fallback or lack of a stale-content
precondition.

### From Claude Code

- Require the agent to establish a current file view before destructive
  overwrite or edit.
- Recheck staleness immediately before mutation.
- Return structured patches suitable for a rich transcript UI.
- Add read-range continuation and duplicate-read suppression at the
  application/runtime layer.
- Enforce aggregate result budgets across parallel calls, not just per-tool
  limits.

Do not encode the safety contract solely in session-local `mtime` state. It
does not survive all durable/multi-worker Cayu execution patterns.

### From Codex

- Keep the mutation interface compact and expressive.
- Make add, update, move, and delete intent explicit.
- Treat mutating calls as parallel-ordering barriers.
- Emit a durable turn/interaction diff for UI consumers.
- Preserve command exit status and pre-truncation output size.

Do not copy `apply_patch`'s partial-commit semantics, fuzzy matching as the only
mode, delete-without-precondition, or add-that-can-overwrite behavior.

## Required correction to the coding-workspace issue

The issue currently says tools accept an expected SHA-256 digest, but it does
not explicitly require the workspace to perform the check and mutation as one
conditional operation.

The current `Workspace` API cannot provide that guarantee. It has separate
`read_bytes`, `write_bytes`, and `delete` calls.
`src/cayu/workspaces/base.py:200-225`
Runtime `parallel_safe=False` only orders calls inside the relevant Cayu tool
round. It does not prevent another session, worker, command, linter, or user
from modifying the same file between a digest check and a later write.

The contract should add workspace-level conditional primitives, conceptually:

```python
class Workspace:
    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult: ...

    async def create_bytes(
        self,
        path: str,
        content: bytes,
    ) -> WorkspaceMutationResult: ...

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult: ...

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult: ...
```

`WorkspaceReadResult` should carry:

- the actual byte `offset`;
- returned and total bytes;
- `next_offset`;
- truncation reason;
- an opaque `revision` suitable for a later conditional mutation; and
- a SHA-256 digest when the backend has a complete content snapshot or can
  calculate it within its bounded operation.

`WorkspaceMutationResult` should carry:

- operation (`create`, `replace`, or `delete`);
- before/after revisions and SHA-256 values where applicable;
- before/after byte counts; and
- whether the backend used its strong conditional-mutation capability.

The concurrency domain must be explicit. A conforming strong backend should
guarantee that no other mutation through that workspace resource can interleave
between comparison and replacement. For local/runner filesystems, this likely
requires a shared lock keyed by `(resource_key, path)`, digest verification
inside the critical section, a temporary file, and atomic rename. Arbitrary
non-cooperating external processes may remain outside that guarantee unless the
backend can provide a stronger primitive; that limitation must be documented
rather than calling a two-call read/write sequence atomic.

An opaque revision is preferable to making every backend and every page read
hash an entire large file. SHA-256 should remain durable evidence and can be the
revision implementation for simple backends, but the compare-and-swap contract
should not require downloading the complete file into the Cayu process.

## Recommended tool contracts

### `ReadFileTool`

- Add byte `offset` plus bounded `max_bytes`.
- Return actual range, total size, `next_offset`, revision, optional SHA-256,
  encoding, and explicit truncation reason.
- Preserve image/PDF artifact behavior.
- Keep model content and structured evidence independently bounded.
- Optionally let Cayu Code deduplicate an unchanged repeated range, following
  Claude Code, without making hidden session state the mutation precondition.

### `EditFileTool`

- One existing UTF-8 file per call.
- One or more exact, non-overlapping replacements, all resolved against one
  original revision.
- Explicit expected replacement count per edit.
- No fuzzy matching by default.
- Compute and validate the full result before mutation.
- Call one workspace conditional replace.
- Return before/after revision and digest, replacement counts, byte counts, and
  a bounded unified diff with `truncated`, reasons, and continuation metadata.
- "Atomic" means one file. Cross-file transactions are out of scope unless the
  Workspace contract later adds them.

### `WriteFileTool`

- Require an explicit `mode: "create" | "overwrite"`.
- Create uses create-if-missing.
- Overwrite requires the current revision/digest and uses conditional replace.
- Do not retain blind overwrite as the default.

### `DeleteFileTool`

- Delete one workspace-relative regular file only.
- Require expected revision/digest.
- Use one conditional delete.
- Reject missing paths, directories, stale revisions, traversal, and truncated
  or incomplete precondition state.

This would be safer than Codex delete and more reviewable than shell `rm`.

### `GitChangesTool`

No reference implementation provides the exact proposed capability, so Cayu
should design it for durable evidence rather than mimic shell output:

- `status` mode: parse `git status --porcelain=v2 -z`, return bounded structured
  tracked/untracked/renamed/conflicted entries.
- `summary` mode: return deterministic per-file additions, deletions, binary
  state, and staging state.
- `diff` mode: return bounded text patches by file/hunk; never inline binary
  data or invoke external diff/textconv helpers.
- Page by stable file/hunk cursor rather than arbitrary output bytes.
- Use process-form runner commands, `--no-ext-diff`, `--no-textconv`,
  `--no-color`, and `--` path boundaries.
- Report repository root, base/index/worktree scope, truncation reasons, and
  continuation cursor in structured output.
- Keep it read-only (`ToolEffect.NONE`) and parallel-safe.

As a later runtime/UI enhancement, Cayu can also emit interaction-level net
change evidence similar to Codex's `TurnDiff` event. That is complementary:
the event helps the user see what happened; `GitChangesTool` lets the model
prove what remains before declaring completion.

## Delivery order

1. Amend the coding-workspace issue to require workspace-level conditional mutation and define
   the concurrency domain.
2. Extend `WorkspaceReadResult` and add conditional create/replace/delete
   primitives with conformance tests across every built-in backend.
3. Upgrade `ReadFileTool` and `WriteFileTool`.
4. Add exact multi-edit and conditional delete.
5. Add bounded `GitChangesTool`.
6. Add aggregate tool-result budgeting and optional unchanged-read dedup as
   separate follow-up work.
7. Have Cayu Code consume the branch, register the tools, require Git-change
   inspection plus validation before completion, and render the structured
   evidence per interaction.

## Bottom line

The proposed five-tool set is the right set.

- `ListFilesTool` already is Cayu's Glob.
- `SearchTextTool` is already at or near Claude Code's search quality and is
  stronger than Pi on deterministic paging and durable truncation metadata.
- `EditFileTool`, safer `WriteFileTool`, pageable `ReadFileTool`, and
  `DeleteFileTool` close real gaps.
- `GitChangesTool` is not redundant; it is a novel first-party validation and
  observability primitive.

The important change is architectural: do not implement stale protection as a
tool-level read/check/write sequence. Put conditional mutation in `Workspace`,
then make the tools thin, bounded, structured clients of that contract. That is
how Cayu turns a current weakness into a runtime-level advantage over Pi,
Claude Code, and Codex.
