---
name: cayu-cloud
description: Operate Agents on Cayu Cloud with the `cayu cloud` CLI and the `cayu` commands that read a deployed Agent. Use when asked to deploy a Cayu Agent, check its status or logs, roll back or promote a release, change its environment, settings or schedules, read its sessions or tasks, or settle its inbox approvals and questions. Not for building an Agent's code; `cayu guide` covers that.
---

# Cayu Cloud from the command line

Cayu Cloud hosts Cayu Agent applications. Everything an operator does there can be done with
the `cayu` CLI: `cayu cloud` for the Agent's releases and infrastructure, and `cayu
session`, `cayu task` and `cayu inbox` with `--application NAME` for what runs inside it.
This page is the workflow. Each command's `--help` has its options, and `cayu cloud --help`
lists every command; read those instead of guessing a flag.

## 1. Use the CLI the project uses

Run `cayu` from the Agent's repository, through its environment (`uv run cayu ...` in a uv
project), so the CLI is the version the Agent pins. Then:

- `cayu guide cloud` prints this page as that version ships it. If it differs from what you
  are reading, follow the printed one.
- `cayu guide` lists the other package-shipped guides, for the Agent's own code.
- `docs/cayu-cloud.md` in the Cayu source repository explains each command at length.

Commands print a JSON envelope (`operation`, `result`, and `error` with a stable
`error.category` when one fails; a refusal exits `2`). Commands that print readable text by
default take `--json`. Read the category and the message before retrying: most refusals say
what to run instead.

## 2. Know where a command acts

A login acts in one Cayu Cloud install and one Organization. Before anything else:

```console
cayu cloud whoami          # who you are and the Organization this login acts in
cayu cloud organizations   # your Organizations, your role in each, and the current one
cayu cloud context show    # the selected context file, its Cloud and how it signs in
cayu cloud doctor          # whether sign-in and the API work
```

Signing in is for the person, not for you. `cayu cloud login --no-browser` prints a URL and a
code for them to finish in their browser; `cayu cloud login --api-url URL` signs in to
another install (a staging one, say) and writes and selects a context file for it. Never
read, print or copy a login, an API key file or a context's key.

**Before any change, confirm where it will land, then pin it there.** The selected context and
the saved login are shared state that another session, `cayu cloud context use` or
`CAYU_CLOUD_CONTEXT` can change under you.

1. Run `cayu cloud whoami` (and `cayu cloud context show` when a context is in use) and check
   with the person that this is the Cloud and Organization they mean.
2. On every `cayu cloud` command that changes something, pass `--organization ORG_ID`. With a
   login it acts in that Organization for that one command, without changing the login's own
   (so never switch with `cayu cloud organizations use` to act elsewhere: that changes it for
   every session on the machine). An Organization API key acts only in its own, and naming
   another is refused (`organization_mismatch`). Reads take it too, right after `cayu cloud`.
3. Pass `--context PATH` too whenever the Cloud you mean has a context file. It goes right
   after `cayu cloud`, before the command; on `cayu inbox`, `cayu session` and `cayu task` it
   goes after the command, with `--application`.

```console
cayu cloud --context PATH deploy . --organization ORG_ID
```

- A plain `cayu cloud login` to the default Cloud leaves no context file, so there step 3
  doesn't apply: rely on step 1 and `--organization`.
- `cayu inbox`, `cayu session` and `cayu task` take no `--organization`. Pin them with
  `--context` when there is one, and repeat step 1 right before a change.
- Take the context path and the Organization from the person, or from `context show` and
  `whoami` once the person has confirmed that is where they mean. Don't make a context file
  yourself.
- Each Agent remembers where it was first deployed from this machine. A change to it from
  another Organization or Cloud is refused with `agent_organization_mismatch`, and the error
  names the Organization to pass. Don't work around it; ask which one is meant.

Changing commands say on standard error which Organization and Cloud they act in. Reads say
nothing. `docs/cayu-cloud.md` lists which commands change something.

## 3. Common tasks

`--application` takes the Agent's slug (the `application` in its `cayu-cloud.toml`) or its
name. The examples use an Agent called `support-bot`.

### Find the Agent and what it runs

```console
cayu cloud applications list
cayu cloud applications show support-bot
cayu cloud service status --application support-bot
```

`applications show` is the selected release, when it last served a request and any candidate
or in-flight release. `service status` is the web process, worker, schedules and endpoint.

### Deploy

```console
cayu cloud init --check .        # is cayu-cloud.toml valid, without a login or an upload
cayu cloud --context PATH deploy . --organization ORG_ID
```

`deploy` runs the deploy check, uploads the project, waits for the release to build, pass its
smoke test and go live, and reports the release. A GitHub repository also works as
the source, at an exact commit: `deploy https://github.com/example/app.git --revision SHA`.
`--no-promote` stages a candidate release that changes nothing until someone runs `deployment
promote`. When a wait runs out while Cloud is still working (`deployment_still_running`,
`service_still_starting`), nothing failed: follow it with the commands the error prints.
Unfinished sessions on the serving release hold the new one back (`--session-policy`), and a
breaking storage revision needs `--acknowledge-breaking`; both are the person's decision, so
report them and ask before passing those flags.

### Releases and rollback

```console
cayu cloud deployment list --application support-bot
cayu cloud deployment status DEPLOYMENT_ID
cayu cloud deployment timeline DEPLOYMENT_ID
cayu cloud deployment logs DEPLOYMENT_ID
cayu cloud --context PATH rollback DEPLOYMENT_ID --application support-bot --wait --organization ORG_ID
cayu cloud --context PATH deployment promote DEPLOYMENT_ID --application support-bot --organization ORG_ID
```

`timeline` shows where a release stopped and whether it can be retried; `logs` is its build
and publication activity. `rollback` selects an earlier release from `deployment list`.

### Logs and health

```console
cayu cloud service logs --application support-bot --process worker --since 1h --grep Traceback
cayu cloud service check --application support-bot
cayu cloud service database --application support-bot
```

`service logs` filters by `--process` (`web`, `worker`, `migrate`, `schedule:NAME`), time
(`--since`, `--until`), text (`--grep`) and JSON fields (`--event`, `--field`), and follows
with `--follow`. It is the first place to look when a task or schedule fails. `service
restart`, `redeploy`, `pause`, `resume`, `sleep` and `wake` change the running service; see
`cayu cloud service --help`.

### Environment, settings and schedules

```console
cayu cloud env list --application support-bot
cayu cloud --context PATH env set LOG_LEVEL=debug --application support-bot --organization ORG_ID
cayu cloud settings list --application support-bot
cayu cloud --context PATH settings set KEY=VALUE --application support-bot --organization ORG_ID
cayu cloud schedule list --application support-bot
cayu cloud --context PATH schedule run NAME --application support-bot --organization ORG_ID
```

`env set` and `env unset` restart the Agent's service (with `--scope eval` they restart
nothing). A secret goes in with `env set NAME --secret
--value-file PATH`, never on the command line, and `env list` never shows its value.
`settings` changes the Agent's managed settings without a deploy, and `settings unset`
gives a setting back to the release. Schedules themselves live in `cayu-cloud.toml`
(`[[schedules]]`) and change with a deploy; `schedule run` starts one now, and its output is
in `service logs --process schedule:NAME`.

### Sessions and tasks

```console
cayu session list --application support-bot --context PATH
cayu session show SESSION_ID --application support-bot --context PATH
cayu session transcript SESSION_ID --application support-bot --context PATH
cayu task list --application support-bot --context PATH
cayu task show TASK_ID --application support-bot --context PATH
```

These read the Agent's own Cayu API with your login and wake a sleeping Agent first
(`--no-wait` fails instead). `cayu session --help` and `cayu task --help` list the other
reads (events, usage, tools, output). Commands there that change a session or task, such as
`session close` or `task cancel`, need the person's go-ahead like any other change.

### Inbox and approvals

```console
cayu inbox list --application support-bot --context PATH
cayu inbox show ITEM_ID --application support-bot --context PATH
cayu inbox approve ITEM_ID --application support-bot --context PATH --reason "..."
cayu inbox deny ITEM_ID --application support-bot --context PATH --reason "..."
cayu inbox answer ITEM_ID "TEXT" --application support-bot --context PATH
cayu inbox approve ITEM_ID --remember --application support-bot --context PATH
cayu grants list --application support-bot --context PATH
cayu grants revoke GRANT_ID --application support-bot --context PATH --reason "..."
```

The inbox holds what the Agent waits on a person for: tool approvals, questions, recovery
items and knowledge proposals. Approving runs the tool call the Agent proposed, so approve,
deny, answer or dismiss only what the person has decided; run `inbox show` first, since a
decision is bound to the review it issues. `approve --remember` also makes a grant: later
calls of the same kind run without asking, until `cayu grants revoke`. Remember an approval
only when the person asked for exactly that, and say what the grant covers
(`cayu grants show`). An Agent whose Release hosts no grants answers `grants_not_served`.

## 4. Before and after a change

- Before: read the current state (`applications show`, `service status`, `env list` or
  `settings list`), so you can say what changed and undo it.
- Confirm where it lands and pin it as section 2 says (`--organization` on `cayu cloud`
  changes, `--context` when there is one), and run only the change you were asked for.
- After: check that it took (`service status`, `deployment status`) and that the Agent
  still works (`service logs`, `service check`, a task or schedule finishing), not only that
  the command succeeded. Report what you ran and what you checked.
- Don't archive an Agent, destroy its service, revoke API keys or remove connections unless
  the person asked for exactly that.

## Installing this skill

This page is the `cayu-cloud` skill of the `cayu` plugin, which ships in the `cayu` package.
`cayu guide --plugin-dir` prints the plugin's directory in the installed package, a plugin
marketplace for Claude Code and Codex, so the skill matches the CLI that project installs.
From the Agent's repository:

```console
claude plugin marketplace add "$(uv run cayu guide --plugin-dir)"
claude plugin install cayu@cayu
codex plugin marketplace add "$(uv run cayu guide --plugin-dir)"
```

In Codex, then enable `cayu@cayu` (`[plugins."cayu@cayu"]` with `enabled = true` in
`.codex/config.toml`). The skill also works on its own: copy the `cayu/skills/cayu-cloud`
folder of that directory into `.claude/skills/` or `.agents/skills/`. A copy doesn't follow
upgrades of `cayu`; the first step above (`cayu guide cloud`) catches that.
