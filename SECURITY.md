# Security Policy

Cayu is an agent runtime: it executes model-directed commands, injects secrets into
tool calls, and enforces policy, taint, budget, and egress boundaries. Holes in those
boundaries are security vulnerabilities, and we treat them as the highest-priority class
of bug in the project.

## Reporting a vulnerability

**Do not open a public issue, pull request, discussion, or Discord thread for a
suspected exploitable problem, and do not publish the details before disclosure is
coordinated.**

Report it privately via GitHub's private vulnerability reporting: go to the repo's
**Security** tab → **Report a vulnerability**, or use this direct link:

> https://github.com/cayu-dev/cayu/security/advisories/new

No email or account beyond GitHub is required. Reports go only to the maintainers, and
GitHub provides a private thread for follow-up questions and coordinated disclosure.

Please include:

- A description of the boundary being bypassed (see the list below).
- A minimal reproduction — a failing test or short script is ideal.
- The cayu version/commit and any relevant configuration (provider, runner, store).

## What qualifies

Anything that lets an agent, a model response, or tool output do what the configured
contracts say it must not:

- **Command-policy bypass** — executing commands a policy should have denied, including
  argument-smuggling around allowlists and selector rules.
- **Sandbox / runner isolation escape** — reaching the host, other sessions, or
  cloud metadata endpoints from inside a runner that promises isolation.
- **Egress-control bypass** — network traffic escaping enforced virtual egress, or
  credentials reaching destinations that were never approved.
- **Secret exposure** — injected secrets appearing unredacted in transcripts, tool
  results, artifacts, logs, or durable stores.
- **Taint-boundary escape** — tainted content crossing a boundary without the policy
  consequences the taint contract specifies.
- **Budget/approval enforcement bypass** — spend or side effects continuing after a
  limit tripped or an approval was denied.
- **Server/dashboard issues** — authentication or authorization flaws in `cayu.server`.

Ordinary bugs (crashes, wrong results, flaky tests) without a security consequence
should go through the normal public issue tracker.

## What to expect

- **Acknowledgement within 3 business days.**
- We investigate, develop a fix, and coordinate a disclosure timeline with you —
  typically releasing the fix before publishing details.
- Credit is given in the advisory unless you prefer otherwise.
- There is currently no paid bounty program.

## Supported versions

Cayu is pre-1.0. Security fixes land on `main` and ship in the next release; only the
**latest released version** is supported. If you're pinned to an older 0.x release,
upgrade before reporting anything already fixed.
