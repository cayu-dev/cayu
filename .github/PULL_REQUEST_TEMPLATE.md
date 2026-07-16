## What does this PR do?

<!-- What problem does it solve, and why is this approach the right one? -->

Fixes #

## Type of change

<!--
Suspected exploitable vulnerability? Do not open a public PR. Report it privately at
https://github.com/cayu-dev/cayu/security/advisories/new first. "Security hardening"
below is for preventative changes that do not disclose a suspected vulnerability.
-->

- [ ] 🐛 Bug fix
- [ ] ✨ Feature
- [ ] 🔒 Security hardening
- [ ] ♻️ Refactor (no behavior change)
- [ ] 📝 Docs / examples
- [ ] ✅ Tests only

## How to verify

<!-- Repro steps for bugs (before/after), usage example for features. -->

1.

## Checklist

- [ ] I searched [open and closed PRs](https://github.com/cayu-dev/cayu/pulls?q=is%3Apr) — this isn't a duplicate
- [ ] The PR is one logical change (no unrelated commits or drive-by refactors)
- [ ] `uv run pytest` passes locally
- [ ] `uv run ruff check` + `uv run ruff format --check` + `uv run ty check` pass (see [CONTRIBUTING](https://github.com/cayu-dev/cayu/blob/main/CONTRIBUTING.md#lint-and-types) for paths)
- [ ] Bug fix → includes a regression test that fails without the fix
- [ ] Feature / contract change → docs updated (`README`, `docs/`, `docs/runtime-contracts.md`) — or N/A
- [ ] Touches stores / runners / egress → ran the relevant Docker/Postgres test tier locally — or N/A
- [ ] Touches the dashboard → ran its lint, test, typecheck, API-contract, and packaged-asset build checks — or N/A
- [ ] New dependency → justified and constrained to Cayu's verified compatibility range — or N/A
- [ ] New integration or backend adapter → I've read the [placement policy](https://github.com/cayu-dev/cayu/blob/main/CONTRIBUTING.md#placement-policy-what-lands-in-tree) and this belongs in-tree
