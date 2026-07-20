# Repo Maintainer Tournament

Part of Cayu's [advanced runtime example suite](../ADVANCED_RUNTIME_EXAMPLES.md).
See [Advanced runtime strategies](../../docs/advanced-runtime-examples.md) for
measured observations and proof boundaries.

Three repair agents fork from one issue context into isolated local workspaces.
Each repair branch emits complete file changes which are applied to that branch's
workspace. Deterministic tests and diff-policy gates expose a test-weakening
candidate, and an evaluator selects the smallest correct production patch. Only
the promoted winner reaches the GitHub boundary.

The bundled fake GitHub service is a real loopback HTTP API. The same client
methods map to GitHub's pull-request endpoints. Recovery first lists an existing
open pull by head/base and creates one only when no matching receipt exists.

Live mode can additionally replay every candidate in real Git worktrees and
promote the winner to an actual GitHub repository. Set the repository and source
pull request explicitly; the GitHub token is used only for API calls, while the
configured Git remote handles clone and push authentication.

```bash
uv run python -m examples.repo_maintainer_tournament.app
# Gemini
GEMINI_API_KEY=... uv run python -m examples.repo_maintainer_tournament.app --mode live --provider gemini
# OpenAI
OPENAI_API_KEY=... uv run python -m examples.repo_maintainer_tournament.app --mode live --provider openai
# Claude
ANTHROPIC_API_KEY=... uv run python -m examples.repo_maintainer_tournament.app --mode live --provider anthropic

# Real repository boundary
export CAYU_REPO_MAINTAINER_REPOSITORY=owner/repository
export CAYU_REPO_MAINTAINER_SOURCE_PULL=1
export GITHUB_TOKEN=...
OPENAI_API_KEY=... uv run python -m examples.repo_maintainer_tournament.app \
  --mode live --provider openai --trials 1
```

The real-repository envelope asserts that all candidate changes were replayed in
isolated Git worktrees, deterministic gates chose the same winner, the winning
commit exists on the remote branch, a real PR reports the expected head SHA and
files, and retrying PR creation returns the same single open PR.
