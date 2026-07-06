"""Flagship recipe: a cloud PR-review agent, composed end-to-end on cayu.

This file stays intentionally small: it is the runnable map for the recipe.
The implementation lives in neighboring modules that show where real agent code
belongs as the app grows:

- ``github_tools.py``: GitHub REST tools and credential-proxy egress.
- ``qa_policy.py``: the command allowlist for end-to-end QA.
- ``workspace.py``: per-review checkout/workspace construction.
- ``reviewer_app.py``: app, agent, provider, and tool assembly.
- ``worker.py``: durable task enqueueing and worker handling.
- ``webhook.py``: external GitHub webhook ingress.
- ``demo.py``: deterministic no-key demo plus live-review entrypoint.

RUN IT
------
    PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py

    OPENAI_API_KEY=... GITHUB_TOKEN=... \
      PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py --live owner/repo#123
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.github_pr_reviewer.demo import demo, review_pr  # noqa: E402
from examples.github_pr_reviewer.github_tools import (  # noqa: E402
    GITHUB_API,
    DemoPRDiffTool,
    GetPRDiffTool,
    PostPRCommentTool,
)
from examples.github_pr_reviewer.qa_policy import (  # noqa: E402
    _ALLOWED_QA_COMMANDS,
    _DENYLISTED_TOKENS,
    QaCommandPolicy,
)
from examples.github_pr_reviewer.reviewer_app import (  # noqa: E402
    REVIEWER_SYSTEM_PROMPT,
    build_app,
    build_provider,
)
from examples.github_pr_reviewer.webhook import build_webhook_app  # noqa: E402
from examples.github_pr_reviewer.worker import (  # noqa: E402
    _handle_pr_review_task,
    enqueue_pr_review,
    run_pr_review_worker_once,
)
from examples.github_pr_reviewer.workspace import PRReviewWorkspaceFactory  # noqa: E402

__all__ = [
    "GITHUB_API",
    "REVIEWER_SYSTEM_PROMPT",
    "_ALLOWED_QA_COMMANDS",
    "_DENYLISTED_TOKENS",
    "DemoPRDiffTool",
    "GetPRDiffTool",
    "PRReviewWorkspaceFactory",
    "PostPRCommentTool",
    "QaCommandPolicy",
    "_handle_pr_review_task",
    "build_app",
    "build_provider",
    "build_webhook_app",
    "demo",
    "enqueue_pr_review",
    "main",
    "review_pr",
    "run_pr_review_worker_once",
]


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "--live":
        spec = argv[1]
        owner_repo, _, num = spec.partition("#")
        owner, _, repo = owner_repo.partition("/")
        asyncio.run(review_pr(owner, repo, int(num)))
    else:
        asyncio.run(demo())


if __name__ == "__main__":
    main(sys.argv[1:])
