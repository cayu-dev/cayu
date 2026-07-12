from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples._advanced_support import (
    ScenarioResult,
    advanced_run_limits,
    collect_events,
    count_model_completions,
    fork_session,
    session_evidence,
    stable_output_spec,
    validated_output,
)
from examples._advanced_support.fake_github import FakeGitHubServer, GitHubClient
from examples.repo_maintainer_tournament.candidate_gates import (
    apply_candidate,
    run_candidate_gates,
    write_baseline,
)

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ResumeRequest,
    RunRequest,
    StructuredOutputSpec,
)
from cayu.providers import ModelProvider

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["minimal-boundary-check", "weaken-tests", "broad-rewrite"],
        },
        "rationale": {"type": "string"},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "enum": ["calculator.py", "test_calculator.py"],
                    },
                    "content": {"type": "string", "minLength": 1, "maxLength": 2000},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 2,
        },
    },
    "required": ["strategy", "rationale", "changes"],
    "additionalProperties": False,
}
EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "winner": {"type": "string"},
        "rejected": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "reason": {"type": "string"},
    },
    "required": ["winner", "rejected", "reason"],
    "additionalProperties": False,
}

FIXTURE_DIR = Path(__file__).with_name("fixtures")
BASELINE = {
    relative: (FIXTURE_DIR / relative).read_text(encoding="utf-8")
    for relative in ("calculator.py", "test_calculator.py")
}


@dataclass(frozen=True)
class RepositorySourceContext:
    pull: dict[str, Any]
    files: list[dict[str, Any]]
    baseline: dict[str, str]


async def run_scenario(
    root: Path,
    *,
    provider: ModelProvider,
    model: str,
    mode: str,
    source_context: RepositorySourceContext | None = None,
) -> ScenarioResult:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    role_prompts = {
        "maintainer-source": "Prepare the issue and repository context.",
        "minimal": (
            "Return strategy='minimal-boundary-check'. Produce the complete corrected "
            "calculator.py and do not change tests."
        ),
        "test-weakener": (
            "Intentionally return strategy='weaken-tests' so the evaluator can reject it."
        ),
        "broad": (
            "Return strategy='broad-rewrite'. Produce a complete calculator.py using a custom "
            "exception and validation helper."
        ),
        "evaluator": ("Select the smallest passing production patch and reject any changed tests."),
    }
    for role, role_prompt in role_prompts.items():
        app.register_agent(
            AgentSpec(
                name=role,
                model=model,
                system_prompt=("Repair divide-by-zero behavior. " + role_prompt),
            )
        )

    with FakeGitHubServer() as server:
        github = GitHubClient(server.base_url)
        if source_context is None:
            pull = await github.get_pull("acme", "calculator", 1)
            files = await github.list_pull_files("acme", "calculator", 1)
            baseline = BASELINE
        else:
            pull = source_context.pull
            files = source_context.files
            baseline = source_context.baseline
        causal_budget_id = "advanced-repo-tournament-budget"
        source_id = "repo-source"
        await collect_events(
            app.run(
                RunRequest(
                    agent_name="maintainer-source",
                    session_id=source_id,
                    causal_budget_id=causal_budget_id,
                    messages=[Message.text("user", f"Prepare repair context: {pull!r}; {files!r}")],
                    limits=advanced_run_limits(),
                )
            )
        )

        branch_roles = {
            "minimal": ("minimal", "minimal-boundary-check"),
            "test-weakener": ("test-weakener", "weaken-tests"),
            "broad": ("broad", "broad-rewrite"),
        }
        candidates: dict[str, dict[str, Any]] = {}
        for role, (agent_name, expected_strategy) in branch_roles.items():
            session_id = f"repo-{role}"
            await fork_session(
                app,
                source_session_id=source_id,
                session_id=session_id,
                agent_name=agent_name,
            )
            events = await collect_events(
                app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[
                            Message.text(
                                "user",
                                (
                                    f"Propose and write the {role} candidate. Apply your own "
                                    f"repair to these workspace files: {baseline!r}"
                                ),
                            )
                        ],
                        structured_output=StructuredOutputSpec(
                            name=f"{role}-candidate",
                            json_schema=_candidate_schema(expected_strategy),
                            max_retries=2,
                            repair_prompt=(
                                "Return the assigned candidate strategy exactly as specified by "
                                "the schema. This branch exists to be independently evaluated."
                            ),
                        ),
                        limits=advanced_run_limits(),
                    )
                )
            )
            candidates[role] = validated_output(events)

        workspace_root = root / ".cayu-example-workspaces" / f"repo-{uuid4().hex}"
        gates: dict[str, dict[str, Any]] = {}
        workspace_by_strategy: dict[str, Path] = {}
        for candidate in candidates.values():
            strategy = candidate["strategy"]
            workspace = workspace_root / strategy
            workspace.mkdir(parents=True)
            write_baseline(workspace, baseline)
            apply_candidate(workspace, candidate, baseline)
            gates[strategy] = run_candidate_gates(workspace, baseline)
            workspace_by_strategy[strategy] = workspace

        evaluator_id = "repo-evaluator"
        await fork_session(
            app,
            source_session_id=source_id,
            session_id=evaluator_id,
            agent_name="evaluator",
        )
        evaluator_events = await collect_events(
            app.resume(
                ResumeRequest(
                    session_id=evaluator_id,
                    messages=[Message.text("user", f"Select from gate evidence: {gates!r}")],
                    structured_output=stable_output_spec(
                        "repo-tournament-evaluation", EVALUATION_SCHEMA
                    ),
                    limits=advanced_run_limits(),
                )
            )
        )
        evaluation = validated_output(evaluator_events)
        winner = evaluation["winner"]
        promoted = workspace_root / "promoted"
        shutil.copytree(workspace_by_strategy[winner], promoted)

        created = await github.ensure_pull(
            "acme",
            "calculator",
            title="Fix divide by zero",
            head="cayu/fix-divide",
            base="main",
            body=f"Promoted {winner} after deterministic and evaluator gates.",
        )
        retried = await github.ensure_pull(
            "acme",
            "calculator",
            title="Fix divide by zero",
            head="cayu/fix-divide",
            base="main",
            body=f"Promoted {winner} after deterministic and evaluator gates.",
        )

    sessions = await session_evidence(
        app,
        {
            source_id: "source",
            "repo-minimal": "minimal",
            "repo-test-weakener": "test-weakener",
            "repo-broad": "broad",
            evaluator_id: "evaluator",
        },
    )
    eligible = [
        (strategy, gate["diff_lines"])
        for strategy, gate in gates.items()
        if gate["tests_passed"] and not gate["test_files_changed"]
    ]
    expected_winner = min(eligible, key=lambda item: item[1])[0]
    assertions = {
        "candidate_changes_applied": all(
            all(
                (workspace_by_strategy[candidate["strategy"]] / change["path"]).read_text(
                    encoding="utf-8"
                )
                == change["content"]
                for change in candidate["changes"]
            )
            for candidate in candidates.values()
        ),
        "evaluator_rejected_test_weakening": "weaken-tests" in evaluation["rejected"],
        "exactly_one_pull_request": (
            len(server.state.created_pulls) == 1 and created["number"] == retried["number"]
        ),
        "fake_github_api_exercised": (
            server.state.list_pull_requests == 2 and server.state.create_pull_requests == 1
        ),
        "smallest_correct_patch_selected": winner == expected_winner == "minimal-boundary-check",
        "workspaces_are_isolated": (
            len({path.resolve() for path in workspace_by_strategy.values()}) == 3
            and (promoted / "calculator.py").read_text(encoding="utf-8")
            == (workspace_by_strategy[winner] / "calculator.py").read_text(encoding="utf-8")
        ),
    }
    model_requests = await count_model_completions(
        app, [session.session_id for session in sessions]
    )
    result = ScenarioResult(
        scenario="repo-maintainer-tournament",
        mode=mode,
        status="verified" if all(assertions.values()) else "failed",
        assertions=assertions,
        sessions=sessions,
        provider_name=provider.name,
        model=model,
        metrics={
            "model_requests": model_requests,
            "winner": winner,
            "pull_request_number": created["number"],
            "github_create_requests": server.state.create_pull_requests,
            "github_list_requests": server.state.list_pull_requests,
            "github_created_pulls": len(server.state.created_pulls),
        },
        outputs={"candidates": candidates, "gates": gates, "evaluation": evaluation},
    )
    result.write(root)
    result.require_verified()
    return result


def _candidate_schema(expected_strategy: str) -> dict[str, Any]:
    schema = deepcopy(CANDIDATE_SCHEMA)
    schema["properties"]["strategy"]["enum"] = [expected_strategy]
    return schema
