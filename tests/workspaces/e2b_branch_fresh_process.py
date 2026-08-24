"""Fresh-process half of the gated E2B workspace-branch proof."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tests.workspaces.remote_branch_live_support import (
    FileWorkspaceBranchAuthorityProvider,
)

from cayu import (
    E2BRunner,
    RunnerWorkspace,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchRecoveryRequest,
)


async def _run(args: argparse.Namespace) -> None:
    provider = FileWorkspaceBranchAuthorityProvider(Path(args.authority_path))
    runner = await E2BRunner.from_existing(args.sandbox_id, close_action="none")
    try:
        workspace = RunnerWorkspace(
            runner,
            workspace_id="e2b-live-workspace",
            enable_workspace_branches=True,
            branch_authority_resolver=provider,
        )
        recovered = await workspace.recover_branch(
            WorkspaceBranchRecoveryRequest(
                branch_id=args.branch_id,
                session_id=args.session_id,
                expected_run_epoch=args.run_epoch,
                binding_generation=args.binding_generation,
                binding_identity=args.binding_identity,
                recovery_id="fresh-process-recovery",
            )
        )
        if recovered.branch is None:
            raise RuntimeError("Fresh process did not recover an open workspace branch.")
        result = await recovered.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=args.branch_id,
                baseline_revision=args.baseline_revision,
                change_set_digest=args.change_set_digest,
                idempotency_key="fresh-process-publication",
                expected_run_epoch=args.run_epoch,
                binding_generation=args.binding_generation,
            )
        )
        print(json.dumps({"status": result.status.value}, sort_keys=True))
    finally:
        await runner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("authority_path")
    parser.add_argument("sandbox_id")
    parser.add_argument("branch_id")
    parser.add_argument("session_id")
    parser.add_argument("run_epoch", type=int)
    parser.add_argument("binding_generation")
    parser.add_argument("binding_identity")
    parser.add_argument("baseline_revision")
    parser.add_argument("change_set_digest")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
