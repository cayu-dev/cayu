"""Move files explicitly between durable artifacts and a mutable workspace.

Run:
    PYTHONPATH=src python examples/artifact_workspace_bridge.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from cayu import (
    ArtifactScope,
    LocalArtifactStore,
    LocalWorkspace,
    copy_artifact_to_workspace,
    copy_workspace_file_to_artifact,
)


async def main() -> None:
    root = Path(__file__).resolve().parents[1] / ".examples-workspaces" / "artifact-workspace"
    (root / "workspace").mkdir(parents=True, exist_ok=True)
    workspace = LocalWorkspace(root / "workspace", workspace_id="invoice-workspace")
    artifact_store = LocalArtifactStore(root / "artifacts", store_id="invoice-artifacts")

    original = await artifact_store.put_bytes(
        b"Invoice #42\nTotal: 120.00 USD\n",
        filename="invoice-42.txt",
        content_type="text/plain",
        scope=ArtifactScope.SESSION,
        session_id="demo_artifact_workspace",
        agent_name="invoice-agent",
        environment_name="local-dev",
        metadata={"source": "upload"},
    )
    print("original_artifact", original.id)

    materialized = await copy_artifact_to_workspace(
        artifact_store,
        workspace,
        original.id,
        "inputs/invoice-42.txt",
    )
    print("workspace_input", materialized.workspace_path, materialized.bytes_written)

    input_file = await workspace.read_bytes("inputs/invoice-42.txt")
    summary = b"Extracted invoice fields:\n" + input_file.content
    await workspace.write_bytes("results/invoice-42-summary.txt", summary)

    output = await copy_workspace_file_to_artifact(
        workspace,
        artifact_store,
        "results/invoice-42-summary.txt",
        scope=ArtifactScope.SESSION,
        session_id="demo_artifact_workspace",
        agent_name="invoice-agent",
        environment_name="local-dev",
        metadata={"source_artifact_id": original.id, "kind": "analysis_output"},
    )
    print("output_artifact", output.artifact.id)
    print("workspace_result", output.workspace_path, output.bytes_read)


if __name__ == "__main__":
    asyncio.run(main())
