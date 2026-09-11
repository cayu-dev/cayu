"""Use existing process-loss tooling, retaining caller-owned application stores."""

import importlib
import shutil
from pathlib import Path

from tests.recovery.worker_harness import BackendConfig, RecoveryHarness


def exercise_approval_restart(application, reservations, identity, *, root, remote):
    """Called only after coding has settled, before a preparation worker starts."""
    controls = root / "approval-restart-controls"
    controls.mkdir()
    backend = BackendConfig(
        kind="sqlite",
        session_path=str(application.app.session_store.path),
        task_path=str(application.app.task_store.path),
    )
    module_path = importlib.import_module("app").__file__
    assert module_path is not None
    values = {
        "project": str(Path(module_path).parent),
        "source": str(application.project_root),
        "reservation_path": str(reservations.path),
        "public_id": identity.public_id,
        "tenant": identity.intent.tenant,
        "identity": identity.model_dump(mode="json"),
        "toolchain_json": application.toolchain_profile.model_dump_json(),
        "budget_json": application.app.budget_policy.model_dump_json(),
        "budget_path": str(root / "journey-budget.sqlite"),
        "git_host": {
            "broker_root": str(root / "broker"),
            "git_executable": shutil.which("git") or "/usr/bin/git",
            "remote_url": str(remote),
        },
    }
    with RecoveryHarness(controls, backend, owns_backend=False) as harness:
        producer = harness.launch(
            scenario="maintenance_approval",
            action="start",
            session_id=identity.session_id,
            task_id=identity.git_preparation_task_id,
            **values,
        )
        phase = producer.wait_for_phase("maintenance_approval_durable")
        producer.sigkill()
        replacement = harness.launch(
            scenario="maintenance_approval",
            action="recover",
            session_id=identity.session_id,
            task_id=identity.git_delivery_task_id,
            expected_review=phase["review"],
            expected_preparation=phase["preparation"],
            **values,
        )
        result = replacement.wait_success()
        assert result["review"] == phase["review"]
        assert result["approved_task"]["id"] == identity.git_delivery_task_id
        assert result["provider_calls"] == 0
        assert producer.process.poll() is not None and replacement.process.poll() == 0
    assert backend.session_path is not None and backend.task_path is not None
    assert Path(backend.session_path).is_file() and Path(backend.task_path).is_file()
    return result
