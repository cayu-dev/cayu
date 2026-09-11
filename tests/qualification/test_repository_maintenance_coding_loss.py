"""Real process loss through the emitted app; local runner, not Docker qualification."""

import os
from pathlib import Path

import pytest

from cayu import DockerImageIdentity
from tests.qualification.repository_maintenance_case import SEED_FILES, materialize_seed_repository
from tests.qualification.repository_maintenance_toolchain import maintenance_toolchain
from tests.qualification.test_repository_maintenance_application import (
    project as project,
)
from tests.qualification.test_repository_maintenance_application import (
    scripted_journey_budget,
)
from tests.recovery.worker_harness import BackendConfig, RecoveryHarness


@pytest.mark.process
@pytest.mark.sigkill_recovery
@pytest.mark.skipif(os.name != "posix", reason="SIGKILL process groups require POSIX")
@pytest.mark.parametrize("backend", ["sqlite"])
def test_generated_coding_loss_keeps_identity_and_never_redispatches(
    project, tmp_path, monkeypatch, backend
):
    monkeypatch.setenv("CAYU_MODEL", "maintenance-fixture")
    source, target = tmp_path / "source", tmp_path / "target"
    materialize_seed_repository(source)
    target.mkdir()
    controls = tmp_path / "controls"
    controls.mkdir()
    database = project / ".cayu/runtime/cayu.db"
    profile = maintenance_toolchain(
        image_identity=DockerImageIdentity(reference="example.invalid/coding@sha256:" + "a" * 64),
        architecture="amd64",
        build_context_sha256="sha256:" + "b" * 64,
    )
    values = dict(
        project=str(project),
        source=str(source),
        target=str(target),
        toolchain_json=profile.model_dump_json(),
        budget_json=scripted_journey_budget().model_dump_json(),
        budget_path=str(tmp_path / "budget.sqlite"),
        reservation_path=str(tmp_path / "reservations.sqlite"),
    )
    with RecoveryHarness(
        controls,
        BackendConfig(kind=backend, session_path=str(database), task_path=str(database)),
        owns_backend=False,
    ) as harness:
        producer = harness.launch(
            scenario="maintenance_coding",
            action="start",
            session_id="allocated-by-intake",
            **values,
        )
        phase = producer.wait_for_phase("maintenance_model_dispatched")
        producer.sigkill()
        identity = phase["identity"]
        replacement = harness.launch(
            scenario="maintenance_coding",
            action="recover",
            session_id=identity["session_id"],
            task_id=identity["task_id"],
            identity=identity,
            **values,
        )
        result = replacement.wait_success()
        assert result["identity"] == identity
        assert result["task_status"] == "cancelled" and result["provider_calls"] == 0
        assert result["state"] in {"cancelled", "reconstruction_required"}
        assert producer.process.poll() is not None and replacement.process.poll() == 0
    assert database.is_file()
    for relative, expected in SEED_FILES.items():
        assert (source / relative).read_text() == expected
    assert target.is_dir()  # Explicitly retained local fixture, not an orphan Docker guest.
    assert any(path.is_file() for path in Path(project / ".cayu/runtime/artifacts").rglob("*"))
