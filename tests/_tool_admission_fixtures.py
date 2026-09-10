"""Explicit executable availability for tests with simulated tool backends.

These fixtures do not qualify a live runner. They preserve production admission
and supply only the fixed dependencies of the test's simulated backend.
"""

from types import SimpleNamespace

import pytest

from cayu import (
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecutionProfileBehaviorIdentity,
    LocalRunner,
)
from cayu.runners import local
from tests.core.test_environment_allocation_recovery import _FakeRemoteFactory


class SimulatedToolFactory(_FakeRemoteFactory):
    """Allocation fixture with an explicit runner for simulated tool dispatch."""

    def execution_admission_candidate(self, request):
        return LocalRunner(".").execution_admission_candidate_for(request.execution_requirements)

    def _result(self, request, resource, **kwargs):
        result = super()._result(request, resource, **kwargs)
        result.environment.runner = LocalRunner(".")
        return result


def simulated_tool_app(**kwargs):
    app = CayuApp(**kwargs)
    app.register_environment(
        Environment(
            EnvironmentSpec(
                name="tool-fixture",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="simulated-tool-environment",
                    behavior_version="1",
                    implementation_version="1",
                ),
            ),
            runner=LocalRunner("."),
        ),
        default=True,
    )
    return app


@pytest.fixture
def simulated_tool_executables(monkeypatch, request):
    executables = frozenset(getattr(request, "param", ("/usr/local/bin/python",)))
    original = local.shutil.which

    def which(command, **kwargs):
        return command if command in executables else original(command, **kwargs)

    monkeypatch.setattr(local, "shutil", SimpleNamespace(which=which))
