"""Which presets get the `web-app` capability, and which refuse it."""

from __future__ import annotations

import pytest

from cayu.cli.scaffold_plan import ScaffoldPlanError, normalize_application_plan


@pytest.mark.parametrize("preset", ["agent", "coding", "long-running"])
def test_an_operator_preset_has_the_web_app_unless_left_out(preset: str) -> None:
    assert (
        "web-app"
        in normalize_application_plan(name="x", agent_name="x", preset=preset).capabilities
    )
    left_out = normalize_application_plan(
        name="x", agent_name="x", preset=preset, without_capabilities=("web-app",)
    )
    assert "web-app" not in left_out.capabilities


def test_the_service_preset_has_no_web_app_and_says_why() -> None:
    plan = normalize_application_plan(name="x", agent_name="x", preset="service")
    assert "web-app" not in plan.capabilities

    with pytest.raises(ScaffoldPlanError) as excinfo:
        normalize_application_plan(
            name="x", agent_name="x", preset="service", with_capabilities=("web-app",)
        )
    assert excinfo.value.code == "unsupported_capability"
    assert "bearer token" in str(excinfo.value)
    assert "agent, coding or long-running" in str(excinfo.value)


def test_the_b2b_preset_is_the_web_app_with_organizations() -> None:
    plan = normalize_application_plan(name="x", agent_name="x", preset="b2b")
    assert {"web-app", "organizations", "app-database"} <= set(plan.capabilities)
    # Asking for what it has already changes nothing.
    assert (
        normalize_application_plan(
            name="x", agent_name="x", preset="b2b", with_capabilities=("web-app",)
        )
        == plan
    )

    with pytest.raises(ScaffoldPlanError) as excinfo:
        normalize_application_plan(
            name="x", agent_name="x", preset="b2b", without_capabilities=("web-app",)
        )
    assert excinfo.value.code == "capability_dependency_conflict"
    assert "members sign in to the web app" in str(excinfo.value)
