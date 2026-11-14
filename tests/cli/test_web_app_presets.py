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


@pytest.mark.parametrize("option", ["with_capabilities", "without_capabilities"])
def test_the_b2b_preset_keeps_its_own_web_app(option: str) -> None:
    with pytest.raises(ScaffoldPlanError) as excinfo:
        normalize_application_plan(name="x", agent_name="x", preset="b2b", **{option: ("web-app",)})
    assert excinfo.value.code == "capability_required"
