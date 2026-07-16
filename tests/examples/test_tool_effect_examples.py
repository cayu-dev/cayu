from examples.dashboard_pending_actions import EchoTool as DashboardEchoTool
from examples.dashboard_pending_actions import FailingHealthCheckTool
from examples.echo_tool_runtime import EchoTool as RuntimeEchoTool

from cayu import ToolEffect


def test_pure_example_tools_declare_none() -> None:
    assert RuntimeEchoTool.spec.effect is ToolEffect.NONE
    assert DashboardEchoTool.spec.effect is ToolEffect.NONE
    assert FailingHealthCheckTool.spec.effect is ToolEffect.NONE
