"""One protected server surface over the application's browser control owner."""

from fastapi import APIRouter

from cayu.runtime._browser_control_input_tickets import BrowserInputTickets
from cayu.runtime._browser_control_runtime import BrowserControlRuntime
from cayu.runtime._browser_control_view_tickets import BrowserViewTickets
from cayu.server._browser_control_config import BrowserControlServerConfig
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server._browser_guest_routes import create_browser_guest_router
from cayu.server._browser_input_routes import create_browser_input_router
from cayu.server._browser_viewer_routes import create_browser_viewer_router
from cayu.server.auth import AuthDependency


class BrowserControlServer:
    def __init__(
        self,
        *,
        runtime: BrowserControlRuntime,
        config: BrowserControlServerConfig,
        auth: AuthDependency,
    ) -> None:
        # Explicit reconstruction also checks post-construction mutation without
        # serializing secret material into a configuration dump.
        config = BrowserControlServerConfig(
            operator_origin=config.operator_origin, signing_key=config.signing_key
        )
        self.runtime = runtime
        self.views = BrowserViewTickets(runtime.coordinator)
        self.inputs = BrowserInputTickets(runtime.coordinator)
        self.router = APIRouter()
        self.router.include_router(
            create_browser_control_router(
                coordinator=runtime.coordinator,
                auth=auth,
                sessions=BrowserOperatorSessionTokens(config.signing_key.get_secret_value()),
                allowed_origin=config.operator_origin,
                view_tickets=self.views,
                input_tickets=self.inputs,
                service=runtime.service,
            )
        )
        self.router.include_router(
            create_browser_guest_router(
                coordinator=runtime.coordinator,
                service=runtime.service,
            )
        )
        self.router.include_router(
            create_browser_viewer_router(
                coordinator=runtime.coordinator,
                service=runtime.service,
                tickets=self.views,
                allowed_origin=config.operator_origin,
            )
        )
        self.router.include_router(
            create_browser_input_router(
                coordinator=runtime.coordinator,
                service=runtime.service,
                tickets=self.inputs,
                allowed_origin=config.operator_origin,
            )
        )

    async def drain(self) -> bool:
        self.views.close()
        self.inputs.close()
        if not await self.runtime.service.drain():
            # Guest cleanup may still need to publish its exact fence. Do not
            # seal that publisher while the native/channel owner remains live.
            return False
        return await self.runtime.coordinator.drain()
