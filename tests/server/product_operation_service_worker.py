"""One maintained product service process sharing a PostgreSQL database.

Used by the two-process settlement test: the parent reserves product work,
starts two of these processes, and releases them together so both deliver
every operation concurrently.
"""

from __future__ import annotations

import asyncio
import json
import sys

from fastapi import HTTPException, Request

from cayu import (
    AgentSpec,
    CayuApp,
    ModelStreamEvent,
    PostgresProductOperationStore,
    PostgresSessionStore,
    PostgresTaskStore,
    ScriptedModelProvider,
)
from cayu.server import (
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    BasicAuth,
    ProductOperation,
    ProductPrincipal,
    ServiceMode,
    create_agent_service,
)

RESULT = "settled answer"


async def _deny(_request: Request) -> ProductPrincipal:
    raise HTTPException(status_code=401, detail="Authentication required.")


async def main() -> None:
    material = json.loads(sys.stdin.readline())
    dsn = material["dsn"]
    provider_calls = 0

    def respond(_request):
        nonlocal provider_calls
        provider_calls += 1
        return [
            ModelStreamEvent.text_delta(RESULT),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]

    session_store = PostgresSessionStore(dsn)
    task_store = PostgresTaskStore(dsn)
    product_store = PostgresProductOperationStore(dsn)
    try:
        app = CayuApp(session_store=session_store, task_store=task_store, enable_logging=False)
        app.register_provider(ScriptedModelProvider(response_factory=respond), default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        service = create_agent_service(
            app,
            agent_name="assistant",
            mode=ServiceMode.PRODUCTION,
            product_access=AuthenticatedProductAccess(dependency=_deny),
            operator_access=AuthenticatedAccess(
                dependency=BasicAuth(username="operator", password="operator-secret")
            ),
            product_store=product_store,
        )
        await product_store.ensure_schema()
        print(json.dumps({"ready": True}), flush=True)
        await asyncio.to_thread(sys.stdin.readline)
        outcomes = await asyncio.gather(
            *(service.execute_work(work_id) for work_id in material["work_ids"]),
            return_exceptions=True,
        )
        print(
            json.dumps(
                {
                    "provider_calls": provider_calls,
                    "outcomes": {
                        work_id: (
                            outcome.status
                            if isinstance(outcome, ProductOperation)
                            else repr(outcome)
                        )
                        for work_id, outcome in zip(material["work_ids"], outcomes, strict=True)
                    },
                }
            ),
            flush=True,
        )
    finally:
        await product_store.close()
        await task_store.close()
        await session_store.close()


if __name__ == "__main__":
    asyncio.run(main())
