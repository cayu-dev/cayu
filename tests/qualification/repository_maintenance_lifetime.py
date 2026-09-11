"""Observe process-owned tasks without releasing their resources on waiter loss."""

import asyncio


def raise_lifetime_failures(primary, cleanup):
    if isinstance(primary, asyncio.CancelledError):
        signal, secondary = primary, cleanup
        primary_cancelled = True
    elif isinstance(cleanup, asyncio.CancelledError):
        signal, secondary = cleanup, primary
        primary_cancelled = False
    else:
        raise BaseExceptionGroup("Maintenance operation and cleanup failed.", [primary, cleanup])
    previous = BaseException.__dict__["__cause__"].__get__(signal, BaseException)
    cause = secondary
    if previous is not None and previous is not secondary and previous is not signal:
        ordered = [previous, secondary] if primary_cancelled else [secondary, previous]
        cause = BaseExceptionGroup("Maintenance lifetime failure evidence.", ordered)
    raise signal from cause


async def wait_owned_task(task, *, on_cancel=None):
    cancellation = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            if on_cancel is not None:
                on_cancel()
            if cancellation is None:
                cancellation = exc
    try:
        result = task.result()
    except BaseException as failure:
        if cancellation is not None:
            raise_lifetime_failures(cancellation, failure)
        raise
    if cancellation is not None:
        raise cancellation
    return result


async def close_deployment(deployment):
    async def settled():
        while not await deployment.aclose(timeout_s=1):
            await asyncio.sleep(1)

    await wait_owned_task(asyncio.create_task(settled()))
