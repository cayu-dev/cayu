"""Open a network-disabled runner for :func:`~cayu.runners.isolated.run_isolated`.

:func:`open_isolated_runner` picks where an isolated command runs, by the same
rule :func:`~cayu.environments.coding_backends.select_coding_backend` uses:
Cayu Cloud's sandbox broker when the ``CAYU_CLOUD_*`` sandbox variables are
set, Docker otherwise, or the backend ``CAYU_ISOLATED_BACKEND`` (or
``required``) names. It yields a runner with no network and always releases it.

The parts stand alone. :class:`IsolatedBackendRegistry` maps backend names to
openers; ``docker`` and ``cayu-cloud`` are built in and another provider
registers its own opener under its own name. An opener is any callable that
takes an :class:`IsolatedRunnerContext` and returns an async context manager
yielding a :class:`~cayu.runners.base.Runner`.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cayu.environments.coding_backends import (
    CLOUD_CODING_BACKEND,
    DOCKER_CODING_BACKEND,
    CodingBackendAvailability,
    _select_backend_name,
    cloud_sandbox_unavailable_reason,
)
from cayu.runners.base import Runner

if TYPE_CHECKING:
    from cayu.sandbox_broker.client import SandboxBrokerClient

#: Names the one isolated backend a deployment requires.
ISOLATED_BACKEND_ENV = "CAYU_ISOLATED_BACKEND"
_KEY_LABEL_CHARS = 32


@dataclass(frozen=True)
class IsolatedRunnerContext:
    """What an opener gets: a run key, the process environment and its own options."""

    key: str
    environ: Mapping[str, str]
    options: object | None = None


IsolatedRunnerOpener = Callable[[IsolatedRunnerContext], AbstractAsyncContextManager[Runner]]


@dataclass(frozen=True)
class DockerIsolatedOptions:
    """Options for the ``docker`` backend. The image needs ``python3``."""

    image: str = "python:3.13-slim"


@dataclass(frozen=True)
class CloudIsolatedOptions:
    """Options for the ``cayu-cloud`` backend.

    ``broker`` is a client to use instead of one built from the environment;
    the caller then owns and closes it.
    """

    image: str = "coding-python"
    lease_seconds: int = 300
    #: How often the lease is renewed while the runner is open; a third of the lease by default.
    renew_every_s: float | None = None
    ready_timeout_s: float = 330.0
    poll_interval_s: float = 1.0
    labels: Mapping[str, str] = field(default_factory=dict)
    broker: SandboxBrokerClient | None = None


@dataclass(frozen=True)
class _Registration:
    opener: IsolatedRunnerOpener
    available: CodingBackendAvailability | None


class IsolatedBackendRegistry:
    """Named isolated-runner openers. Each registry is independent; none is global."""

    def __init__(self) -> None:
        self._registrations: dict[str, _Registration] = {}

    def register(
        self,
        name: str,
        opener: IsolatedRunnerOpener,
        *,
        available: CodingBackendAvailability | None = None,
        replace: bool = False,
    ) -> None:
        """Add a backend. ``replace=True`` swaps one already registered, for example in tests."""

        if type(name) is not str or not name or name != name.strip().lower():
            raise ValueError("An isolated backend name must be a lowercase identifier.")
        if not callable(opener):
            raise TypeError("opener must be callable.")
        if available is not None and not callable(available):
            raise TypeError("available must be callable or None.")
        if name in self._registrations and not replace:
            raise ValueError(f"An isolated backend named {name!r} is already registered.")
        self._registrations[name] = _Registration(opener, available)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def selected_name(
        self, environ: Mapping[str, str] | None = None, *, required: str | None = None
    ) -> str:
        """Return the backend this environment selects, or raise why it can't select one."""

        return _select_backend_name(
            {name: r.available for name, r in self._registrations.items()},
            environ,
            required=required,
            env_variable=ISOLATED_BACKEND_ENV,
            kind="isolated",
        )

    def opener(self, name: str) -> IsolatedRunnerOpener:
        registration = self._registrations.get(name)
        if registration is None:
            raise ValueError(f"No isolated backend is registered as {name!r}.")
        return registration.opener


def default_isolated_backends() -> IsolatedBackendRegistry:
    """Return a new registry with the built-in ``docker`` and ``cayu-cloud`` backends."""

    registry = IsolatedBackendRegistry()
    registry.register(DOCKER_CODING_BACKEND, open_docker_isolated_runner)
    registry.register(
        CLOUD_CODING_BACKEND,
        open_cloud_isolated_runner,
        available=cloud_sandbox_unavailable_reason,
    )
    return registry


@asynccontextmanager
async def open_isolated_runner(
    key: str | None = None,
    *,
    options: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
    required: str | None = None,
    registry: IsolatedBackendRegistry | None = None,
) -> AsyncIterator[Runner]:
    """Open the selected backend's network-disabled runner and release it on exit.

    ``key`` (for example :attr:`~cayu.runners.isolated.IsolatedRun.key` from
    :func:`~cayu.runners.isolated.isolated_run_key`) names the runner and seeds
    the sandbox's idempotency key; each call still gets a fresh runner.
    ``options`` are keyed by backend name.
    """

    selected = default_isolated_backends() if registry is None else registry
    if not isinstance(selected, IsolatedBackendRegistry):
        raise TypeError("registry must be an IsolatedBackendRegistry.")
    env = os.environ if environ is None else environ
    name = selected.selected_name(env, required=required)
    run_key = secrets.token_hex(16) if key is None else key
    if type(run_key) is not str or not run_key.isalnum() or not run_key.isascii():
        raise ValueError("key must be ASCII letters and digits, such as a run key digest.")
    context = IsolatedRunnerContext(
        key=run_key, environ=env, options=None if options is None else options.get(name)
    )
    async with selected.opener(name)(context) as runner:
        if not isinstance(runner, Runner):
            raise TypeError(f"The {name!r} isolated backend yielded no Runner.")
        yield runner


@asynccontextmanager
async def open_docker_isolated_runner(context: IsolatedRunnerContext) -> AsyncIterator[Runner]:
    """A Docker container with no network and Cayu's workload restrictions, removed on exit."""

    from cayu.runners.docker import DockerRunner
    from cayu.runners.docker_workload import DockerWorkloadRestrictions

    options = context.options or DockerIsolatedOptions()
    if type(options) is not DockerIsolatedOptions:
        raise TypeError("docker options must be DockerIsolatedOptions.")
    runner = await DockerRunner.create(
        f"cayu-isolated-{context.key[:_KEY_LABEL_CHARS]}-{secrets.token_hex(4)}",
        image=options.image,
        network="none",
        replace=False,
        close_action="remove",
        workload_restrictions=DockerWorkloadRestrictions(),
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
    )
    try:
        yield runner
    finally:
        await runner.close()


@asynccontextmanager
async def open_cloud_isolated_runner(context: IsolatedRunnerContext) -> AsyncIterator[Runner]:
    """A brokered sandbox with ``egress_mode="none"``, destroyed on exit.

    Its lease is renewed while the block runs, so a long command keeps its
    sandbox. A process that stops without exiting leaves only the remaining
    lease, which Cayu Cloud ends when it runs out.
    """

    from cayu.sandbox_broker.client import SandboxBrokerClient
    from cayu.sandbox_broker.lease import SandboxLeaseKeeper
    from cayu.sandbox_broker.runner import BrokeredSandboxRunner

    options = context.options or CloudIsolatedOptions()
    if type(options) is not CloudIsolatedOptions:
        raise TypeError("cayu-cloud options must be CloudIsolatedOptions.")
    owned = options.broker is None
    broker = SandboxBrokerClient.from_env(context.environ) if owned else options.broker
    assert broker is not None
    try:
        created = await broker.create(
            idempotency_key=f"cayu-isolated-{context.key}-{secrets.token_hex(8)}",
            image=options.image,
            egress_mode="none",
            lease_seconds=options.lease_seconds,
            labels={"cayu-run-key": context.key[:_KEY_LABEL_CHARS], **options.labels},
        )
        runner: BrokeredSandboxRunner | None = None
        keeper: SandboxLeaseKeeper | None = None
        try:
            sandbox = await broker.wait_until_ready(
                created.id,
                timeout_s=options.ready_timeout_s,
                poll_interval_s=options.poll_interval_s,
            )
            if sandbox.egress.mode != "none":
                raise RuntimeError(
                    f"The broker gave sandbox {sandbox.id} {sandbox.egress.mode!r} egress, "
                    "not 'none'."
                )
            # Renew the lease for as long as the block runs; close() stops it.
            keeper = SandboxLeaseKeeper(
                broker,
                sandbox,
                lease_seconds=options.lease_seconds,
                renew_every_s=options.renew_every_s,
            ).start()
            runner = BrokeredSandboxRunner(broker, sandbox, lease_keeper=keeper)
            yield runner
        finally:
            if runner is not None:
                await runner.close()
            else:
                if keeper is not None:
                    await keeper.stop()
                await broker.destroy(created.id)
    finally:
        if owned:
            await broker.aclose()


__all__ = [
    "ISOLATED_BACKEND_ENV",
    "CloudIsolatedOptions",
    "DockerIsolatedOptions",
    "IsolatedBackendRegistry",
    "IsolatedRunnerContext",
    "IsolatedRunnerOpener",
    "default_isolated_backends",
    "open_cloud_isolated_runner",
    "open_docker_isolated_runner",
    "open_isolated_runner",
]
