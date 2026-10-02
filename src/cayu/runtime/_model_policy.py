"""Opt-in multi-agent model-policy adoption; no financial admission machinery."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cayu._task_wait import capture_awaitable_outcome
from cayu.runtime._policy_freshness import boot_identity, require_fresh
from cayu.runtime._policy_installation import PolicyInstallationOwner
from cayu.runtime._policy_storage import ModelPolicyStore
from cayu.runtime._policy_wire import canonical, decode, identifier, require
from cayu.sessions.base import ModelTarget


class PolicyChannel(ABC):
    """Explicit authenticated management authority, independent of inference keys."""

    @property
    @abstractmethod
    def scope(self) -> dict[str, str]: ...

    @property
    @abstractmethod
    def incarnation(self) -> tuple[str, int]: ...

    @abstractmethod
    async def read_snapshot(self) -> bytes: ...

    @abstractmethod
    async def report(self, report: bytes) -> bytes: ...

    async def aclose(self) -> None:
        """Close channel-owned transport resources, if any."""
        return None


@dataclass(frozen=True)
class PolicySelection:
    target: ModelTarget
    evidence: bytes


class ModelPolicyController:
    """One explicit Cloud-to-local-agent mapping; store/channel ownership is external."""

    def __init__(
        self,
        *,
        store: ModelPolicyStore,
        agent_name: str,
        provider_name: str,
        scope: dict[str, str],
        incarnation: tuple[str, int],
        channel: PolicyChannel | None = None,
    ):
        identifier(agent_name)
        identifier(provider_name)
        self.agent_name, self.provider_name = agent_name, provider_name
        self._scope = decode(canonical(scope))
        self._incarnation = incarnation
        self._channel = channel
        self._owner = PolicyInstallationOwner(
            store,
            binding={"scope": scope, "agent_name": agent_name, "provider_name": provider_name},
            incarnation=incarnation,
        )
        self._lock = asyncio.Lock()
        self._boot = boot_identity()
        self._catalog: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None
        self._preflight: Callable[..., None] | None = None
        self.status = "stopped"
        self._require_channel(optional=True)

    def _require_channel(self, *, optional=False):
        channel = self._channel
        if optional and channel is None:
            return None
        require(isinstance(channel, PolicyChannel))
        assert channel is not None
        require(channel.scope == self._scope and channel.incarnation == self._incarnation)
        return channel

    async def start(self, provider) -> None:
        self._catalog = getattr(provider, "get_models", None)
        require(callable(self._catalog) or self._channel is None)
        self._preflight = provider.preflight_model_target
        await self._owner.start()
        self.status = "ready"

    def selection(self) -> PolicySelection | None:
        report = self._owner.current()
        if report is None:
            return None
        value = decode(report)
        model = value["installed_model"]
        if model is None:
            return None
        return PolicySelection(
            ModelTarget(provider_name=self.provider_name, model=model),
            canonical(
                {
                    "scope": self._scope,
                    "incarnation_id": value["incarnation_id"],
                    "incarnation_epoch": value["incarnation_epoch"],
                    "installation_id": value["installation_id"],
                    "installation_seq": value["installation_seq"],
                    "action": value["action"],
                    "snapshot": value["snapshot"],
                    "model": model,
                    "provider_name": self.provider_name,
                }
            ),
        )

    def _report(self, *, action: str, model: str | None, snapshot=None, reason=None) -> bytes:
        seq = self._owner.state()["sequence"] + 1
        base: dict[str, Any] = {
            "schema_version": 1,
            "scope": self._scope,
            "incarnation_id": self._incarnation[0],
            "incarnation_epoch": self._incarnation[1],
            "operation_id": f"policy-{seq}",
            "target": "application_default",
        }
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        ref = (
            None
            if snapshot is None
            else {
                "snapshot_id": snapshot["snapshot_id"],
                "effective_revision": snapshot["effective"]["effective_revision"],
                "config_sha256": snapshot["config_sha256"],
            }
        )
        if reason is not None:
            base.update(
                kind="adoption_refusal",
                decision_seq=seq,
                snapshot=ref,
                reason=reason,
                observed_at=timestamp,
            )
        else:
            base.update(
                kind="installation_report",
                installation_seq=seq,
                installation_id=f"install-{seq}",
                action=action,
                installed_model=model,
                snapshot=ref,
                locally_committed_at=timestamp,
            )
        return canonical(base)

    async def poll_once(self) -> None:
        async with self._lock:
            await self._owner.reconcile()
            channel = self._require_channel(optional=True)
            if channel is None:
                return
            # Drain before accepting more work, so unavailable reporting cannot
            # fill the bounded journal with identical observations on every poll.
            await self._acknowledge(channel)
            if self._owner.state()["adoption_enabled"]:
                started = time.monotonic_ns()
                async with asyncio.timeout(10):
                    wire = await channel.read_snapshot()
                self._require_channel()
                await self._owner.observe(
                    wire, started_ns=started, received_ns=time.monotonic_ns(), boot=self._boot
                )
                await self._adopt_observed()
            await self._acknowledge(channel)
            self.status = "ready"

    async def adopt_cached(self) -> None:
        """Re-adopt an authenticated same-boot snapshot within its original deadline."""
        async with self._lock:
            await self._adopt_observed()

    async def _adopt_observed(self) -> None:
        state = self._owner.state()
        require(state["adoption_enabled"])
        observation = state["observation"]
        require_fresh(observation, now_ns=time.monotonic_ns(), boot=self._boot)
        snapshot = observation["snapshot"]
        effective = snapshot["effective"]
        # An unchanged effective configuration needs no duplicate installation.
        current = state["current"]
        if (
            current is not None
            and current["action"] == "install_default"
            and current["snapshot"]["config_sha256"] == snapshot["config_sha256"]
        ):
            return
        reason = {"absent": "default_absent", "ineligible": "default_ineligible"}.get(
            effective["default_state"]
        )
        model = effective["default_model"]
        if reason is None:
            require(self._catalog is not None)
            assert self._catalog is not None
            async with asyncio.timeout(10):
                models = await self._catalog()
            require(type(models) is list and len(models) <= 4096)
            require(all(type(item) is dict and type(item.get("id")) is str for item in models))
            if model not in {item["id"] for item in models}:
                reason = "model_unknown"
            else:
                try:
                    assert self._preflight is not None
                    self._preflight(model=model)
                except Exception:
                    reason = "model_unsupported"
        require_fresh(observation, now_ns=time.monotonic_ns(), boot=self._boot)
        report = self._report(
            action="install_default", model=model, snapshot=snapshot, reason=reason
        )
        await self._owner.install(report, deadline_ns=observation["deadline_ns"])

    async def _acknowledge(self, channel) -> None:
        for report in self._owner.pending():
            self._require_channel()
            async with asyncio.timeout(10):
                receipt = await channel.report(report)
            self._require_channel()
            await self._owner.acknowledge(receipt, expected_report=report)

    async def override_default(self, *, model: str) -> None:
        identifier(model)
        async with self._lock:
            assert self._preflight is not None
            self._preflight(model=model)
            await self._owner.install(self._report(action="override_default", model=model))

    async def withdraw(self) -> None:
        async with self._lock:
            current = self._owner.current()
            model = None if current is None else decode(current)["installed_model"]
            await self._owner.install(self._report(action="withdraw", model=model))

    async def resume_adoption(self) -> None:
        async with self._lock:
            await self._owner.resume_adoption()

    async def close(self) -> None:
        await self._owner.close()
        self.status = "stopped"


class ModelPolicy:
    """Owned background workers for explicit, independent local-agent bindings."""

    def __init__(
        self,
        controllers: Iterable[ModelPolicyController],
        *,
        poll_interval: float = 20,
        close_channels: bool = False,
    ):
        require(type(close_channels) is bool)
        self._close_channels = close_channels
        require(type(poll_interval) in (int, float) and 0 < poll_interval <= 30)
        self.controllers = tuple(controllers)
        require(all(type(item) is ModelPolicyController for item in self.controllers))
        require(len({item.agent_name for item in self.controllers}) == len(self.controllers))
        require(len({canonical(item._scope) for item in self.controllers}) == len(self.controllers))
        self.poll_interval = poll_interval
        self._tasks: list[asyncio.Task] = []
        self._started: list[ModelPolicyController] = []
        self._closing: asyncio.Task | None = None

    def selection(self, agent_name: str) -> PolicySelection | None:
        for controller in self.controllers:
            if controller.agent_name == agent_name:
                return controller.selection()
        return None

    async def start(self, resolve_provider: Callable[[str], Any]) -> None:
        require(not self._started and not self._tasks)
        try:
            for controller in self.controllers:
                # Retain cleanup ownership even if start commits then loses ACK.
                self._started.append(controller)
                await controller.start(resolve_provider(controller.provider_name))
                self._tasks.append(asyncio.create_task(self._renew(controller)))
            async with asyncio.TaskGroup() as group:
                for controller in self.controllers:
                    group.create_task(self._poll_once(controller))
            for controller in self.controllers:
                self._tasks.append(asyncio.create_task(self._poll(controller)))
        except BaseException as primary:
            try:
                await self.close()
            except BaseException as cleanup:
                if isinstance(primary, asyncio.CancelledError):
                    evidence = [cleanup]
                    if primary.__cause__ is not None:
                        evidence.insert(0, primary.__cause__)
                    raise primary from BaseExceptionGroup(
                        "Model policy startup cleanup failed.", evidence
                    )
                if isinstance(cleanup, asyncio.CancelledError):
                    evidence = [primary]
                    if cleanup.__cause__ is not None:
                        evidence.append(cleanup.__cause__)
                    raise cleanup from BaseExceptionGroup(
                        "Model policy startup cleanup failed.", evidence
                    )
                raise BaseExceptionGroup(
                    "Model policy startup and cleanup failed.", [primary, cleanup]
                ) from None
            raise

    async def _poll(self, controller):
        while True:
            await asyncio.sleep(self.poll_interval)
            await self._poll_once(controller)

    async def _poll_once(self, controller):
        try:
            await controller.poll_once()
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            controller.status = "unavailable"
        except Exception:
            controller.status = "unavailable"

    async def _renew(self, controller):
        while True:
            await asyncio.sleep(10)
            try:
                await controller._owner.renew()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                controller.status = "ownership_unavailable"
            except Exception:
                controller.status = "ownership_unavailable"

    async def close(self) -> None:
        if self._closing is None or self._closing.done():
            # Capture owned cleanup failures before they reach shield's task
            # machinery. Python 3.14 reports a failing shielded future to the
            # loop after caller cancellation, even when we subsequently await
            # that same task and retain its failure as cancellation evidence.
            self._closing = asyncio.create_task(capture_awaitable_outcome(self._close))
        cancellation = None
        while not self._closing.done():
            try:
                await asyncio.shield(self._closing)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except BaseException:
                break
        failure = self._closing.result().error
        if cancellation is not None:
            raise cancellation from failure
        if failure is not None:
            raise failure

    async def _close(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()

        async def completion(task):
            # gather replaces a cancelled task's exception with a fresh signal,
            # losing any owned-operation failure attached during quiescence.
            # Observe the original outcome before gathering; shutdown's expected
            # cancellation is not itself a failure.
            try:
                await task
            except asyncio.CancelledError as cancellation:
                return cancellation.__cause__
            except BaseException as failure:
                return failure
            return None

        results = await asyncio.gather(*(completion(task) for task in tasks))
        failures = [result for result in results if result is not None]
        for controller in reversed(self._started):
            try:
                await controller.close()
                if self._close_channels and controller._channel is not None:
                    await controller._channel.aclose()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise BaseExceptionGroup("Model policy shutdown failed.", failures)
        self._started.clear()
