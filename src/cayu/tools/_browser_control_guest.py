"""Private native-input fence shipped with the browser worker (stdlib only).

Only the authenticated control-channel owner calls these methods. There is no
model JSON operation that binds, acquires, renews or hands back this authority.
The daemon owns quiescence and calls grant/finish_handback only under its lock.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any


class GuestControlFailure(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Browser control authority is unavailable or stale.")


class GuestCaptureDenied(RuntimeError):
    """Known pre-capture refusal, not loss of the shared control channel."""


class GuestControlAllocationLost(GuestControlFailure):
    """A command was refused before dispatch after observed browser disconnection.

    This remains a channel failure, but is not itself a failed cleanup operation.
    Only successful native shutdown can retire it as a cleanup blocker.
    """


class GuestFrameOwner:
    """One transient capture/send, with positive settling before sensitive input.

    There is no unbounded frame queue and no frame persistence. Pausing prevents
    dispatch synchronously; a frame captured before pause is discarded unless its
    send already began, in which case pause waits for that send to settle.
    """

    def __init__(self) -> None:
        self.paused = True
        self.generation = 1
        self.task: asyncio.Task[tuple[BaseException, ...]] | None = None

    def resume(self) -> None:
        if self.task is not None:
            if not self.task.done() or self.task.result():
                raise GuestControlFailure()
            self.task = None
        self.generation += 1
        self.paused = False

    def suspend(self) -> None:
        self.paused = True
        self.generation += 1

    async def pause(self, *, timeout_s: float = 5.0) -> None:
        if type(timeout_s) not in {float, int} or not 0 < timeout_s <= 5:
            raise GuestControlFailure()
        self.suspend()
        generation = self.generation
        task = self.task
        if task is None:
            return
        _, pending = await asyncio.wait({task}, timeout=timeout_s)
        if pending:
            raise GuestControlFailure()
        if not self.paused or self.generation != generation:
            raise GuestControlFailure()
        failures = task.result()
        if failures:
            raise BaseExceptionGroup(
                "Browser frame capture or delivery did not settle.", list(failures)
            )
        if self.task is task:
            self.task = None

    async def capture_one(
        self,
        *,
        capture: Callable[[], Awaitable[bytes]],
        send: Callable[[int, bytes], Awaitable[None]],
        guard: Callable[[], None],
    ) -> None:
        if self.paused:
            raise GuestControlFailure()
        if self.task is not None and (not self.task.done() or self.task.result()):
            raise GuestControlFailure()
        generation = self.generation

        async def capture_and_send() -> tuple[BaseException, ...]:
            frame = None
            try:
                try:
                    guard()
                except GuestControlFailure:
                    self.suspend()
                    return ()
                frame = await capture()
                if type(frame) is not bytes or not 0 < len(frame) <= 2 * 1024 * 1024:
                    raise GuestControlFailure()
                if self.paused or self.generation != generation:
                    return ()
                try:
                    guard()
                except GuestControlFailure:
                    self.suspend()
                    return ()
                await send(generation, frame)
                return ()
            except BaseException as failure:
                self.suspend()
                return (failure,)
            finally:
                frame = None

        task = asyncio.create_task(capture_and_send(), name="cayu-private-browser-frame")
        self.task = task
        try:
            failures = await asyncio.shield(task)
        except BaseException:
            self.suspend()
            raise
        if failures:
            raise BaseExceptionGroup("Browser frame capture or delivery failed.", list(failures))
        if self.task is task:
            self.task = None


class GuestControlChannel:
    """One authenticated runtime-to-guest control connection, not a model API.

    A verified private bootstrap opens the connection. Scope and worker identity
    bind its handshake; a fresh nonce excludes commands from an older connection.
    Model arguments can neither construct nor attach this channel.
    """

    def __init__(self, daemon: Any, *, scope_sha256: str) -> None:
        GuestControlFence._digest(scope_sha256)
        self._daemon = daemon
        self._scope = scope_sha256
        self._nonce = "bc_" + secrets.token_hex(16)
        self._sequence = 0
        self._binding: str | None = None

    @staticmethod
    def _decode(value: Any) -> dict[str, Any]:
        if type(value) is not str or len(value) > 65536 or len(value.encode("utf-8")) > 65536:
            raise GuestControlFailure()

        def unique(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise GuestControlFailure()
                result[key] = item
            return result

        try:
            result = json.loads(value, object_pairs_hook=unique)
        except (ValueError, RecursionError):
            raise GuestControlFailure() from None
        if type(result) is not dict:
            raise GuestControlFailure()
        return result

    async def run(self, connection: Any) -> None:
        daemon = self._daemon
        task = asyncio.current_task()
        cancellation_baseline = 0 if task is None else task.cancelling()
        failure: BaseException | None = None
        claimed = False
        try:
            daemon.claim_operator_channel(self._nonce)
            claimed = True
            await connection.send(
                json.dumps(
                    {
                        "kind": "hello",
                        "schema_version": 1,
                        "scope_sha256": self._scope,
                        "channel_id": self._nonce,
                        "browser_session_id": daemon.session_id,
                        "worker_instance": daemon.visual_worker_instance,
                    }
                )
            )
            async with asyncio.timeout(5):
                reply = self._decode(await connection.recv())
            if (
                set(reply)
                != {
                    "kind",
                    "schema_version",
                    "scope_sha256",
                    "channel_id",
                    "worker_instance",
                    "binding_sha256",
                }
                or reply["kind"] != "bind"
                or type(reply["schema_version"]) is not int
                or reply["schema_version"] != 1
                or reply["scope_sha256"] != self._scope
                or reply["channel_id"] != self._nonce
                or reply["worker_instance"] != daemon.visual_worker_instance
            ):
                raise GuestControlFailure()
            GuestControlFence._digest(reply["binding_sha256"])
            self._binding = reply["binding_sha256"]
            evidence = await daemon.bind_operator_control(self._binding)
            await connection.send(
                json.dumps({"kind": "bound", "channel_id": self._nonce, **evidence})
            )
            # The transport's keepalive alone does not renew operator authority.
            # Each receive has an idle bound, and each native command checks lease.
            async with asyncio.timeout(3600):
                while True:
                    async with asyncio.timeout(30):
                        message = self._decode(await connection.recv())
                    evidence = await self._command(message, connection)
                    await connection.send(
                        json.dumps(
                            {
                                "kind": "settled",
                                "channel_id": self._nonce,
                                "sequence": self._sequence,
                                **evidence,
                            }
                        )
                    )
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError) and (
                task is None or task.cancelling() <= cancellation_baseline
            ):
                failure = GuestControlFailure()
                failure.__cause__ = error
            else:
                failure = error
        finally:
            # Synchronous fencing precedes any potentially failing network close.
            if claimed:
                daemon.operator_frames.suspend()
                daemon.control.uncertain()
            try:
                await connection.close()
            except BaseException as cleanup:
                if failure is not None:
                    failure = BaseExceptionGroup(
                        "Browser control channel and close failed.", [failure, cleanup]
                    )
                else:
                    failure = cleanup
            finally:
                if claimed:
                    daemon.release_operator_channel(self._nonce)
        if failure is not None:
            raise failure

    async def _command(self, message: dict[str, Any], connection: Any) -> dict[str, Any]:
        common = {"kind", "channel_id", "worker_instance", "binding_sha256", "sequence"}
        if (
            not common <= message.keys()
            or message["channel_id"] != self._nonce
            or message["worker_instance"] != self._daemon.visual_worker_instance
            or message["binding_sha256"] != self._binding
            or self._daemon.control.binding_sha256 != self._binding
            or type(message["sequence"]) is not int
            or message["sequence"] != self._sequence + 1
            or message["sequence"] > 2**53 - 1
        ):
            raise GuestControlFailure()
        operation = message["kind"]
        self._daemon.control.expire()
        if operation == "status" and set(message) == common:
            result = self._daemon._operator_control_evidence()
        elif operation == "pages" and set(message) == common | {"epoch"}:
            result = await self._daemon.operator_page_descriptors(epoch=message["epoch"])
        elif operation == "view" and set(message) == common | {"view_id", "epoch", "until_ms"}:
            result = await self._daemon.grant_operator_view(
                binding_sha256=self._binding,
                view_id=message["view_id"],
                epoch=message["epoch"],
                until_ms=message["until_ms"],
            )
        elif operation == "frame" and set(message) == common | {
            "view_id",
            "epoch",
            "page_id",
            "page_epoch",
        }:
            sent = False

            async def send_frame(generation: int, metadata: dict[str, Any], frame: bytes) -> None:
                nonlocal sent
                header = json.dumps(
                    {
                        "kind": "frame",
                        "channel_id": self._nonce,
                        "worker_instance": self._daemon.visual_worker_instance,
                        "binding_sha256": self._binding,
                        "sequence": message["sequence"],
                        "view_id": message["view_id"],
                        "generation": generation,
                        **metadata,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                if len(header) > 4096:
                    raise GuestControlFailure()
                await connection.send(len(header).to_bytes(4, "big") + header + frame)
                sent = True

            try:
                await self._daemon.capture_operator_frame(
                    view_id=message["view_id"],
                    epoch=message["epoch"],
                    page_id=message["page_id"],
                    page_epoch=message["page_epoch"],
                    send=send_frame,
                )
            except GuestCaptureDenied:
                if sent:
                    raise GuestControlFailure() from None
            if not sent:
                # A denied admission or a positively settled discarded capture
                # has no pixels. Still bind its response to the exact exchange.
                header = json.dumps(
                    {
                        "kind": "frame_denied",
                        "channel_id": self._nonce,
                        "worker_instance": self._daemon.visual_worker_instance,
                        "binding_sha256": self._binding,
                        "sequence": message["sequence"],
                        "view_id": message["view_id"],
                        "control_epoch": message["epoch"],
                        "reason": "capture_unavailable",
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
                await connection.send(len(header).to_bytes(4, "big") + header)
            result = self._daemon._operator_control_evidence()
        elif operation == "takeover" and set(message) == common | {
            "request_id",
            "request_sha256",
            "expected_epoch",
            "expires_at_ms",
            "maximum_until_ms",
            "lease_until_ms",
            "pages",
        }:
            result = await self._daemon.acquire_operator_control(
                **{key: value for key, value in message.items() if key not in common},
                binding_sha256=self._binding,
            )
        elif operation == "handback" and set(message) == common | {"request_id", "epoch"}:
            result = await self._daemon.handback_operator_control(
                request_id=message["request_id"], epoch=message["epoch"]
            )
        elif operation == "sensitive" and set(message) == common | {"request_id", "epoch"}:
            result = await self._daemon.enter_operator_sensitive_entry(
                request_id=message["request_id"], epoch=message["epoch"]
            )
        elif operation == "text_input" and set(message) == common | {
            "request_id",
            "epoch",
            "input_sequence",
            "page_id",
            "page_epoch",
            "text",
        }:
            # The private channel is the only transport for this payload. Remove
            # it from the request before any native await or result publication.
            text = message.pop("text")
            try:
                result = await self._daemon.operator_text_input(
                    request_id=message["request_id"],
                    epoch=message["epoch"],
                    sequence=message["input_sequence"],
                    page_id=message["page_id"],
                    page_epoch=message["page_epoch"],
                    text=text,
                )
            finally:
                text = None
        elif operation == "key_input" and set(message) == common | {
            "request_id",
            "epoch",
            "input_sequence",
            "page_id",
            "page_epoch",
            "key",
        }:
            result = await self._daemon.operator_key_input(
                request_id=message["request_id"],
                epoch=message["epoch"],
                sequence=message["input_sequence"],
                page_id=message["page_id"],
                page_epoch=message["page_epoch"],
                key=message["key"],
            )
        elif operation == "renew" and set(message) == common | {
            "request_id",
            "epoch",
            "lease_until_ms",
        }:
            self._daemon.control.renew(
                request_id=message["request_id"],
                epoch=message["epoch"],
                lease_until_ms=message["lease_until_ms"],
            )
            result = self._daemon._operator_control_evidence()
            result["lease_until_ms"] = self._daemon.control.lease_wall_ms
        else:
            raise GuestControlFailure()
        self._sequence = message["sequence"]
        return result


class GuestControlFence:
    def __init__(
        self,
        *,
        worker_instance: str,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.worker_instance = worker_instance
        self.binding_sha256: str | None = None
        self.state = "unbound"
        self.epoch = 1
        self.request_id: str | None = None
        self.request_sha256: str | None = None
        self.request_epoch: int | None = None
        self.handback_source_epoch: int | None = None
        self.acquisition_audit_json: str | None = None
        self.handback_audit_json: str | None = None
        self.request_until = 0.0
        self.maximum_until = 0.0
        self.lease_until = 0.0
        self.lease_wall_ms: int | None = None
        self.maximum_wall_ms = 0
        self.settled_sequence = 0
        self.pending_sequence: int | None = None
        self.fresh_observation_required = False
        self.sensitive_entry = False
        self.capture_restricted = False
        self.view_id: str | None = None
        self.view_until = 0.0
        self.view_epoch = 0
        self._monotonic = monotonic
        self._wall_clock = wall_clock

    def grant_view(self, *, binding_sha256: str, view_id: str, epoch: int, until_ms: int) -> None:
        self.expire()
        if (
            self.binding_sha256 is None
            or self.binding_sha256 != binding_sha256
            or type(epoch) is not int
            or epoch != self.epoch
            or self.state not in {"agent_controlled", "takeover_requested", "operator_controlled"}
            or self.sensitive_entry
            or type(view_id) is not str
            or re.fullmatch(r"bv_[0-9a-f]{32}", view_id) is None
        ):
            raise GuestControlFailure()
        deadline = self._deadline(until_ms, maximum_seconds=30)
        self.view_id = view_id
        self.view_epoch = epoch
        self.view_until = deadline

    def check_view(self, *, view_id: str, epoch: int) -> None:
        self.expire()
        if (
            type(epoch) is not int
            or epoch != self.epoch
            or epoch != self.view_epoch
            or view_id != self.view_id
            or self._monotonic() >= self.view_until
            or self.sensitive_entry
            or self.state not in {"agent_controlled", "takeover_requested", "operator_controlled"}
        ):
            raise GuestControlFailure()

    def prepare_rebind(self) -> None:
        """Called only after exact egress rotation and old channel settlement."""
        if (
            self.binding_sha256 is None
            or self.state not in {"agent_controlled", "control_uncertain"}
            or self.request_id is not None
            or self.sensitive_entry
            or self.capture_restricted
            or self.pending_sequence is not None
            or self.settled_sequence != 0
        ):
            raise GuestControlFailure()
        self.binding_sha256 = None
        self.state = "unbound"
        self.epoch += 1
        self.fresh_observation_required = True
        self.view_id = None
        self.view_until = 0.0
        self.view_epoch = 0

    def bind(self, binding_sha256: str) -> None:
        self._digest(binding_sha256)
        if self.binding_sha256 is not None:
            if self.binding_sha256 != binding_sha256:
                raise GuestControlFailure()
            return
        self.binding_sha256 = binding_sha256
        self.state = "agent_controlled"

    def check_model(self, epoch: int | None, operation: str) -> None:
        self.expire()
        if self.binding_sha256 is None:
            if epoch is not None:
                raise GuestControlFailure()
            return
        if type(epoch) is not int or epoch != self.epoch or self.state != "agent_controlled":
            raise GuestControlFailure()
        if self.fresh_observation_required and operation not in {"observe", "close"}:
            raise GuestControlFailure()
        if self.capture_restricted and operation in {
            "screenshot",
            "download",
            "observe_visual",
            "click_visual_target",
            "click_visual_point",
        }:
            raise GuestControlFailure()

    def request(
        self,
        *,
        binding_sha256: str,
        request_id: str,
        request_sha256: str,
        expected_epoch: int,
        expires_at_ms: int,
        maximum_until_ms: int,
    ) -> None:
        self._digest(request_sha256)
        if type(request_id) is not str or re.fullmatch(r"bt_[0-9a-f]{32}", request_id) is None:
            raise GuestControlFailure()
        self.expire()
        if self.binding_sha256 != binding_sha256 or self.binding_sha256 is None:
            raise GuestControlFailure()
        if self.request_id == request_id:
            if self.request_sha256 != request_sha256:
                raise GuestControlFailure()
            # Historical readback does not renew or grant native input.
            return
        if (
            type(expected_epoch) is not int
            or not 1 <= expected_epoch <= 2**53 - 3
            or self.epoch != expected_epoch
            or self.state != "agent_controlled"
        ):
            raise GuestControlFailure()
        request_until = self._deadline(expires_at_ms, maximum_seconds=3600)
        maximum_until = self._deadline(maximum_until_ms, maximum_seconds=3600)
        if expires_at_ms > maximum_until_ms:
            raise GuestControlFailure()
        self.request_id = request_id
        self.request_sha256 = request_sha256
        self.request_epoch = expected_epoch
        self.handback_source_epoch = None
        self.acquisition_audit_json = None
        self.handback_audit_json = None
        self.request_until = request_until
        self.maximum_until = maximum_until
        self.maximum_wall_ms = maximum_until_ms
        # Synchronous fencing precedes waiting for the daemon lifecycle lock.
        self.state = "takeover_requested"

    def grant(self, *, request_id: str, lease_until_ms: int) -> None:
        self.expire()
        if self.state != "takeover_requested" or request_id != self.request_id:
            raise GuestControlFailure()
        deadline = self._deadline(lease_until_ms, maximum_seconds=60)
        if lease_until_ms > self.maximum_wall_ms:
            raise GuestControlFailure()
        self.lease_until = min(deadline, self.maximum_until)
        self.lease_wall_ms = lease_until_ms
        self.epoch += 1
        self.state = "operator_controlled"

    def renew(self, *, request_id: str, epoch: int, lease_until_ms: int) -> None:
        self.check_operator(request_id=request_id, epoch=epoch)
        if self.pending_sequence is not None:
            raise GuestControlFailure()
        deadline = self._deadline(lease_until_ms, maximum_seconds=60)
        if lease_until_ms > self.maximum_wall_ms or deadline <= self.lease_until:
            raise GuestControlFailure()
        self.lease_until = min(deadline, self.maximum_until)
        self.lease_wall_ms = lease_until_ms

    def check_operator(self, *, request_id: str, epoch: int) -> None:
        self.expire()
        if (
            type(epoch) is not int
            or epoch != self.epoch
            or request_id != self.request_id
            or self.state != "operator_controlled"
        ):
            raise GuestControlFailure()

    def begin_input(self, *, request_id: str, epoch: int, sequence: int) -> None:
        self.check_operator(request_id=request_id, epoch=epoch)
        if (
            type(sequence) is not int
            or not 1 <= sequence <= 2**53 - 1
            or sequence != self.settled_sequence + 1
            or self.pending_sequence is not None
        ):
            raise GuestControlFailure()
        self.pending_sequence = sequence

    def settle_input(self, sequence: int) -> None:
        if type(sequence) is not int or self.pending_sequence != sequence:
            raise GuestControlFailure()
        self.settled_sequence = sequence
        self.pending_sequence = None
        # A late acknowledgement cannot undo expiry/disconnect fencing.
        self.expire()

    def begin_handback(self, *, request_id: str, epoch: int) -> bool:
        self.expire()
        if (
            self.request_id is not None
            and self.request_id == request_id
            and type(epoch) is int
            and epoch == self.handback_source_epoch
            and self.state == "agent_controlled"
        ):
            return False
        if (
            self.request_id is None
            or type(epoch) is not int
            or epoch != self.epoch
            or request_id != self.request_id
            or self.state not in {"operator_controlled", "control_uncertain", "handback_pending"}
        ):
            raise GuestControlFailure()
        self.state = "handback_pending"
        self.handback_source_epoch = epoch
        return True

    def finish_handback(self) -> None:
        if (
            self.state != "handback_pending"
            or self.pending_sequence is not None
            or self.request_epoch is None
        ):
            raise GuestControlFailure()
        self.epoch = self.request_epoch + 2
        self.state = "agent_controlled"
        self.lease_until = 0.0
        self.sensitive_entry = False
        self.fresh_observation_required = True

    def uncertain(self) -> None:
        if self.binding_sha256 is not None and self.state != "closed":
            self.state = "control_uncertain"

    def expire(self) -> None:
        now = self._monotonic()
        if (self.state == "takeover_requested" and now >= self.request_until) or (
            self.state == "operator_controlled" and now >= self.lease_until
        ):
            self.uncertain()

    def request_remaining_seconds(self) -> float:
        return max(0.0, self.request_until - self._monotonic())

    def retirement_deferral_seconds(self) -> float:
        """Keep the allocation through its admitted, bounded control interval.

        Expired or disconnected input remains fenced. The original maximum,
        never channel activity, bounds time available for explicit settlement.
        """
        self.expire()
        if self.binding_sha256 is None or self.request_id is None:
            return 0.0
        if self.state == "takeover_requested":
            deadline = min(self.request_until, self.maximum_until)
        elif self.state in {"operator_controlled", "handback_pending", "control_uncertain"}:
            deadline = self.maximum_until
        else:
            return 0.0
        return max(0.0, deadline - self._monotonic())

    def _deadline(self, wall_ms: int, *, maximum_seconds: int) -> float:
        if type(wall_ms) is not int or not 0 <= wall_ms <= 2**53 - 1:
            raise GuestControlFailure()
        remaining = wall_ms / 1000 - self._wall_clock()
        if not math.isfinite(remaining) or not 0 < remaining <= maximum_seconds:
            raise GuestControlFailure()
        return self._monotonic() + remaining

    @staticmethod
    def _digest(value: str) -> None:
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise GuestControlFailure()
