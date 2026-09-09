"""Protected recording ingestion and application-authorized playback routes."""

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from starlette.responses import JSONResponse, Response

from cayu._browser_recording_store import BrowserRecordingStore, BrowserRecordingUnavailable
from cayu.browser_recording import (
    BrowserRecordingManifest,
)
from cayu.server.auth import AuthContext, AuthDependency, server_auth_dependency

RecordingAccess = Callable[
    [AuthContext, BrowserRecordingManifest, Literal["view", "download"]], Awaitable[bool]
]


class BrowserRecordingServer:
    """Explicit persistence and playback policy, independent of live-view access."""

    def __init__(self, *, store: BrowserRecordingStore, authorize: RecordingAccess) -> None:
        if not isinstance(store, BrowserRecordingStore) or not callable(authorize):
            raise TypeError("Recording requires a private store and application access policy.")
        self.store = store
        self.authorize = authorize
        self._connections: set[WebSocket] = set()
        self._maintenance: asyncio.Task[None] | None = None

    def router(self, *, auth: AuthDependency) -> APIRouter:
        router = APIRouter(prefix="/browser-recordings")
        authenticated = server_auth_dependency(auth)

        async def admitted(
            recording_id: str, principal: AuthContext, purpose: Literal["view", "download"]
        ) -> BrowserRecordingManifest:
            try:
                manifest = await self.store.manifest(recording_id)
                if await self.authorize(principal, manifest, purpose) is not True:
                    raise BrowserRecordingUnavailable()
                return manifest
            except BrowserRecordingUnavailable:
                raise HTTPException(404, "Browser recording is unavailable.") from None

        @router.get("/sessions/{session_id}")
        async def recordings(
            session_id: str, principal: Annotated[AuthContext, Depends(authenticated)]
        ):
            result = []
            for recording_id in await self.store.recordings_for_session(session_id):
                try:
                    receipt = await admitted(recording_id, principal, "view")
                    result.append(
                        {
                            **receipt.model_dump(mode="json"),
                            "can_download": await self.authorize(principal, receipt, "download")
                            is True,
                        }
                    )
                except HTTPException:
                    continue
            return JSONResponse(result, headers={"Cache-Control": "no-store"})

        @router.get("/{recording_id}")
        async def manifest(
            recording_id: str, principal: Annotated[AuthContext, Depends(authenticated)]
        ):
            receipt = await admitted(recording_id, principal, "view")
            return JSONResponse(
                {
                    **receipt.model_dump(mode="json"),
                    "can_download": await self.authorize(principal, receipt, "download") is True,
                },
                headers={"Cache-Control": "no-store"},
            )

        @router.get("/{recording_id}/media")
        @router.get("/{recording_id}/segments/{sequence}")
        async def segment(
            recording_id: str,
            request: Request,
            principal: Annotated[AuthContext, Depends(authenticated)],
            download: bool = False,
            sequence: int | None = None,
        ):
            manifest = await admitted(recording_id, principal, "download" if download else "view")
            if manifest.status not in {"complete", "partial"}:
                raise HTTPException(404, "Browser recording is unavailable.")
            try:
                media = (
                    await self.store.video(recording_id)
                    if sequence is None
                    else await self.store.media(recording_id, sequence)
                )
            except BrowserRecordingUnavailable:
                raise HTTPException(404, "Browser recording is unavailable.") from None
            filename = (
                f"recording-{recording_id[:12]}.webm"
                if sequence is None
                else f"recording-segment-{sequence}.webm"
            )
            headers = {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Disposition": f'{"attachment" if download else "inline"}; filename="{filename}"',
                "Accept-Ranges": "bytes",
            }
            status = 200
            ranges = request.headers.getlist("range")
            if ranges:
                import re

                match = (
                    re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", ranges[0])
                    if len(ranges) == 1
                    else None
                )
                if match is None or not any(match.groups()) or len(ranges[0]) > 80:
                    raise HTTPException(416, "Invalid media range.")
                first, last = match.groups()
                size = len(media)
                start = int(first) if first else max(0, size - int(last))
                end = min(size - 1, int(last)) if first and last else size - 1
                if start > end or start >= size:
                    raise HTTPException(416, "Invalid media range.")
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                media = media[start : end + 1]
                status = 206
            return Response(media, media_type="video/webm", headers=headers, status_code=status)

        @router.websocket("/guest")
        async def guest(socket: WebSocket):
            # A capture credential is not an operator credential. Browser-origin
            # connections are excluded even when they possess server login state.
            if (
                socket.url.scheme != "wss"
                or socket.query_params
                or socket.headers.getlist("origin")
                or socket.scope.get("subprotocols") != ["cayu.browser-recording.v1"]
                or len(self._connections) >= 32
            ):
                await socket.close(code=1008)
                return
            headers = socket.headers.getlist("authorization")
            if len(headers) != 1 or not headers[0].startswith("Bearer "):
                await socket.close(code=1008)
                return
            self._connections.add(socket)
            try:
                policy, session_id = await self.store.authenticate_capture(headers[0][7:])
            except BrowserRecordingUnavailable:
                self._connections.discard(socket)
                await socket.close(code=1008)
                return
            except BaseException:
                self._connections.discard(socket)
                raise
            headers.clear()
            try:
                await socket.accept(subprotocol="cayu.browser-recording.v1")
            except BaseException:
                self._connections.discard(socket)
                raise
            recording_id = None
            try:
                async with asyncio.timeout(5):
                    hello_text = await socket.receive_text()
                    if len(hello_text) > 4096:
                        raise BrowserRecordingUnavailable()
                    hello = json.loads(hello_text)
                    if (
                        type(hello) is not dict
                        or set(hello) != {"identity", "started_at_ms"}
                        or hello["identity"].get("session_id") != session_id
                    ):
                        raise BrowserRecordingUnavailable()
                    recording_id = await self.store.begin(policy, hello["identity"])
                    owner = await self.store.claim(recording_id)
                    await socket.send_json({"ready": True})
                async with asyncio.timeout(policy.max_duration_seconds + 10):
                    for _ in range(policy.max_duration_seconds * policy.frames_per_second + 2):
                        async with asyncio.timeout(15):
                            message = await socket.receive()
                        if message["type"] == "websocket.disconnect":
                            break
                        raw = message.get("bytes")
                        if raw is not None:
                            if not 4 < len(raw) <= 2 * 1024 * 1024 + 4096:
                                raise BrowserRecordingUnavailable()
                            size = int.from_bytes(raw[:4], "big")
                            if not 1 <= size <= 4096 or size + 4 >= len(raw):
                                raise BrowserRecordingUnavailable()
                            metadata = json.loads(raw[4 : 4 + size])
                            if type(metadata) is not dict or set(metadata) != {
                                "sequence",
                                "page_id",
                                "elapsed_ms",
                            }:
                                raise BrowserRecordingUnavailable()
                            try:
                                await self.store.append(
                                    recording_id,
                                    policy=policy,
                                    owner=owner,
                                    png=raw[4 + size :],
                                    **metadata,
                                )
                            except BrowserRecordingUnavailable as failure:
                                await self.store.finish(
                                    recording_id,
                                    owner=owner,
                                    reason=failure.reason,
                                    gap=True,
                                    elapsed_ms=metadata["elapsed_ms"],
                                )
                                await socket.send_json({"stop": failure.reason})
                                break
                            await socket.send_json({"accepted": metadata["sequence"]})
                        else:
                            raw = message.get("text")
                            if type(raw) is not str or len(raw) > 4096:
                                raise BrowserRecordingUnavailable()
                            final = json.loads(raw)
                            if type(final) is dict and set(final) == {"heartbeat"}:
                                await self.store.heartbeat(
                                    recording_id, owner=owner, elapsed_ms=final["heartbeat"]
                                )
                                await socket.send_json({"heartbeat": True})
                                continue
                            if (
                                type(final) is not dict
                                or set(final) != {"finish", "gap", "elapsed_ms"}
                                or type(final["gap"]) is not bool
                            ):
                                raise BrowserRecordingUnavailable()
                            await self.store.finish(
                                recording_id,
                                owner=owner,
                                reason=final["finish"],
                                gap=final["gap"],
                                elapsed_ms=final["elapsed_ms"],
                            )
                            await socket.send_json({"finalized": True})
                            break
            except Exception:
                # A disconnected worker may reconnect to the same recording.
                # Only explicit recovery may seal its unacknowledged tail.
                pass
            finally:
                self._connections.discard(socket)
                with contextlib.suppress(Exception):
                    await socket.close()

        return router

    def start(self) -> None:
        if self._maintenance is not None:
            return

        async def maintain() -> None:
            while True:
                try:
                    await self.store.purge_expired()
                    await self.store.recover_abandoned()
                except Exception:
                    # Storage failures cannot grant retrieval or replay input.
                    pass
                await asyncio.sleep(30)

        self._maintenance = asyncio.create_task(maintain(), name="cayu-recording-retention")

    async def drain(self) -> None:
        if self._maintenance is not None:
            self._maintenance.cancel()
            await asyncio.gather(self._maintenance, return_exceptions=True)
            self._maintenance = None
        # Shutdown does not manufacture a normal-completion receipt.
        await asyncio.gather(
            *(socket.close(code=1012) for socket in tuple(self._connections)),
            return_exceptions=True,
        )
