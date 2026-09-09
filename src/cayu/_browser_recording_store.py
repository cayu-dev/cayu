"""Private, transactional recording publication; never an ordinary artifact store."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import secrets
import sqlite3
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import SecretStr

from cayu.browser_recording import (
    BrowserRecordingConfig,
    BrowserRecordingGap,
    BrowserRecordingManifest,
    BrowserRecordingPolicy,
    BrowserRecordingSegment,
)

_T = TypeVar("_T")


async def _settle_owned(task: asyncio.Task[_T]) -> _T:
    cancellation = None
    while True:
        try:
            value = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                raise
            cancellation = error
        except BaseException as error:
            if cancellation is not None:
                raise cancellation from error
            raise
    if cancellation is not None:
        raise cancellation
    return value


class BrowserRecordingUnavailable(RuntimeError):
    """Bounded failure with no browser content or credentials."""

    def __init__(self, reason: str = "storage_failure") -> None:
        super().__init__("Browser recording is unavailable.")
        self.reason = (
            reason if reason in {"storage_failure", "limit_exhausted"} else "storage_failure"
        )


async def _bounded_process(
    command: list[str],
    content: bytes,
    maximum: int,
    *,
    pass_fds: tuple[int, ...] = (),
    timeout: int = 10,
) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        pass_fds=pass_fds,
    )
    assert process.stdin is not None and process.stdout is not None

    async def write() -> None:
        assert process.stdin is not None
        process.stdin.write(content)
        await process.stdin.drain()
        process.stdin.close()

    writer = asyncio.create_task(write())
    try:
        async with asyncio.timeout(timeout):
            output = await process.stdout.read(maximum + 1)
            # StreamReader.read may return before EOF.
            while len(output) <= maximum:
                chunk = await process.stdout.read(min(65536, maximum + 1 - len(output)))
                if not chunk:
                    break
                output += chunk
            if len(output) > maximum:
                raise BrowserRecordingUnavailable("limit_exhausted")
            await writer
            if await process.wait() != 0:
                raise BrowserRecordingUnavailable()
            return output
    finally:

        async def cleanup() -> None:
            if process.returncode is None:
                process.kill()
            await process.wait()
            writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)

        await _settle_owned(asyncio.create_task(cleanup()))


async def encode_recording_frame(
    png: bytes, policy: BrowserRecordingPolicy, *, ffmpeg: str
) -> bytes:
    """Encode an already-admitted frame through pipes; no encoder temporary files."""
    from PIL import Image

    if type(png) is not bytes or not 1 <= len(png) <= 2 * 1024 * 1024:
        raise BrowserRecordingUnavailable()
    try:
        with Image.open(io.BytesIO(png)) as frame:
            if (
                frame.format != "PNG"
                or frame.width > policy.max_width
                or frame.height > policy.max_height
                or frame.width < 1
                or frame.height < 1
            ):
                raise BrowserRecordingUnavailable()
            frame.verify()
    except Exception:
        raise BrowserRecordingUnavailable() from None
    video = await _bounded_process(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "image2pipe",
            "-framerate",
            str(policy.frames_per_second),
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            "libvpx",
            "-threads",
            "1",
            "-deadline",
            "realtime",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "webm",
            "pipe:1",
        ],
        png,
        min(policy.max_bytes, 2 * 1024 * 1024),
    )
    if not video.startswith(b"\x1aE\xdf\xa3"):
        raise BrowserRecordingUnavailable()
    # Decoder success is required before the transaction can publish a segment.
    await _bounded_process(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-xerror",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        video,
        4096,
    )
    return video


class BrowserRecordingStore:
    """Private local SQLite media store with bounded publication and retention.

    Keep this database outside model-accessible workspaces. SQLite transactions
    commit metadata and media together: acknowledgement loss can never publish
    an orphan or require replaying an action. Applications supply retrieval
    authorization separately to the recording router. Expired data is denied on
    every read; call ``purge_expired`` periodically for physical reclamation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_storage_bytes: int = 512 * 1024 * 1024,
        max_recordings: int = 128,
        ffmpeg: str = "ffmpeg",
    ) -> None:
        if (
            type(max_storage_bytes) is not int
            or not 1024 * 1024 <= max_storage_bytes <= 16 * 1024**3
        ):
            raise ValueError("Recording storage bound is invalid.")
        if type(max_recordings) is not int or not 1 <= max_recordings <= 1024:
            raise ValueError("Recording count bound is invalid.")
        if os.name != "posix":
            raise ValueError("Local browser recording storage requires a POSIX application worker.")
        try:
            import shutil

            from PIL import Image  # noqa: F401

            if shutil.which(ffmpeg) is None:
                raise ValueError("Browser recording requires FFmpeg on the application worker.")
        except ImportError:
            raise ValueError("Browser recording requires cayu[recording].") from None
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Do not follow an existing symlink or create a world-readable database.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
        self.max_storage_bytes = max_storage_bytes
        self.max_recordings = max_recordings
        self.ffmpeg = ffmpeg
        with self._connect() as db:
            db.executescript("""
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                PRAGMA secure_delete=ON;
                CREATE TABLE IF NOT EXISTS grants (
                  scope TEXT PRIMARY KEY, token_sha256 TEXT NOT NULL, policy TEXT NOT NULL,
                  session_id TEXT NOT NULL, expires INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS recordings (
                  id TEXT PRIMARY KEY, scope TEXT NOT NULL, identity TEXT NOT NULL,
                  status TEXT NOT NULL, reason TEXT NOT NULL, started INTEGER NOT NULL,
                  expires INTEGER NOT NULL, last_seen INTEGER NOT NULL, gap INTEGER NOT NULL DEFAULT 0, owner TEXT NOT NULL DEFAULT '',
                  elapsed_end INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS videos (
                  recording TEXT PRIMARY KEY, sha256 TEXT NOT NULL, media BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS segments (
                  recording TEXT NOT NULL, sequence INTEGER NOT NULL, receipt TEXT NOT NULL,
                  source_sha256 TEXT NOT NULL, media BLOB NOT NULL,
                  PRIMARY KEY(recording,sequence));
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=0)
        db.execute("PRAGMA busy_timeout=0")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA secure_delete=ON")
        # Bound the database itself, including free pages and indexes. Rollback
        # journals may temporarily consume up to one additional database bound.
        db.execute(f"PRAGMA max_page_count={self.max_storage_bytes // (4 * 4096)}")
        return db

    async def _transaction(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        def run() -> _T:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                return operation(db)

        until = time.monotonic() + 5
        while True:
            task = asyncio.create_task(asyncio.to_thread(run))
            try:
                # The exact transaction settles even if its caller is cancelled.
                return await _settle_owned(task)
            except sqlite3.OperationalError as error:
                if (
                    getattr(error, "sqlite_errorcode", None)
                    in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
                    and time.monotonic() < until
                ):
                    await asyncio.sleep(0.01)
                    continue
                raise BrowserRecordingUnavailable() from None

    async def authorize_capture(
        self, *, session_id: str, policy: BrowserRecordingPolicy, guest_endpoint: str
    ) -> BrowserRecordingConfig:
        """Application call before the run; never exposed as an HTTP/model mutation."""
        policy = BrowserRecordingPolicy.model_validate(policy)
        if not session_id or len(session_id) > 128:
            raise ValueError("Recording requires an exact application session.")
        token = secrets.token_hex(32)
        config = BrowserRecordingConfig(
            policy=policy,
            session_id=session_id,
            guest_endpoint=guest_endpoint,
            credential=SecretStr(token),
        )
        now = int(time.time() * 1000)

        def issue(db: sqlite3.Connection) -> None:
            if db.execute("SELECT 1 FROM grants WHERE scope=?", (policy.scope,)).fetchone():
                raise BrowserRecordingUnavailable()
            if db.execute("SELECT COUNT(*) FROM grants").fetchone()[0] >= self.max_recordings:
                raise BrowserRecordingUnavailable()
            db.execute(
                "INSERT INTO grants VALUES (?,?,?,?,?)",
                (
                    policy.scope,
                    hashlib.sha256(token.encode()).hexdigest(),
                    policy.model_dump_json(),
                    session_id,
                    now + policy.retention_seconds * 1000,
                ),
            )

        await self._transaction(issue)
        return config

    async def authenticate_capture(self, token: str) -> tuple[BrowserRecordingPolicy, str]:
        if type(token) is not str or len(token) != 64 or not token.isascii():
            raise BrowserRecordingUnavailable()
        digest = hashlib.sha256(token.encode()).hexdigest()

        def read(db: sqlite3.Connection) -> tuple[BrowserRecordingPolicy, str]:
            row = db.execute(
                "SELECT policy,session_id FROM grants WHERE token_sha256=? AND expires>?",
                (digest, int(time.time() * 1000)),
            ).fetchone()
            if row is None:
                raise BrowserRecordingUnavailable()
            return BrowserRecordingPolicy.model_validate_json(row[0]), row[1]

        return await self._transaction(read)

    async def begin(self, policy: BrowserRecordingPolicy, identity: dict[str, str]) -> str:
        from cayu.browser_recording import BrowserRecordingIdentity

        owned = BrowserRecordingIdentity.model_validate(identity)
        material = owned.model_dump_json()
        recording_id = hashlib.sha256((policy.scope + "\0" + material).encode()).hexdigest()
        now = int(time.time() * 1000)

        def begin(db: sqlite3.Connection) -> str:
            grant = db.execute(
                "SELECT policy,session_id,expires FROM grants WHERE scope=?", (policy.scope,)
            ).fetchone()
            if (
                grant is None
                or grant[0] != policy.model_dump_json()
                or grant[1] != owned.session_id
                or grant[2] <= now
            ):
                raise BrowserRecordingUnavailable()
            row = db.execute(
                "SELECT identity FROM recordings WHERE id=?", (recording_id,)
            ).fetchone()
            if row:
                if row[0] != material:
                    raise BrowserRecordingUnavailable()
                return recording_id
            if db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] >= self.max_recordings:
                raise BrowserRecordingUnavailable()
            db.execute(
                "INSERT INTO recordings (id,scope,identity,status,reason,started,expires,last_seen,gap) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    recording_id,
                    policy.scope,
                    material,
                    "recording",
                    "recording",
                    now,
                    min(grant[2], now + policy.retention_seconds * 1000),
                    now,
                    0,
                ),
            )
            return recording_id

        return await self._transaction(begin)

    async def claim(self, recording_id: str) -> str:
        owner = secrets.token_hex(32)

        def claim(db: sqlite3.Connection) -> str:
            changed = db.execute(
                "UPDATE recordings SET owner=?,last_seen=? WHERE id=? AND status='recording' AND expires>?",
                (owner, int(time.time() * 1000), recording_id, int(time.time() * 1000)),
            ).rowcount
            if changed != 1:
                raise BrowserRecordingUnavailable()
            return owner

        return await self._transaction(claim)

    async def heartbeat(self, recording_id: str, *, owner: str, elapsed_ms: int) -> None:
        if type(elapsed_ms) is not int or not 0 <= elapsed_ms <= 3600000:
            raise BrowserRecordingUnavailable()

        def heartbeat(db: sqlite3.Connection) -> None:
            if (
                db.execute(
                    "UPDATE recordings SET last_seen=?,elapsed_end=MAX(elapsed_end,?) WHERE id=? AND owner=? AND status='recording'",
                    (int(time.time() * 1000), elapsed_ms, recording_id, owner),
                ).rowcount
                != 1
            ):
                raise BrowserRecordingUnavailable()

        await self._transaction(heartbeat)

    async def recordings_for_session(self, session_id: str) -> list[str]:
        def read(db: sqlite3.Connection) -> list[str]:
            return [
                row[0]
                for row in db.execute(
                    "SELECT r.id FROM recordings r JOIN grants g ON g.scope=r.scope WHERE g.session_id=? AND r.expires>? AND r.status != 'deleted' ORDER BY r.started,r.id",
                    (session_id, int(time.time() * 1000)),
                )
            ]

        return await self._transaction(read)

    async def append(
        self,
        recording_id: str,
        *,
        policy: BrowserRecordingPolicy,
        owner: str,
        sequence: int,
        page_id: str,
        elapsed_ms: int,
        png: bytes,
    ) -> None:
        if (
            type(sequence) is not int
            or not 0 <= sequence < policy.max_duration_seconds * policy.frames_per_second
            or type(elapsed_ms) is not int
            or not 0 <= elapsed_ms < policy.max_duration_seconds * 1000
            or not sequence * 1000 // policy.frames_per_second
            <= elapsed_ms
            <= (sequence + 1) * 1000 // policy.frames_per_second
            or type(page_id) is not str
            or type(png) is not bytes
            or len(png) > 2 * 1024 * 1024
        ):
            raise BrowserRecordingUnavailable()
        source = hashlib.sha256(png).hexdigest()

        # Do not encode an acknowledged duplicate or accept conflicting retries.
        def existing(db: sqlite3.Connection) -> bool:
            grant = db.execute(
                "SELECT policy FROM grants WHERE scope=?", (policy.scope,)
            ).fetchone()
            if grant is None or grant[0] != policy.model_dump_json():
                raise BrowserRecordingUnavailable()
            lease = db.execute(
                "SELECT scope,owner,expires FROM recordings WHERE id=?", (recording_id,)
            ).fetchone()
            if (
                lease is None
                or lease[0] != policy.scope
                or lease[1] != owner
                or lease[2] <= int(time.time() * 1000)
            ):
                raise BrowserRecordingUnavailable()
            row = db.execute(
                "SELECT source_sha256,receipt FROM segments WHERE recording=? AND sequence=?",
                (recording_id, sequence),
            ).fetchone()
            if row:
                receipt = BrowserRecordingSegment.model_validate_json(row[1])
                if (
                    row[0] != source
                    or receipt.page_id != page_id
                    or receipt.elapsed_ms != elapsed_ms
                ):
                    raise BrowserRecordingUnavailable()
                return True
            return False

        if await self._transaction(existing):
            return
        media = await encode_recording_frame(png, policy, ffmpeg=self.ffmpeg)
        receipt = BrowserRecordingSegment(
            sequence=sequence,
            page_id=page_id,
            elapsed_ms=elapsed_ms,
            duration_ms=(sequence + 1) * 1000 // policy.frames_per_second
            - sequence * 1000 // policy.frames_per_second,
            sha256=hashlib.sha256(media).hexdigest(),
            size_bytes=len(media),
        )

        def append(db: sqlite3.Connection) -> None:
            if existing(db):
                return
            row = db.execute(
                "SELECT scope,status,expires FROM recordings WHERE id=?", (recording_id,)
            ).fetchone()
            if (
                row is None
                or row[0] != policy.scope
                or row[1] != "recording"
                or row[2] <= int(time.time() * 1000)
            ):
                raise BrowserRecordingUnavailable()
            total, last = db.execute(
                "SELECT COALESCE(SUM(length(media)),0), MAX(sequence) FROM segments WHERE recording=?",
                (recording_id,),
            ).fetchone()
            if (
                (last is not None and sequence <= last)
                or elapsed_ms >= policy.max_duration_seconds * 1000
                or total + len(media) > policy.max_bytes
            ):
                raise BrowserRecordingUnavailable("limit_exhausted")
            gap = sequence != (0 if last is None else last + 1)
            db.execute(
                "INSERT INTO segments VALUES (?,?,?,?,?)",
                (recording_id, sequence, receipt.model_dump_json(), source, media),
            )
            db.execute(
                "UPDATE recordings SET last_seen=?,gap=MAX(gap,?),elapsed_end=MAX(elapsed_end,?) WHERE id=?",
                (int(time.time() * 1000), int(gap), elapsed_ms + receipt.duration_ms, recording_id),
            )

        await self._transaction(append)

    async def finish(
        self,
        recording_id: str,
        *,
        reason: str = "normal_close",
        gap: bool = False,
        owner: str | None = None,
        elapsed_ms: int = 0,
    ) -> None:
        if reason not in {
            "normal_close",
            "capture_gap",
            "limit_exhausted",
            "worker_lost",
            "storage_failure",
        }:
            raise ValueError("Invalid recording finalization reason.")
        if type(elapsed_ms) is not int or not 0 <= elapsed_ms <= 3600000:
            raise BrowserRecordingUnavailable()

        def prepare(db: sqlite3.Connection):
            row = db.execute(
                "SELECT status,gap,owner,scope,reason,elapsed_end FROM recordings WHERE id=?",
                (recording_id,),
            ).fetchone()
            if row is None or (owner is not None and row[2] != owner):
                raise BrowserRecordingUnavailable()
            if row[0] not in {"recording", "finalizing"}:
                return None
            frames = list(
                db.execute(
                    "SELECT receipt,media FROM segments WHERE recording=? ORDER BY sequence",
                    (recording_id,),
                )
            )
            policy_row = db.execute("SELECT policy FROM grants WHERE scope=?", (row[3],)).fetchone()
            if policy_row is None:
                raise BrowserRecordingUnavailable()
            selected = BrowserRecordingPolicy.model_validate_json(policy_row[0])
            partial = gap or bool(row[1]) or reason != "normal_close" or row[0] == "finalizing"
            previous = 0
            for raw_receipt, _media in frames:
                receipt = BrowserRecordingSegment.model_validate_json(raw_receipt)
                partial |= receipt.elapsed_ms > previous + 50
                previous = receipt.elapsed_ms + receipt.duration_ms
            partial |= max(row[5], elapsed_ms) > previous + 50
            final_reason = "capture_gap" if partial and reason == "normal_close" else reason
            # This write fences all further capture before media finalization.
            db.execute(
                "UPDATE recordings SET status='finalizing',reason=?,gap=?,elapsed_end=MAX(elapsed_end,?) WHERE id=?",
                (final_reason, int(partial), elapsed_ms, recording_id),
            )
            return frames, selected, partial, final_reason

        prepared = await self._transaction(prepare)
        if prepared is None:
            return
        frames, selected, partial, final_reason = prepared
        video = None
        if frames:
            try:
                video = await self._combine(frames, selected, recording_id=recording_id)
            except BrowserRecordingUnavailable as failure:
                partial, final_reason = True, failure.reason
            except Exception:
                partial, final_reason = True, "storage_failure"
        status = (
            ("partial" if partial else "complete")
            if frames
            else ("failed" if final_reason == "storage_failure" else "unavailable")
        )

        def publish(db: sqlite3.Connection) -> None:
            current = db.execute(
                "SELECT status FROM recordings WHERE id=?", (recording_id,)
            ).fetchone()
            if current is None or current[0] != "finalizing":
                return
            if video is not None:
                db.execute(
                    "INSERT INTO videos VALUES (?,?,?)",
                    (recording_id, hashlib.sha256(video).hexdigest(), video),
                )
            db.execute(
                "UPDATE recordings SET status=?,reason=? WHERE id=?",
                (status, final_reason, recording_id),
            )

        try:
            await self._transaction(publish)
        except BrowserRecordingUnavailable:
            # The combined video needs capacity in addition to its committed
            # segments. A full database rolls back that entire publication,
            # including the terminal status. Settle metadata without adding a
            # media BLOB so validated segments remain independently retrievable.
            # publish still checks finalizing: a concurrent terminal result or
            # deletion must never be overwritten by this fallback.
            video = None
            status = "partial" if frames else "failed"
            final_reason = "storage_failure"
            await self._transaction(publish)

    async def _staging_lock(self) -> int:
        import fcntl

        fd = os.open(
            self.path.with_suffix(".recording-lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            async with asyncio.timeout(130):
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return fd
                    except BlockingIOError:
                        await asyncio.sleep(0.01)
        except BaseException:
            os.close(fd)
            raise

    async def _combine(
        self,
        frames: list[tuple[str, bytes]],
        policy: BrowserRecordingPolicy,
        *,
        recording_id: str | None = None,
    ) -> bytes:
        lock_fd = await self._staging_lock()
        try:
            if recording_id is not None:

                def admitted(db: sqlite3.Connection) -> bool:
                    row = db.execute(
                        "SELECT status,expires FROM recordings WHERE id=?", (recording_id,)
                    ).fetchone()
                    return (
                        row is not None
                        and row[0] == "finalizing"
                        and row[1] > int(time.time() * 1000)
                    )

                if not await self._transaction(admitted):
                    raise BrowserRecordingUnavailable()
            with tempfile.TemporaryFile(dir="/tmp") as staging:
                entries = []
                offset = 0
                for raw_receipt, media in frames:
                    receipt = BrowserRecordingSegment.model_validate_json(raw_receipt)
                    if (
                        len(media) != receipt.size_bytes
                        or hashlib.sha256(media).hexdigest() != receipt.sha256
                    ):
                        raise BrowserRecordingUnavailable()
                    staging.write(media)
                    entries.append(
                        f"file 'subfile,,start,{offset},end,{offset + len(media)},,:/dev/fd/{staging.fileno()}'\nduration {receipt.duration_ms / 1000}\n"
                    )
                    offset += len(media)
                    if offset > policy.max_bytes:
                        raise BrowserRecordingUnavailable()
                staging.flush()
                maximum = min(policy.max_bytes, offset + 65536)
                with tempfile.TemporaryFile(dir="/tmp") as finalized:
                    await _bounded_process(
                        [
                            self.ffmpeg,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-nostdin",
                            "-y",
                            "-protocol_whitelist",
                            "pipe,file,subfile",
                            "-f",
                            "concat",
                            "-safe",
                            "0",
                            "-i",
                            "pipe:0",
                            "-map",
                            "0:v:0",
                            "-c",
                            "copy",
                            "-fs",
                            str(maximum),
                            "-f",
                            "webm",
                            f"/dev/fd/{finalized.fileno()}",
                        ],
                        "".join(entries).encode("ascii"),
                        4096,
                        pass_fds=(staging.fileno(), finalized.fileno()),
                        timeout=60,
                    )
                    finalized.seek(0)
                    video = finalized.read(maximum + 1)
                    if not video or len(video) > maximum:
                        raise BrowserRecordingUnavailable("limit_exhausted")
            # Full decoding proves frame count as well as container validity.
            decoded = await _bounded_process(
                [
                    self.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-xerror",
                    "-i",
                    "pipe:0",
                    "-map",
                    "0:v:0",
                    "-vf",
                    "scale=1:1",
                    "-pix_fmt",
                    "gray",
                    "-fps_mode",
                    "passthrough",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ],
                video,
                len(frames),
                timeout=60,
            )
            if len(decoded) != len(frames):
                raise BrowserRecordingUnavailable()
            return video
        finally:
            os.close(lock_fd)

    async def video(self, recording_id: str) -> bytes:
        def read(db: sqlite3.Connection) -> bytes:
            row = db.execute(
                "SELECT v.sha256,v.media FROM videos v JOIN recordings r ON r.id=v.recording WHERE r.id=? AND r.expires>? AND r.status IN ('complete','partial')",
                (recording_id, int(time.time() * 1000)),
            ).fetchone()
            if row is None or hashlib.sha256(row[1]).hexdigest() != row[0]:
                raise BrowserRecordingUnavailable()
            return row[1]

        return await self._transaction(read)

    async def manifest(self, recording_id: str) -> BrowserRecordingManifest:
        def read(db: sqlite3.Connection) -> BrowserRecordingManifest:
            row = db.execute(
                "SELECT scope,identity,status,reason,started,expires,elapsed_end FROM recordings WHERE id=? AND expires>? AND status != 'deleted'",
                (recording_id, int(time.time() * 1000)),
            ).fetchone()
            if row is None:
                raise BrowserRecordingUnavailable()
            identity = json.loads(row[1])
            segments = tuple(
                BrowserRecordingSegment.model_validate_json(item[0])
                for item in db.execute(
                    "SELECT receipt FROM segments WHERE recording=? ORDER BY sequence",
                    (recording_id,),
                )
            )
            gaps = []
            previous = 0
            for segment in segments:
                if segment.elapsed_ms > previous + 50:
                    gaps.append(BrowserRecordingGap(start_ms=previous, end_ms=segment.elapsed_ms))
                previous = segment.elapsed_ms + segment.duration_ms
            if row[6] > previous + 50:
                gaps.append(
                    BrowserRecordingGap(
                        start_ms=previous,
                        end_ms=row[6],
                        reason="worker_lost" if row[3] == "worker_lost" else "capture_gap",
                    )
                )
            artifact = db.execute(
                "SELECT sha256,length(media) FROM videos WHERE recording=?", (recording_id,)
            ).fetchone()
            return BrowserRecordingManifest(
                video_sha256=None if artifact is None else artifact[0],
                video_size_bytes=None if artifact is None else artifact[1],
                recording_id=recording_id,
                scope=row[0],
                identity=identity,
                gaps=tuple(gaps),
                browser_id=identity["browser_id"],
                worker_instance=identity["worker_instance"],
                status="recording" if row[2] == "finalizing" else row[2],
                reason=row[3],
                started_at_ms=row[4],
                expires_at_ms=row[5],
                segments=segments,
            )

        return await self._transaction(read)

    async def media(self, recording_id: str, sequence: int) -> bytes:
        def read(db: sqlite3.Connection) -> bytes:
            row = db.execute(
                "SELECT s.media,s.receipt FROM segments s JOIN recordings r ON r.id=s.recording WHERE r.id=? AND s.sequence=? AND r.expires>? AND r.status IN ('complete','partial')",
                (recording_id, sequence, int(time.time() * 1000)),
            ).fetchone()
            if row is None:
                raise BrowserRecordingUnavailable()
            receipt = BrowserRecordingSegment.model_validate_json(row[1])
            if (
                hashlib.sha256(row[0]).hexdigest() != receipt.sha256
                or len(row[0]) != receipt.size_bytes
            ):
                raise BrowserRecordingUnavailable()
            return row[0]

        return await self._transaction(read)

    async def recover_abandoned(self, *, inactive_seconds: int = 30) -> None:
        """Seal stale capture as partial/unavailable, never infer completion or replay."""
        if type(inactive_seconds) is not int or inactive_seconds < 30:
            raise ValueError("Recording recovery requires at least 30 seconds of inactivity.")

        def scan(db: sqlite3.Connection) -> list[tuple[str, int]]:
            return [
                (row[0], row[1])
                for row in db.execute(
                    "SELECT r.id,json_extract(g.policy,'$.max_duration_seconds')*1000 FROM recordings r JOIN grants g ON g.scope=r.scope WHERE r.status IN ('recording','finalizing') AND r.last_seen<? AND r.started+json_extract(g.policy,'$.max_duration_seconds')*1000<?",
                    (int(time.time() * 1000) - inactive_seconds * 1000, int(time.time() * 1000)),
                )
            ]

        for recording_id, elapsed_ms in await self._transaction(scan):
            await self.finish(recording_id, reason="worker_lost", elapsed_ms=elapsed_ms)

    async def delete(self, recording_id: str) -> None:
        def delete(db: sqlite3.Connection) -> None:
            db.execute("DELETE FROM videos WHERE recording=?", (recording_id,))
            db.execute("DELETE FROM segments WHERE recording=?", (recording_id,))
            db.execute("UPDATE recordings SET status='deleted' WHERE id=?", (recording_id,))

        await self._transaction(delete)
        os.close(await self._staging_lock())

    async def purge_expired(self) -> None:
        def purge(db: sqlite3.Connection) -> None:
            now = int(time.time() * 1000)
            db.execute(
                "DELETE FROM videos WHERE recording IN (SELECT id FROM recordings WHERE expires<=?)",
                (now,),
            )
            db.execute(
                "DELETE FROM segments WHERE recording IN (SELECT id FROM recordings WHERE expires<=?)",
                (now,),
            )
            db.execute("DELETE FROM recordings WHERE expires<=?", (now,))
            db.execute("DELETE FROM grants WHERE expires<=?", (now,))

        await self._transaction(purge)
        os.close(await self._staging_lock())
