"""One immutable saved HTTP publication; retry never imports or executes a target."""

from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import Field, StrictBool, StrictStr, model_validator

from cayu._validation import durable_json_object_from_pairs, require_durable_clean_nonblank
from cayu.cli._eval_report_metadata import add_metadata_arguments, report_origin
from cayu.evals._report_projection import direct_report_projection, report_plan_app
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.reported import (
    EVAL_REPORT_MAX_BYTES,
    EvalCorpusReportEvidence,
    EvalDirectReportEvidence,
    EvalReport,
    EvalReportCorpusAssociation,
    EvalReportedRun,
    EvalReportOrigin,
    EvalReportReceipt,
    _ReportModel,
    report_canonical_bytes,
    report_fingerprint,
)
from cayu.vaults.redaction import SecretRedactor

TOKEN_VARIABLE = "CAYU_EVAL_REPORT_TOKEN"
_SAVED_MAX_BYTES = EVAL_REPORT_MAX_BYTES + (8 << 10)


class SavedEvalReport(_ReportModel):
    schema_version: int = Field(default=1, strict=True, ge=1, le=1)
    destination: StrictStr = Field(max_length=2048)
    allow_loopback_http: StrictBool = False
    report: EvalReport

    @model_validator(mode="after")
    def validate_destination(self):
        if report_destination(self.destination, self.allow_loopback_http) != self.destination:
            raise ValueError("Saved report destination must be canonical.")
        return self


def report_destination(value: str, allow_loopback: bool) -> str:
    if (
        type(value) is not str
        or not value.isascii()
        or any(ord(char) < 33 or ord(char) > 126 for char in value)
    ):
        raise ValueError("Reporting destination must be a clean ASCII URL.")
    parsed = urlsplit(value)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Reporting destination cannot contain credentials, query or fragment.")
    if not parsed.hostname or len(value) > 2048 or parsed.port == 0:
        raise ValueError("Reporting destination requires an HTTP API base URL.")
    loopback = parsed.hostname == "localhost"
    with contextlib.suppress(ValueError):
        loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
    if parsed.scheme != "https" and not (parsed.scheme == "http" and allow_loopback and loopback):
        raise ValueError("Reporting requires HTTPS or explicit loopback HTTP.")
    return value.rstrip("/")


def report_token(path: str | None) -> str:
    if path is None:
        value = os.environ.get(TOKEN_VARIABLE, "")
    else:
        with Path(path).open("rb") as stream:
            raw = stream.read(4097)
        value = raw.decode("utf-8").strip()
    if (
        not value
        or len(value.encode("utf-8")) > 4089
        or any(not 33 <= ord(ch) <= 126 for ch in value)
    ):
        raise ValueError("A bounded reporting credential is required through environment or file.")
    return value


@dataclass(frozen=True, repr=False)
class ReportRequest:
    destination: str
    allow_loopback_http: bool
    target_key: str
    program_key: str | None
    revision: str | None
    origin: EvalReportOrigin
    file: Path
    token: str
    associate_corpus: bool
    credential_file: Path | None
    report_id: str


def add_report_arguments(run, inner):
    run.add_argument(
        "--report-to", metavar="API_URL", help="Publish completed evidence to a dashboard HTTP API."
    )
    run.add_argument(
        "--report-target",
        metavar="TARGET_KEY",
        help="Authorized dashboard target key; never inferred from a Python name.",
    )
    run.add_argument(
        "--report-file",
        metavar="NEW_FILE",
        help="Save a new immutable sanitized report envelope before publication.",
    )
    run.add_argument("--report-program", metavar="PROGRAM_KEY")
    run.add_argument("--report-suite-revision", metavar="SHA256_REVISION")
    run.add_argument(
        "--report-associate-corpus",
        action="store_true",
        help="Require exact association to an already stored corpus contract.",
    )
    run.add_argument("--report-allow-loopback-http", action="store_true")
    run.add_argument(
        "--report-token-file",
        metavar="FILE",
        help=f"Credential file; otherwise use {TOKEN_VARIABLE}.",
    )
    add_metadata_arguments(run)
    publish = inner.add_parser(
        "publish",
        help="Publish a saved report exactly, without rerunning evaluation.",
        description=(
            "Publish immutable saved evidence without rerunning evaluation. "
            "Pass --report-to with the independently trusted dashboard API URL, "
            "then inspect the returned run in the dashboard Evals history."
        ),
    )
    publish.add_argument("input", metavar="REPORT_FILE")
    publish.add_argument("--report-token-file", metavar="FILE")
    publish.add_argument(
        "--report-to",
        metavar="API_URL",
        help="Trusted API destination; must match the saved envelope before credential access.",
    )
    publish.add_argument(
        "--report-allow-loopback-http",
        action="store_true",
        help="Independently authorize a saved loopback HTTP destination.",
    )


def prepare_report_request(args) -> ReportRequest | None:
    try:
        return _prepare_report_request(args)
    except Exception:
        raise ValueError(
            "Invalid Evals reporting configuration; check destination, credential and metadata options."
        ) from None


def _prepare_report_request(args) -> ReportRequest | None:
    destination = getattr(args, "report_to", None)
    options = {
        key: value
        for key, value in vars(args).items()
        if key.startswith("report_") and value is not None and value is not False
    }
    if destination is None:
        if options:
            raise ValueError("Reporting options require --report-to.")
        return None
    if getattr(args, "package", None) is not None:
        raise ValueError("Dashboard reporting does not support benchmark campaigns.")
    target = getattr(args, "report_target", None)
    if type(target) is not str or not 1 <= len(target) <= 128:
        raise ValueError("Dashboard reporting requires --report-target.")
    require_durable_clean_nonblank(target, "report target")
    program = args.report_program
    if program is not None:
        if type(program) is not str or len(program) > 256:
            raise ValueError("Invalid reporting program identity.")
        require_durable_clean_nonblank(program, "report program")
    revision = args.report_suite_revision
    if revision is not None and re.fullmatch(r"sha256:[0-9a-f]{64}", revision) is None:
        raise ValueError("Invalid declared suite revision.")
    allow = args.report_allow_loopback_http
    destination = report_destination(destination, allow)
    path = Path(args.report_file or f".cayu/evals/reports/{uuid4()}.json").resolve()
    if path.exists() or path.is_symlink():
        raise ValueError("Report file must be new; use eval publish for an exact retry.")
    credential_file = (
        None if args.report_token_file is None else Path(args.report_token_file).resolve()
    )
    token = report_token(None if credential_file is None else str(credential_file))
    if SecretRedactor(token).redact_json(destination) != destination:
        raise ValueError("Reporting destination contains the publication credential.")
    return ReportRequest(
        destination=destination,
        allow_loopback_http=allow,
        target_key=target,
        program_key=program,
        revision=revision,
        origin=report_origin(args),
        file=path,
        token=token,
        associate_corpus=args.report_associate_corpus,
        credential_file=credential_file,
        report_id=str(uuid4()),
    )


def report_request_context(request: ReportRequest, program_key: str):
    context = {
        "destination": request.destination,
        "report_id": request.report_id,
        "report_file": str(request.file),
        "target_key": request.target_key,
        "program_key": request.program_key or program_key,
        "declared_suite_revision": request.revision,
        "origin": request.origin.model_dump(mode="json"),
    }
    if SecretRedactor(request.token).redact_json(context) != context:
        raise ValueError("Reporting context contains the publication credential.")
    return context


def _save_exact(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        if os.name == "nt":
            from cayu.cli._guarded_tree_publication import _sync_windows_path

            _sync_windows_path(path.parent, directory=True)
        else:
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


async def _post(saved: SavedEvalReport, token: str):
    payload = report_canonical_bytes(saved.report)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False, trust_env=False) as client:
        async with client.stream(
            "POST",
            saved.destination + "/evals/reports",
            content=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        ) as response:
            if response.status_code != 200:
                raise ValueError("Dashboard rejected the eval report.")
            retained = bytearray()
            async for chunk in response.aiter_bytes():
                if len(retained) + len(chunk) > 8192:
                    raise ValueError("Invalid dashboard report receipt.")
                retained.extend(chunk)
            receipt = EvalReportReceipt.model_validate_json(retained)
            document = receipt.model_dump(mode="json")
            if SecretRedactor(token).redact_json(document) != document:
                raise ValueError("Invalid dashboard report receipt.")
        if (
            receipt.report_id != saved.report.report_id
            or receipt.target_key != saved.report.target_key
            or receipt.fingerprint != report_fingerprint(saved.report)
        ):
            raise ValueError("Dashboard receipt disagrees with the exact submitted report.")
    return receipt


async def finish_report(
    request, result, *, program_key, plan=None, projected: EvalReportedRun | None = None
) -> bool:
    if request is None:
        return True
    try:
        app = None if plan is None else report_plan_app(plan)
        credential_redactor = SecretRedactor(request.token)

        def redact(value):
            return credential_redactor.redact_json(value if app is None else app.redact_json(value))

        association = None
        if type(result) is CorpusExecutionResult:
            evidence = EvalCorpusReportEvidence(result=result)
            if request.associate_corpus:
                association = EvalReportCorpusAssociation(
                    corpus_revision=result.run.corpus_revision, suite_id=result.run.suite_id
                )
        else:
            if request.associate_corpus:
                raise ValueError("Native corpus association requires corpus execution.")
            if projected is None:
                if app is None:
                    raise ValueError("Direct reporting requires the workload redaction owner.")
                projected = direct_report_projection(result, redact_json=redact)
            evidence = EvalDirectReportEvidence(run=projected)
        report = EvalReport(
            report_id=request.report_id,
            target_key=request.target_key,
            program_key=request.program_key or program_key,
            declared_suite_revision=request.revision,
            origin=request.origin,
            evidence=evidence,
            association=association,
        )
        report = EvalReport.model_validate_json(
            json.dumps(redact(report.model_dump(mode="json")), ensure_ascii=False)
        )
        saved = SavedEvalReport(
            destination=request.destination,
            allow_loopback_http=request.allow_loopback_http,
            report=report,
        )
        safe_document = saved.model_dump(mode="json")
        if redact(safe_document) != safe_document:
            raise ValueError("Saved publication context contains a configured credential.")
        data = saved.model_dump_json().encode("utf-8")
        if (
            len(data) > _SAVED_MAX_BYTES
            or len(report_canonical_bytes(report)) > EVAL_REPORT_MAX_BYTES
        ):
            raise ValueError("Eval report exceeds the byte limit.")
        _save_exact(request.file, data)
        await _post(saved, request.token)
        print(
            f"Eval report published; saved envelope: {redact(str(request.file))}", file=sys.stderr
        )
        return True
    except Exception:
        print(
            "Eval reporting failed; the evaluation result is unchanged. If saved, retain the report file and retry with cayu eval publish.",
            file=sys.stderr,
        )
        return False


async def publish_saved(args) -> int:
    try:
        # An artifact is unsigned caller input, never authority to receive a credential.
        destination = report_destination(args.report_to, args.report_allow_loopback_http)
        with Path(args.input).open("rb") as stream:
            raw = stream.read(_SAVED_MAX_BYTES + 1)
        if len(raw) > _SAVED_MAX_BYTES:
            raise ValueError("Saved report exceeds the byte limit.")
        parsed = json.loads(
            raw,
            object_pairs_hook=lambda pairs: durable_json_object_from_pairs(pairs, "saved report"),
        )
        saved = SavedEvalReport.model_validate_json(json.dumps(parsed, ensure_ascii=False))
        if saved.destination != destination:
            raise ValueError("Saved report destination disagrees with --report-to.")
        token = report_token(args.report_token_file)
        if SecretRedactor(token).redact_json(saved.model_dump(mode="json")) != saved.model_dump(
            mode="json"
        ):
            raise ValueError("Saved report contains a credential.")
        receipt = await _post(saved, token)
        print(receipt.model_dump_json())
        return 0
    except Exception:
        print(
            "Eval reporting failed; the saved report was not changed and evaluation was not rerun.",
            file=sys.stderr,
        )
        return 2
