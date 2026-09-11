"""Application-owned acceptance of a sealed result; never delivery authority."""

import json
from dataclasses import dataclass
from hashlib import sha256

from cayu import (
    CODING_PRODUCT_EVIDENCE_KIND,
    ArtifactStore,
    CodingArtifactReference,
    CodingProductArtifactRepository,
    CodingProductRequest,
    CompletionResultReference,
    CompletionVerdict,
    DockerCodingToolchainProfile,
    PublicAuthorityAliasCodec,
    WorkEvidenceReference,
    Workspace,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    coding_product_completion_decision,
    copy_artifact_read_result,
)
from cayu.workspaces.revisions import observe_deterministic_workspace
from tests.qualification.repository_maintenance_case import (
    ALLOWED_CHANGE_PATHS,
    SEED_BASE_REVISION,
    SEED_FILES,
    BehavioralOutcome,
    corpus_fingerprint,
)
from tests.qualification.repository_maintenance_probe import PROBE_CHECK_NAME, evaluate_probe_stdout
from tests.qualification.repository_maintenance_toolchain import maintenance_checks


class MaintenanceAcceptanceRejected(ValueError):
    def __init__(self) -> None:
        super().__init__("Repository-maintenance acceptance evidence is incomplete or conflicting.")


@dataclass(frozen=True)
class MaintenanceAcceptance:
    result_reference_id: str
    result_digest: str
    request_fingerprint: str
    final_revision: str
    corpus_fingerprint: str
    probe_output_digest: str


async def _read_bound_artifact(store: ArtifactStore, reference: CodingArtifactReference) -> bytes:
    maximum = 512 * 1024
    read = copy_artifact_read_result(
        await store.read_bytes(reference.artifact_id, max_bytes=maximum),
        expected_artifact_id=reference.artifact_id,
        max_content_bytes=maximum,
    )
    if (
        read.truncated
        or read.redaction_truncated
        or read.offset != 0
        or read.total_bytes != len(read.content)
        or len(read.content) != reference.size_bytes
        or "sha256:" + sha256(read.content).hexdigest() != reference.sha256
    ):
        raise MaintenanceAcceptanceRejected()
    return read.content


async def verify_maintenance_result(
    *,
    request: CodingProductRequest,
    reference: CompletionResultReference,
    repository: CodingProductArtifactRepository,
    toolchain: DockerCodingToolchainProfile,
    workspace: Workspace,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
) -> MaintenanceAcceptance:
    """Verify retained output and current exact source without executing code.

    Request, reference, toolchain and workspace must come from the application's
    tenant-qualified lookup, not directly from a product-route request body.
    A returned record does not authorize a later write or bypass fresh delivery
    approval/revision checks.
    """

    candidate = await repository.read_candidate(reference)
    decision = coding_product_completion_decision(
        request,
        candidate,
        evidence=WorkEvidenceReference(
            kind=CODING_PRODUCT_EVIDENCE_KIND,
            reference_id=reference.reference_id,
            version="v1",
            digest=reference.digest,
            available=True,
        ),
        public_authority_alias_codec=public_authority_alias_codec,
    )
    if (
        decision.verdict is not CompletionVerdict.ACCEPTED
        or candidate.final_revision is None
        or request.source.git_baseline.head_revision != SEED_BASE_REVISION
        or request.runtime.toolchain_profile_fingerprint != toolchain.fingerprint
        or request.runtime.dependency_identity != toolchain.dependency_identity
        or request.runtime.image_fingerprint != toolchain.image_identity.fingerprint
    ):
        raise MaintenanceAcceptanceRejected()
    checks = {check.check: check for check in candidate.checks}
    declared = maintenance_checks()
    if set(checks) != {check.name for check in declared} or any(
        checks[check.name].profile_fingerprint != check.profile_fingerprint for check in declared
    ):
        raise MaintenanceAcceptanceRejected()
    initial = WorkspaceRevisionObservation.model_validate_json(
        await _read_bound_artifact(repository.store, candidate.initial_source.artifact)
    )
    final = WorkspaceRevisionObservation.model_validate_json(
        await _read_bound_artifact(repository.store, candidate.final_source.artifact)
    )
    for observation, revision in (
        (initial, candidate.initial_revision),
        (final, candidate.final_revision),
    ):
        if (
            observation.status is not WorkspaceRevisionObservationStatus.SUPPORTED
            or observation.path_scope != "complete"
            or observation.revision != revision
            or observation.identity.workspace_id != request.source.workspace_id
            or {item.path for item in observation.paths} != set(SEED_FILES)
            or any(item.kind != "file" or item.content_sha256 is None for item in observation.paths)
        ):
            raise MaintenanceAcceptanceRejected()
    initial_paths = {item.path: item for item in initial.paths}
    final_paths = {item.path: item for item in final.paths}
    for path, content in SEED_FILES.items():
        if initial_paths[path].content_sha256 != sha256(content.encode()).hexdigest():
            raise MaintenanceAcceptanceRejected()
        if path not in ALLOWED_CHANGE_PATHS and final_paths[path] != initial_paths[path]:
            raise MaintenanceAcceptanceRejected()
    probe = checks[PROBE_CHECK_NAME]
    if probe.output_artifact is None or probe.output_artifact_status != "stored":
        raise MaintenanceAcceptanceRejected()
    raw = await _read_bound_artifact(repository.store, probe.output_artifact)
    try:
        record = json.loads(raw)
        valid = (
            type(record) is dict
            and set(record)
            == {
                "schema_version",
                "check",
                "check_profile_fingerprint",
                "stdout",
                "stderr",
                "stdout_encoding",
                "stderr_encoding",
                "stdout_runner_truncated",
                "stderr_runner_truncated",
            }
            and record["schema_version"] == "5"
            and record["check"] == PROBE_CHECK_NAME
            and record["check_profile_fingerprint"] == probe.profile_fingerprint
            and record["stdout_encoding"] == "text"
            and record["stderr_encoding"] == "text"
            and record["stdout_runner_truncated"] is False
            and record["stderr_runner_truncated"] is False
            and record["stderr"] == ""
            and type(record["stdout"]) is str
            and evaluate_probe_stdout(record["stdout"].encode()) is BehavioralOutcome.PASSED
        )
    except (ValueError, UnicodeError, RecursionError):
        valid = False
    if not valid:
        raise MaintenanceAcceptanceRejected()
    for phase in range(2):
        observed = await observe_deterministic_workspace(
            workspace,
            observer=final.identity.observer,
            limits=request.source.observation_limits,
        )
        if observed != final:
            raise MaintenanceAcceptanceRejected()
        if phase == 0:
            for path, entry in final_paths.items():
                read = await workspace.read_bytes(path, max_bytes=16 * 1024 + 1)
                if (
                    read.truncated
                    or read.total_bytes > 16 * 1024
                    or sha256(read.content).hexdigest() != entry.content_sha256
                ):
                    raise MaintenanceAcceptanceRejected()
    return MaintenanceAcceptance(
        reference.reference_id,
        reference.digest,
        request.fingerprint,
        candidate.final_revision,
        corpus_fingerprint(),
        probe.output_artifact.sha256,
    )
