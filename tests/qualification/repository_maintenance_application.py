"""Generate the maintained consumer using the existing installed CLI scaffold.

This is qualification setup, not application runtime code. The emitted consumer
has no import dependency on this module or the Runtime test suite.
"""

import ast
from pathlib import Path

from cayu.cli.scaffold import project_files
from tests.qualification.repository_maintenance_compose import compose_assets
from tests.qualification.repository_maintenance_probe import probe_program

_HERE = Path(__file__).parent
_PROBE_INPUT = "environments/range_probe.py"


def _replace_once(source: str, original: str, replacement: str) -> str:
    if source.count(original) != 1:
        raise ValueError("Generated application customization no longer matches its owner.")
    return source.replace(original, replacement, 1)


def _specialize_python(
    source: str, *, functions: dict[str, str] | None = None, remove: tuple[str, ...] = ()
) -> str:
    """Replace declared top-level functions, rejecting generator drift."""

    module = ast.parse(source)
    replacements = dict(functions or {})
    body = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in remove:
                continue
            if node.name in replacements:
                body.extend(ast.parse(replacements.pop(node.name)).body)
                continue
        body.append(node)
    if replacements:
        raise ValueError("Generated application function owner is missing.")
    module.body = body
    # Removing the built-in profile and fixture-only Git materializer also removes
    # their imports. Generated code retains annotations as actual AST expressions.
    used = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            node.names = [
                alias for alias in node.names if (alias.asname or alias.name.split(".")[0]) in used
            ]
    module.body = [
        node for node in body if not isinstance(node, (ast.Import, ast.ImportFrom)) or node.names
    ]
    return ast.unparse(module) + "\n"


def maintenance_project_files(*, database: str = "postgres") -> dict[str, str]:
    """Produce the exact custom consumer; do not build images or create stores."""

    files = project_files(
        "maintenance-app",
        preset="coding",
        database=database,
        execution="docker",
        coding_toolchain="python",
        with_capabilities=("github-delivery",),
    )
    # This consumer's host-only Git integration accepts native secret refs;
    # the coding composition and its environment never receive these objects.
    remote_git = files["integrations/remote_git.py"]
    remote_git = _replace_once(
        remote_git,
        "    RemoteGitRemoteConfig,\n",
        "    RemoteGitHttpCredentials,\n    RemoteGitRemoteConfig,\n",
    )
    remote_git = _replace_once(
        remote_git,
        "    default_branch_ref: str,\n",
        "    default_branch_ref: str,\n"
        "    credentials: RemoteGitHttpCredentials | None = None,\n"
        '    egress_profile_id: str = "application-local",\n',
    )
    remote_git = _replace_once(
        remote_git,
        '    """Construct the separate no-credential default delivery integration.\n\n'
        "    Extend this application-owned seam with RemoteGitHttpCredentials and a\n"
        "    SecretResolver when the admitted remote requires authentication. Never pass\n",
        '    """Construct the host-only delivery integration without resolving secrets.\n\n'
        "    Optional native credentials are resolved only by broker remote operations.\n"
        "    The caller must enforce the configured egress boundary. Never pass\n",
    )
    remote_git = _replace_once(
        remote_git,
        "    coding_repository = CodingProductArtifactRepository(artifact_store)\n",
        "    if credentials is not None and type(credentials) is not RemoteGitHttpCredentials:\n"
        "        raise ValueError(\n"
        '            "Maintenance Git credentials require native secret references."\n'
        "        )\n"
        "    coding_repository = CodingProductArtifactRepository(artifact_store)\n",
    )
    remote_git = _replace_once(
        remote_git,
        "        default_branch_ref=default_branch_ref,\n",
        "        default_branch_ref=default_branch_ref,\n"
        "        credential_profile_id=(\n"
        '            "none" if credentials is None else credentials.credential_profile_id\n'
        "        ),\n"
        "        credentials=credentials,\n"
        "        egress_profile_id=egress_profile_id,\n",
    )
    files["integrations/remote_git.py"] = remote_git
    files["integrations/maintenance_git_host.py"] = (
        (_HERE / "repository_maintenance_git_host.py")
        .read_text()
        .replace("  # ty: ignore[unresolved-import]", "")
    )
    files["integrations/maintenance_github_host.py"] = (
        (_HERE / "repository_maintenance_github_host.py")
        .read_text()
        .replace("  # ty: ignore[unresolved-import]", "")
    )
    # The application-owned API/operator host is deployed without dev groups.
    # Keep the coding preset; this does not adopt the service execution owner.
    runtime_extra = "[postgres]" if database == "postgres" else ""
    deployed_extra = "[postgres,server]" if database == "postgres" else "[server]"
    files["pyproject.toml"] = _replace_once(
        files["pyproject.toml"], f'"cayu{runtime_extra}==', f'"cayu{deployed_extra}=='
    )
    files["pyproject.toml"] = _replace_once(
        files["pyproject.toml"],
        'factory = "app:build_app"',
        'factory = "app:build_maintenance_app"',
    )
    files["configuration/maintenance.py"] = (
        (_HERE / "repository_maintenance_configuration.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_budget", "domain.maintenance_budget")
        .replace("tests.qualification.repository_maintenance_auth", "integrations.maintenance_auth")
    )
    files["operations/maintenance_roles.py"] = (
        (_HERE / "repository_maintenance_roles.py")
        .read_text()
        .replace(
            "tests.qualification.repository_maintenance_deployment",
            "operations.maintenance_deployment",
        )
        .replace("  # ty: ignore[unresolved-import]", "")
        .replace(
            "tests.qualification.repository_maintenance_lifetime", "operations.maintenance_lifetime"
        )
    )
    files["pyproject.toml"] += (
        '\n[tool.cayu.workers]\ncoding = "operations.maintenance_roles:run_coding"\n'
        'git_preparation = "operations.maintenance_roles:run_git_preparation"\n'
        'git_delivery = "operations.maintenance_roles:run_git_delivery"\n'
        'github_delivery = "operations.maintenance_roles:run_github_delivery"\n'
    )
    files["app.py"] += '''

def build_maintenance_app() -> CayuApp:
    """Construct the configured production graph without starting active work."""
    from configuration.maintenance import configured_maintenance_budget
    from operations.maintenance_deployment import build_maintenance_deployment

    return build_maintenance_deployment(
        budget_policy=configured_maintenance_budget(),
    ).application.app
'''
    files["workflows/maintenance_coding.py"] = (
        (_HERE / "repository_maintenance_workflow.py")
        .read_text()
        .replace(
            "from tests.qualification.repository_maintenance_acceptance import",
            "from domain.maintenance_acceptance import",
        )
        .replace("tests.qualification.repository_maintenance_request", "domain.maintenance_request")
        .replace("  # ty: ignore[unresolved-import]", "")
    )
    evaluation = (_HERE / "repository_maintenance_eval.py").read_text()
    for name, package in (
        ("deployment", "operations"),
        ("identity", "domain"),
        ("lifetime", "operations"),
        ("request", "domain"),
        ("results", "operations"),
    ):
        evaluation = evaluation.replace(
            f"tests.qualification.repository_maintenance_{name}", f"{package}.maintenance_{name}"
        )
    files["evals/maintenance.py"] = evaluation.replace("  # ty: ignore[unresolved-import]", "")
    files["evals/maintenance.md"] = (_HERE / "repository_maintenance_evals.md").read_text()
    files["domain/maintenance_request.py"] = (
        _HERE / "repository_maintenance_request.py"
    ).read_text()
    files["domain/maintenance_budget.py"] = (_HERE / "repository_maintenance_budget.py").read_text()
    files["operations/maintenance-incidents.md"] = (
        _HERE / "repository_maintenance_incidents.md"
    ).read_text()
    files["operations/maintenance_schema.py"] = (
        (_HERE / "repository_maintenance_schema.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_runs", "operations.maintenance_runs")
    )
    files["operations/maintenance_deployment.py"] = (
        (_HERE / "repository_maintenance_deployment.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_budget", "domain.maintenance_budget")
        .replace("tests.qualification.repository_maintenance_runs", "operations.maintenance_runs")
        .replace("  # ty: ignore[unresolved-import]", "")
    )
    files["integrations/maintenance_auth.py"] = (
        _HERE / "repository_maintenance_auth.py"
    ).read_text()
    http = (_HERE / "repository_maintenance_http.py").read_text()
    for original, emitted in (
        ("auth", "integrations.maintenance_auth"),
        ("budget", "domain.maintenance_budget"),
        ("cost", "operations.maintenance_cost"),
        ("final", "operations.maintenance_final"),
        ("delivery_view", "operations.maintenance_delivery_view"),
        ("git_intake", "operations.maintenance_git_intake"),
        ("github_intake", "operations.maintenance_github_intake"),
        ("operator", "operations.maintenance_operator"),
        ("identity", "domain.maintenance_identity"),
        ("request", "domain.maintenance_request"),
        ("intake", "operations.maintenance_intake"),
        ("runs", "operations.maintenance_runs"),
        ("results", "operations.maintenance_results"),
    ):
        http = http.replace(f"tests.qualification.repository_maintenance_{original}", emitted)
    files["operations/maintenance_http.py"] = http.replace("  # ty: ignore[unresolved-import]", "")
    files["operations/maintenance_asgi.py"] = (_HERE / "repository_maintenance_asgi.py").read_text()
    files["operations/maintenance_lifetime.py"] = (
        _HERE / "repository_maintenance_lifetime.py"
    ).read_text()
    api = (_HERE / "repository_maintenance_api.py").read_text()
    for name in ("asgi", "deployment", "http", "lifetime"):
        api = api.replace(
            f"tests.qualification.repository_maintenance_{name}", f"operations.maintenance_{name}"
        )
    files["operations/maintenance_api.py"] = api.replace("  # ty: ignore[unresolved-import]", "")
    files[
        "operations/maintenance_requests.py"
    ] = '''"""Host-owned capture of the accepted coding configuration; read-only."""

from cayu import LocalWorkspace
from domain.coding_product import CodingProductTask
from domain.maintenance_case import SEED_BASE_REVISION, corpus_fingerprint
from domain.maintenance_request import (
    MaintenanceAcceptedRequest, bounded_text, default_maintenance_settlement,
    invalid_request,
)
from environments.maintenance_probe import probe_fingerprint


async def capture_accepted_request(application, task):
    if type(task) is not CodingProductTask:
        raise invalid_request()
    instruction = bounded_text(task.instruction, bound=4096)
    origin = bounded_text(task.source_origin_id)
    destination = bounded_text(task.source_destination_id)
    if task.settlement is not None or task.review_settlement is not None:
        raise invalid_request()
    workspace = application.source_workspace
    if type(workspace) is not LocalWorkspace or workspace.root != application.project_root:
        raise invalid_request()
    return MaintenanceAcceptedRequest(
        schema_version="maintenance.accepted.v1",
        instruction=instruction,
        repository_root=str(workspace.root),
        source_workspace_id=workspace.id,
        source_origin_id=origin,
        source_destination_id=destination,
        artifact_store_id=application.artifact_store.id,
        toolchain_profile_fingerprint=application.toolchain_profile.fingerprint,
        execution_profile_fingerprint=await application.inspect_execution_profile(task),
        corpus_fingerprint=corpus_fingerprint(),
        probe_fingerprint=probe_fingerprint(),
        base_revision=SEED_BASE_REVISION,
        settlement_json=default_maintenance_settlement().model_dump_json(),
    )
'''
    files[
        "operations/maintenance_worker.py"
    ] = '''"""Coding handler under the existing Runtime task worker's lease."""

from contextlib import aclosing

from cayu import CodingProductArtifactRepository, EventType, complete_managed_task
from domain.maintenance_acceptance import MaintenanceAcceptanceRejected
from domain.maintenance_request import decode_request
from operations.maintenance_intake import load_claimed_coding_identity
from operations.maintenance_results import coding_task_from_identity
from workflows.maintenance_coding import MaintenanceCodingWorkflow


async def handle_coding_task(application, reservations, claimed, worker_id):
    identity = await load_claimed_coding_identity(
        application.app, reservations, claimed, worker_id,
    )
    accepted = decode_request(identity.intent.request_json)
    task = coding_task_from_identity(identity)
    completion = None
    async with aclosing(
        MaintenanceCodingWorkflow(application, task, accepted=accepted).execute(
            identity.workflow_session_id, execution_deadline=identity.coding_deadline(),
        )
    ) as events:
        async for event in events:
            if event.type is EventType.WORKFLOW_COMPLETED:
                if completion is not None:
                    raise MaintenanceAcceptanceRejected()
                completion = event
    if (
        completion is None
        or completion.payload.get("verdict") != "verified"
        or completion.payload.get("product_run_id") != identity.product_run_id
    ):
        raise MaintenanceAcceptanceRejected()
    repository = CodingProductArtifactRepository(application.artifact_store)
    admitted = await repository.load_request(task.product_run_id, session_id=task.session_id)
    publication = await repository.load_publication(
        request_fingerprint=admitted.fingerprint, digest=completion.payload["result_digest"],
    )
    verified = await application.verify(task, publication)
    await complete_managed_task(
        application.app.task_store, claimed, worker_id,
        {"product_run_id": identity.product_run_id, "result_digest": verified.result_digest},
    )
'''
    files[
        "workflows/maintenance_delivery.py"
    ] = '''"""Verified host-side Git handoff; no automatic approval or merge."""

from cayu import (
    CodingProductPublication,
    GitHubDeliveryApproval,
    GitHubDeliveryPublication,
    GitHubPullRequestConnector,
    GitHubPullRequestDeliveryRequest,
    RemoteGitDeliveryApproval,
    RemoteGitDeliveryBroker,
    RemoteGitDeliveryPublication,
    RemoteGitDeliveryRequest,
    RemoteGitPreparedIntent,
)
from domain.coding_product import CodingProductTask
from workflows.coding_product import CodingProductApplication


async def prepare_verified_git_delivery(
    application: CodingProductApplication,
    task: CodingProductTask,
    publication: CodingProductPublication,
    broker: RemoteGitDeliveryBroker,
    request: RemoteGitDeliveryRequest,
) -> RemoteGitPreparedIntent:
    """Inputs come from trusted application lookup, not unauthenticated route data."""
    await application.verify(task, publication)
    return await broker.prepare(request, publication, source_workspace=application.source_workspace)


async def run_verified_git_delivery(
    application: CodingProductApplication,
    task: CodingProductTask,
    publication: CodingProductPublication,
    broker: RemoteGitDeliveryBroker,
    request: RemoteGitDeliveryRequest,
    *,
    approval: RemoteGitDeliveryApproval | None = None,
) -> RemoteGitDeliveryPublication:
    """Revalidate acceptance, then let the broker enforce exact approval and replay."""
    await application.verify(task, publication)
    return await broker.run(
        request, publication, source_workspace=application.source_workspace, approval=approval,
    )


async def run_verified_github_delivery(
    application: CodingProductApplication,
    task: CodingProductTask,
    publication: CodingProductPublication,
    remote: RemoteGitDeliveryPublication,
    connector: GitHubPullRequestConnector,
    request: GitHubPullRequestDeliveryRequest,
    *,
    approval: GitHubDeliveryApproval | None = None,
) -> GitHubDeliveryPublication:
    """Verify trusted application inputs; caller retains connector lifetime ownership."""
    await application.verify(task, publication)
    return await connector.run(request, publication, remote, approval=approval)
'''
    files["operations/maintenance_github.py"] = (
        (_HERE / "repository_maintenance_github.py")
        .read_text()
        .replace(
            "tests.qualification.repository_maintenance_lifetime", "operations.maintenance_lifetime"
        )
        .replace("  # ty: ignore[unresolved-import]", "")
    )
    for name in ("git_intake", "github_intake", "intake"):
        files["operations/maintenance_github.py"] = files[
            "operations/maintenance_github.py"
        ].replace(
            f"tests.qualification.repository_maintenance_{name}", f"operations.maintenance_{name}"
        )
    files["operations/maintenance_results.py"] = (
        (_HERE / "repository_maintenance_results.py")
        .read_text()
        .replace(
            "tests.qualification.repository_maintenance_identity", "domain.maintenance_identity"
        )
        .replace(
            "tests.qualification.repository_maintenance_intake", "operations.maintenance_intake"
        )
        .replace("tests.qualification.repository_maintenance_request", "domain.maintenance_request")
        .replace("  # ty: ignore[unresolved-import]", "")
    )
    git_intake = (_HERE / "repository_maintenance_git_intake.py").read_text()
    for name, package in (
        ("identity", "domain"),
        ("intake", "operations"),
        ("results", "operations"),
    ):
        git_intake = git_intake.replace(
            f"tests.qualification.repository_maintenance_{name}", f"{package}.maintenance_{name}"
        )
    files["operations/maintenance_git_intake.py"] = git_intake.replace(
        "  # ty: ignore[unresolved-import]", ""
    )
    git_worker = (_HERE / "repository_maintenance_git_worker.py").read_text()
    for name in ("git_intake", "intake", "lifetime", "results"):
        git_worker = git_worker.replace(
            f"tests.qualification.repository_maintenance_{name}", f"operations.maintenance_{name}"
        )
    files["operations/maintenance_git_worker.py"] = git_worker.replace(
        "  # ty: ignore[unresolved-import]", ""
    )
    github_intake = (_HERE / "repository_maintenance_github_intake.py").read_text()
    for name, package in (
        ("git_intake", "operations"),
        ("identity", "domain"),
        ("intake", "operations"),
    ):
        github_intake = github_intake.replace(
            f"tests.qualification.repository_maintenance_{name}", f"{package}.maintenance_{name}"
        )
    files["operations/maintenance_github_intake.py"] = github_intake.replace(
        "  # ty: ignore[unresolved-import]", ""
    )
    operator = (_HERE / "repository_maintenance_operator.py").read_text()
    for name, package in (
        ("git_intake", "operations"),
        ("github_intake", "operations"),
        ("identity", "domain"),
        ("intake", "operations"),
    ):
        operator = operator.replace(
            f"tests.qualification.repository_maintenance_{name}", f"{package}.maintenance_{name}"
        )
    files["operations/maintenance_operator.py"] = operator
    delivery_view = (_HERE / "repository_maintenance_delivery_view.py").read_text()
    for name, package in (
        ("git_intake", "operations"),
        ("github_intake", "operations"),
        ("identity", "domain"),
        ("intake", "operations"),
    ):
        delivery_view = delivery_view.replace(
            f"tests.qualification.repository_maintenance_{name}", f"{package}.maintenance_{name}"
        )
    files["operations/maintenance_delivery_view.py"] = delivery_view
    final = (_HERE / "repository_maintenance_final.py").read_text()
    for name, package in (
        ("cost", "operations"),
        ("github_intake", "operations"),
        ("identity", "domain"),
        ("intake", "operations"),
        ("results", "operations"),
    ):
        final = final.replace(
            f"tests.qualification.repository_maintenance_{name}", f"{package}.maintenance_{name}"
        )
    files["operations/maintenance_final.py"] = final.replace(
        "  # ty: ignore[unresolved-import]", ""
    )
    files["operations/maintenance_cost.py"] = (
        (_HERE / "repository_maintenance_cost.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_budget", "domain.maintenance_budget")
        .replace(
            "tests.qualification.repository_maintenance_identity", "domain.maintenance_identity"
        )
    )
    case = (_HERE / "repository_maintenance_case.py").read_text()
    materializers = [
        node
        for node in ast.parse(case).body
        if isinstance(node, ast.FunctionDef) and node.name == "materialize_seed_repository"
    ]
    if len(materializers) != 1:
        raise ValueError("Maintenance seed preparation owner is missing or ambiguous.")
    files["evals/maintenance_seed.py"] = (
        '"""Fixed new-directory seed preparation; no reset or reuse of existing sources."""\n'
        "import os\nimport shutil\nimport subprocess\nfrom pathlib import Path\n"
        "from domain.maintenance_case import CASE_ID, SEED_FILES\n\n"
        + ast.unparse(materializers[0])
        + "\n"
    )
    files["domain/maintenance_identity.py"] = (
        _HERE / "repository_maintenance_identity.py"
    ).read_text()
    files["operations/maintenance_runs.py"] = (
        (_HERE / "repository_maintenance_runs.py")
        .read_text()
        .replace(
            "tests.qualification.repository_maintenance_identity", "domain.maintenance_identity"
        )
    )
    files["operations/maintenance_intake.py"] = (
        (_HERE / "repository_maintenance_intake.py")
        .read_text()
        .replace(
            "tests.qualification.repository_maintenance_identity", "domain.maintenance_identity"
        )
        .replace("tests.qualification.repository_maintenance_runs", "operations.maintenance_runs")
    )
    files["domain/maintenance_case.py"] = _specialize_python(
        case, remove=("materialize_seed_repository",)
    )
    probe = (
        (_HERE / "repository_maintenance_probe.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_case", "domain.maintenance_case")
    )
    files["environments/maintenance_probe.py"] = probe
    files["environments/maintenance_toolchain.py"] = (
        (_HERE / "repository_maintenance_toolchain.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_case", "domain.maintenance_case")
        .replace(
            "tests.qualification.repository_maintenance_probe", "environments.maintenance_probe"
        )
    )
    files[_PROBE_INPUT] = probe_program().decode()
    files["policies/maintenance.py"] = (
        (_HERE / "repository_maintenance_policy.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_case", "domain.maintenance_case")
    )
    files["prompts/coding.py"] += (
        '\nPRIMARY_SYSTEM_PROMPT += "\\nOnly range_ops.py and tests/test_range_ops.py may change. '
        "For write_file and edit_file explicitly set max_bytes=16384 or less. "
        "Patches permit at most two operations and files up to 16384 bytes. "
        "Reproduce the failure before repair; then run format, lint, test and "
        'independent-range-probe on the final revision."\n'
    )
    files["domain/maintenance_acceptance.py"] = (
        (_HERE / "repository_maintenance_acceptance.py")
        .read_text()
        .replace("tests.qualification.repository_maintenance_case", "domain.maintenance_case")
        .replace(
            "tests.qualification.repository_maintenance_probe", "environments.maintenance_probe"
        )
        .replace(
            "tests.qualification.repository_maintenance_toolchain",
            "environments.maintenance_toolchain",
        )
    )
    files["Dockerfile.coding"] = _replace_once(
        files["Dockerfile.coding"],
        "COPY pyproject.toml uv.lock ./\n",
        "COPY pyproject.toml uv.lock ./\n"
        f"COPY {_PROBE_INPUT} /opt/cayu-acceptance/range_probe.py\n",
    )
    files["build_coding_image.py"] = _replace_once(
        files["build_coding_image.py"],
        '    "docker-coding-build.json": 16 * 1024,\n',
        f'    "docker-coding-build.json": 16 * 1024,\n    "{_PROBE_INPUT}": 16 * 1024,\n',
    )
    for name in ("build_coding_image.py", "docker-coding-image.json"):
        files[name] = _replace_once(
            files[name],
            '"profile_id": "maintenance-app-python"',
            '"profile_id": "repository-maintenance-python"',
        )
        files[name] = _replace_once(
            files[name],
            '"profile_revision": "2"',
            '"profile_revision": "1"',
        )
    source = _replace_once(
        files["operations/coding.py"],
        '    "docker-coding-build.json",\n',
        f'    "docker-coding-build.json",\n    "{_PROBE_INPUT}",\n',
    )
    source = _replace_once(
        source,
        "from environments.coding import workspace_candidate\n",
        "from environments.coding import workspace_candidate\n"
        "from environments.maintenance_toolchain import maintenance_checks, maintenance_toolchain\n",
    )
    source = _replace_once(
        source,
        "from policies.coding import require_coding_tool_policy\n",
        "from policies.coding import require_coding_tool_policy\n"
        "from policies.maintenance import MaintenancePatchScopeRule, MaintenanceWriteBoundRule\n"
        "from domain.maintenance_case import ALLOWED_CHANGE_PATHS\n",
    )
    source = _replace_once(
        source,
        'name="cayu.generated.coding.primary_tool_policy"',
        'name="repository-maintenance.primary-tool-policy"',
    )
    source = _replace_once(
        source,
        '"apply_patch": (RequiredFieldRule("operations"),),',
        '"apply_patch": (MaintenancePatchScopeRule(),),',
    )
    for name in ("write_file", "edit_file"):
        source = _replace_once(
            source,
            f'"{name}": _path_rules(required=True),',
            f'"{name}": (*_path_rules(required=True), MaintenanceWriteBoundRule()),',
        )
    source = _replace_once(
        source,
        "        ApplyPatchTool(),\n",
        "        ApplyPatchTool(max_operations=2, max_file_bytes=16 * 1024),\n",
    )
    source = _replace_once(
        source,
        "        command_policy=command_policy,\n        toolchain_profile=toolchain_profile,\n",
        "        command_policy=command_policy,\n        toolchain_profile=toolchain_profile,\n"
        "        max_model_output_bytes=256,\n",
    )
    source = _replace_once(
        source,
        '_PYTHON_TOOLCHAIN_PROFILE_ID = "maintenance-app-python"',
        '_PYTHON_TOOLCHAIN_PROFILE_ID = "repository-maintenance-python"',
    )
    source = _replace_once(
        source,
        '_PYTHON_TOOLCHAIN_PROFILE_REVISION = "2"',
        '_PYTHON_TOOLCHAIN_PROFILE_REVISION = "1"',
    )
    source = _replace_once(
        source,
        '_CHECK_NAMES = ("format", "lint", "test")',
        '_CHECK_NAMES = ("format", "independent-range-probe", "lint", "test")',
    )
    source = _replace_once(
        source,
        '_COMMAND_SELECTOR_NAMES = ("focused-test", "lint-file", "python-version")',
        '_COMMAND_SELECTOR_NAMES = ("python-version",)',
    )
    files["operations/coding.py"] = _specialize_python(
        source,
        functions={
            "_path_rules": """
def _path_rules(*, required):
    rules = []
    if required:
        rules.append(RequiredAllowlistRule("path", values=ALLOWED_CHANGE_PATHS))
    rules.append(DenyPatternRule("path", patterns=_DENIED_PATH_PATTERNS))
    return tuple(rules)
""",
            "_python_toolchain_profile": """
def _python_toolchain_profile(image_identity, *, profile_id=None, profile_revision=None,
                              platform_architecture="amd64", dependency_inputs=(),
                              trusted_build_context_sha256=None):
    # Build metadata has already been validated by _read_docker_toolchain_profile.
    # Workspace dependencies belong to the target corpus, not the application.
    return maintenance_toolchain(image_identity=image_identity,
                                 architecture=platform_architecture,
                                 build_context_sha256=trusted_build_context_sha256)
""",
            "_named_checks": """
def _named_checks(profile):
    checks = maintenance_checks()
    for check in checks:
        authority = profile.command_authority(check.name)
        if authority is None or authority.command_argv() != tuple(check.command.argv):
            raise RuntimeError("Maintenance check escapes admitted toolchain authority")
    return checks
""",
        },
    )
    files["workflows/coding_product.py"] = _replace_once(
        files["workflows/coding_product.py"],
        """CodingSettlementPolicy(
                    required_checks=("format", "lint", "test"),
                    reviewer_required=False,
                    human_approval_required=False,
                )""",
        "default_maintenance_settlement()",
    )
    files["workflows/coding_product.py"] = _replace_once(
        files["workflows/coding_product.py"],
        "from domain.coding_product import CodingProductTask\n",
        "from domain.coding_product import CodingProductTask\n"
        "from domain.maintenance_case import SEED_BASE_REVISION\n"
        "from domain.maintenance_request import default_maintenance_settlement\n"
        "from domain.maintenance_acceptance import MaintenanceAcceptance, verify_maintenance_result\n",
    )
    files["workflows/coding_product.py"] = _replace_once(
        files["workflows/coding_product.py"],
        "    RunRequest,\n",
        "    RunRequest,\n    RunLimits,\n",
    )
    files["workflows/coding_product.py"] = _replace_once(
        files["workflows/coding_product.py"],
        '            environment_name="coding",\n',
        '            environment_name="coding",\n'
        "            max_steps=8,\n"
        '            limits=RunLimits(max_elapsed_seconds=180, scope="session"),\n',
    )
    files["workflows/coding_product.py"] = _replace_once(
        files["workflows/coding_product.py"],
        "    async def _prepare(\n",
        '''    async def verify(
        self, task: CodingProductTask, publication: CodingProductPublication,
    ) -> MaintenanceAcceptance:
        """Verify one sealed result; do not authorize or perform delivery."""
        _, request, _ = await self._prepare(task, require_existing=True)
        return await verify_maintenance_result(
            request=request, reference=publication.result_reference,
            repository=CodingProductArtifactRepository(self.artifact_store),
            toolchain=self.toolchain_profile, workspace=self.source_workspace,
            public_authority_alias_codec=self.app.session_store.public_authority_alias_codec,
        )

    async def _prepare(
''',
    )
    files["workflows/coding_product.py"] = _replace_once(
        files["workflows/coding_product.py"],
        '        messages = [Message.text("user", task.instruction)]\n',
        """        if task.settlement is not None and (
            type(task.settlement) is not CodingSettlementPolicy
            or task.settlement.required_checks != (
                "format", "independent-range-probe", "lint", "test"
            )
        ):
            raise ValueError("Maintenance tasks require the fixed check set.")
        messages = [Message.text("user", task.instruction)]
""",
    )
    files["workflows/coding_product.py"] = _replace_once(
        files["workflows/coding_product.py"],
        "        request = await admit_or_recover_coding_product_request(\n",
        "        if source_git_baseline.head_revision != SEED_BASE_REVISION:\n"
        '            raise ValueError("Maintenance source does not match the fixed Git base.")\n'
        "        request = await admit_or_recover_coding_product_request(\n",
    )
    if database == "postgres":
        files.update(compose_assets())
        files[".gitignore"] += (
            "\ndeployment/*.env\ndeployment/wheels/\ndeployment/requirements.lock\ndeployment/backups/\n"
        )
    return files
