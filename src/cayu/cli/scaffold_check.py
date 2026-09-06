"""Read-only diagnostics for projects declaring the Cayu scaffold convention."""

from __future__ import annotations

import ast
import builtins
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from cayu.cli.scaffold_plan import (
    ApplicationPlan,
    ScaffoldPlanError,
    normalize_application_plan,
    preset_spec,
)
from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic
from cayu.runtime.manifest import AppManifest

_VERIFY = "cayu check --fail-on warning --json"
_DOCS = "cayu guide applications#convention"
_MAX_DECLARATION_MODULES = 1024


class _DeclarationGraphLimit(Exception):
    pass


_IMPORT_SAFETY_TAGS = ("authoring", "configuration", "deploy", "providers", "security")
_INERT_GENERATED_COLLECTION_NAMES = frozenset({"_SYSTEM_PROMPT_PARTS", "_WORKFLOW_TOOL_NAMES"})
_COMPLETE_REQUIRED_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "app.py",
    "run.py",
    "configuration/__init__.py",
    "configuration/settings.py",
    "configuration/providers.py",
    "configuration/storage.py",
    "configuration/runtime.py",
    "agents/__init__.py",
    "agents/agent.py",
    "agents/registration.py",
    "prompts/__init__.py",
    "prompts/agent.py",
    "tools/__init__.py",
    "tools/registration.py",
    "policies/__init__.py",
    "policies/tools.py",
    "policies/context.py",
    "policies/execution.py",
    "policies/egress.py",
    "policies/budgets.py",
    "policies/retries.py",
    "environments/__init__.py",
    "environments/registration.py",
    "environments/local.py",
    "workflows/__init__.py",
    "operations/__init__.py",
    "operations/tasks.py",
    "operations/workers.py",
    "operations/watchers.py",
    "operations/approvals.py",
    "operations/completion.py",
    "operations/recovery.py",
    "knowledge/__init__.py",
    "knowledge/retrieval.py",
    "knowledge/curation.py",
    "knowledge/maintenance.py",
    "knowledge/seeds",
    "memory/__init__.py",
    "memory/context.py",
    "memory/recall.py",
    "domain/__init__.py",
    "integrations/__init__.py",
    "integrations/mcp.py",
    "evals/__init__.py",
    "evals/agent.py",
    "observability/__init__.py",
    "observability/events.py",
    "observability/tracing.py",
    "tests/test_application.py",
    "tests/test_agent.py",
    "tests/test_architecture.py",
    "data",
)
_REQUIRED_DIRECTORY_PATHS = frozenset({"data", "knowledge/seeds"})
_APPLICATION_SOURCE_DIRECTORIES = (
    "configuration",
    "agents",
    "prompts",
    "tools",
    "policies",
    "environments",
    "workflows",
    "operations",
    "knowledge",
    "memory",
    "domain",
    "integrations",
    "evals",
    "observability",
)


def check_declared_scaffold_source(
    root: Path,
    *,
    tags: frozenset[str] = frozenset(),
    deploy_only: bool = False,
) -> tuple[ProjectDiagnostic, ...]:
    """Check declared source structure without importing project code."""

    diagnostics, _ = _source_diagnostics(root)
    return _filter(diagnostics, tags=tags, deploy_only=deploy_only)


def check_declared_scaffold(
    root: Path,
    manifest: AppManifest,
    *,
    tags: frozenset[str] = frozenset(),
    deploy_only: bool = False,
) -> tuple[ProjectDiagnostic, ...]:
    """Return convention findings, or nothing for a freeform project."""

    source_diagnostics, declared = _source_diagnostics(root)
    diagnostics = list(source_diagnostics)
    if declared:
        diagnostics.extend(_check_registration_provenance(manifest))
    return _filter(tuple(diagnostics), tags=tags, deploy_only=deploy_only)


def _source_diagnostics(
    root: Path,
) -> tuple[tuple[ProjectDiagnostic, ...], bool | None]:
    """Return source-only findings and the presence of a valid convention."""

    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return (), None
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return (), None
    contract = _scaffold_contract(document)
    if contract is None:
        return (), None

    diagnostics: list[ProjectDiagnostic] = []
    convention = contract.get("convention")
    if convention != 1:
        diagnostics.append(
            _diagnostic(
                code="SCAFFOLD_CONVENTION_UNSUPPORTED",
                path="pyproject.toml:[tool.cayu.scaffold].convention",
                message=f"Declared scaffold convention {convention!r} is not supported.",
                hint=(
                    "Use the installed Cayu version that owns this convention, or perform "
                    "an explicit reviewed custom-layout migration."
                ),
                parameters={"observed_convention": convention},
                severity=DiagnosticSeverity.ERROR,
            )
        )
        return tuple(diagnostics), None

    try:
        plan = _normalized_declared_plan(contract)
    except ScaffoldPlanError as exc:
        diagnostics.append(
            _diagnostic(
                code="SCAFFOLD_CONTRACT_INVALID",
                path="pyproject.toml:[tool.cayu.scaffold]",
                message=f"The declared scaffold plan is invalid ({exc.code}).",
                hint="Restore the normalized plan emitted by `cayu new --dry-run --json`.",
                parameters={"reason": exc.code},
                severity=DiagnosticSeverity.ERROR,
            )
        )
        return tuple(diagnostics), None

    required = tuple(
        relative
        for relative in _COMPLETE_REQUIRED_PATHS
        if relative != "evals/agent.py" or "evals" in plan.capabilities
    )
    for relative in required:
        if not _required_path_is_present(root, relative):
            expected_kind = "directory" if relative in _REQUIRED_DIRECTORY_PATHS else "file"
            diagnostics.append(
                _diagnostic(
                    code="SCAFFOLD_LAYOUT_PATH_MISSING",
                    path=relative,
                    message=(
                        f"Declared scaffold {expected_kind} {relative!r} is missing or has "
                        "the wrong filesystem type."
                    ),
                    hint=(
                        "Restore the owning package or perform an explicit custom-layout "
                        "migration; do not delete the scaffold contract to hide drift."
                    ),
                    parameters={"missing_path": relative, "expected_kind": expected_kind},
                    severity=DiagnosticSeverity.ERROR,
                )
            )

    diagnostics.extend(_check_selected_plan_source(root, document, plan))
    diagnostics.extend(_check_composition_root(root))
    diagnostics.extend(_check_import_inertness(root))
    return tuple(diagnostics), True


def _normalized_declared_plan(contract: Mapping[str, object]) -> ApplicationPlan:
    required_strings = ("preset", "database", "provider", "execution")
    for field in required_strings:
        if type(contract.get(field)) is not str:
            raise ScaffoldPlanError("invalid_" + field, f"{field} must be a string")
    capabilities = contract.get("capabilities")
    if not isinstance(capabilities, list) or any(type(item) is not str for item in capabilities):
        raise ScaffoldPlanError(
            "invalid_capabilities",
            "capabilities must be a list of strings",
        )

    preset = cast("str", contract["preset"])
    selected_preset = preset_spec(preset)
    recorded = cast("list[str]", capabilities)
    defaults = set(selected_preset.default_capabilities)
    selected = set(recorded)
    plan = normalize_application_plan(
        name="declared-scaffold",
        agent_name="declared-agent",
        preset=preset,
        database=cast("str", contract["database"]),
        provider=cast("str", contract["provider"]),
        execution=cast("str", contract["execution"]),
        with_capabilities=tuple(sorted(selected - defaults)),
        without_capabilities=tuple(sorted(defaults - selected)),
    )
    if tuple(recorded) != plan.capabilities:
        raise ScaffoldPlanError(
            "capabilities_not_normalized",
            "capabilities must match the normalized selected plan",
        )
    return plan


def _check_selected_plan_source(
    root: Path,
    document: object,
    plan: ApplicationPlan,
) -> tuple[ProjectDiagnostic, ...]:
    """Verify that every declared selection still matches generated source."""

    diagnostics: list[ProjectDiagnostic] = []

    def drift(field: str, expected: object, observed: object, path: str) -> None:
        if observed == expected:
            return
        diagnostics.append(
            _diagnostic(
                code="SCAFFOLD_PLAN_DRIFT",
                path=path,
                message=(
                    f"Declared scaffold {field} {expected!r} does not match source "
                    f"evidence {observed!r}."
                ),
                hint=(
                    "Restore the selected generated variant or perform an explicit reviewed "
                    "plan migration; changing only scaffold metadata is not a migration."
                ),
                parameters={"field": field, "expected": expected, "observed": observed},
                severity=DiagnosticSeverity.ERROR,
            )
        )

    cayu = _tool_cayu(document)
    session_store = cayu.get("session_store")
    backend = (
        cast("Mapping[str, object]", session_store).get("backend")
        if isinstance(session_store, Mapping)
        else None
    )
    drift("database", plan.database, backend, "pyproject.toml:[tool.cayu.session_store].backend")

    configuration_path = "configuration/settings.py"
    provider = _static_assignment(root / configuration_path, "_SCAFFOLDED_PROVIDER")
    expected_provider = None if plan.provider == "neutral" else plan.provider
    drift("provider", expected_provider, provider, configuration_path)

    database = _static_assignment(root / "configuration/settings.py", "SCAFFOLDED_DATABASE")
    drift("database", plan.database, database, "configuration/settings.py")
    storage_relative = (
        "configuration/coding_storage.py" if plan.preset == "coding" else "configuration/storage.py"
    )
    storage = root / storage_relative
    if plan.preset == "coding":
        storage_profile = _static_assignment(storage, "GENERATED_STORE_PROFILE")
        drift("database", plan.database, storage_profile, storage_relative)
    else:
        expected_store = (
            "PostgresSessionStore" if plan.database == "postgres" else "SQLiteSessionStore"
        )
        observed_store = expected_store if _source_contains_name(storage, expected_store) else None
        drift("database", expected_store, observed_store, storage_relative)

    logging = _runtime_logging_value(root / "configuration/runtime.py")
    expected_logging = "observability" in plan.capabilities
    drift("capabilities", expected_logging, logging, "configuration/runtime.py")

    service_factory = cayu.get("service_factory")
    service_files = ("service.py", "product_store.py", "tests/test_public_service_security.py")
    coding_files = (
        "configuration/coding_storage.py",
        "operations/coding.py",
        "agents/reviewer.py",
        "tools/coding.py",
        "policies/coding.py",
        "prompts/coding.py",
        "tests/test_coding_composition.py",
    )
    if plan.preset == "service":
        drift("preset", "service:build_service", service_factory, "pyproject.toml:[tool.cayu]")
        for relative in service_files:
            drift("preset", "present", _path_state(root / relative), relative)
    else:
        drift("preset", None, service_factory, "pyproject.toml:[tool.cayu]")
    app_path = root / "app.py"
    if plan.preset == "coding":
        for relative in coding_files:
            drift("preset", "present", _path_state(root / relative), relative)
        observed_factory = (
            "build_coding_app" if _source_contains_name(app_path, "build_coding_app") else None
        )
        drift("preset", "build_coding_app", observed_factory, "app.py")
    elif plan.preset == "agent":
        observed_factory = (
            "register_agents" if _source_contains_name(app_path, "register_agents") else None
        )
        drift("preset", "register_agents", observed_factory, "app.py")

    docker_files = (
        "Dockerfile.coding",
        "docker-coding-build.json",
        "docker-coding-image.json",
        "build_coding_image.py",
        "tests/test_project.py",
    )
    selected_docker_files = docker_files if plan.execution == "docker" else docker_files[:-1]
    for relative in selected_docker_files:
        expected = "present" if plan.execution == "docker" else "absent"
        drift("execution", expected, _path_state(root / relative), relative)
    return tuple(diagnostics)


def _tool_cayu(document: object) -> Mapping[str, object]:
    if not isinstance(document, Mapping):
        return {}
    tool = cast("Mapping[str, object]", document).get("tool")
    if not isinstance(tool, Mapping):
        return {}
    cayu = cast("Mapping[str, object]", tool).get("cayu")
    return cast("Mapping[str, object]", cayu) if isinstance(cayu, Mapping) else {}


def _path_state(path: Path) -> str:
    return "present" if path.is_file() and not path.is_symlink() else "absent"


def _parsed_module(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    except (OSError, UnicodeError, SyntaxError):
        return None


def _static_assignment(path: Path, name: str) -> object:
    tree = _parsed_module(path)
    if tree is None:
        return None
    for node in tree.body:
        targets: tuple[ast.expr, ...] = ()
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
            value = node.value
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            continue
        try:
            return ast.literal_eval(value)
        except (ValueError, TypeError):
            return None
    return None


def _source_contains_name(path: Path, name: str) -> bool:
    tree = _parsed_module(path)
    return tree is not None and any(
        isinstance(node, ast.Name) and node.id == name for node in ast.walk(tree)
    )


def _runtime_logging_value(path: Path) -> bool | None:
    tree = _parsed_module(path)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RuntimeOptions"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "enable_logging" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value if type(keyword.value.value) is bool else None
    return None


def _required_path_is_present(root: Path, relative: str) -> bool:
    path = root
    for component in Path(relative).parts:
        path /= component
        if path.is_symlink():
            return False
    if relative in _REQUIRED_DIRECTORY_PATHS:
        return path.is_dir()
    return path.is_file()


def _scaffold_contract(document: object) -> Mapping[str, object] | None:
    if not isinstance(document, Mapping):
        return None
    root = cast("Mapping[str, object]", document)
    tool = root.get("tool")
    if not isinstance(tool, Mapping):
        return None
    cayu = cast("Mapping[str, object]", tool).get("cayu")
    if not isinstance(cayu, Mapping):
        return None
    scaffold = cast("Mapping[str, object]", cayu).get("scaffold")
    return cast("Mapping[str, object]", scaffold) if isinstance(scaffold, Mapping) else None


def _check_composition_root(root: Path) -> tuple[ProjectDiagnostic, ...]:
    app_path = root / "app.py"
    if not app_path.is_file():
        return ()
    try:
        tree = ast.parse(app_path.read_text(encoding="utf-8"), filename="app.py")
    except (OSError, UnicodeError, SyntaxError):
        return ()

    diagnostics: list[ProjectDiagnostic] = []
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    functions = [
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if "build_app" not in functions:
        diagnostics.append(
            _diagnostic(
                code="SCAFFOLD_APP_FACTORY_MISSING",
                path="app.py",
                message="The declared scaffold does not define the build_app factory.",
                hint=(
                    "Restore build_app() in app.py so Cayu and coding agents have one "
                    "stable application-construction entry point."
                ),
                parameters={"expected_factory": "app:build_app"},
                severity=DiagnosticSeverity.ERROR,
            )
        )
    unexpected_functions = [name for name in functions if name != "build_app"]
    if classes or unexpected_functions:
        diagnostics.append(
            _diagnostic(
                code="SCAFFOLD_APP_COMPOSITION_DRIFT",
                path="app.py",
                message=(
                    "The declared composition root contains implementation definitions "
                    f"(classes={classes!r}, functions={unexpected_functions!r})."
                ),
                hint=(
                    "Move implementation into its owning package and keep only explicit "
                    "construction and registration in app.py."
                ),
                parameters={
                    "classes": classes,
                    "unexpected_functions": unexpected_functions,
                },
            )
        )
    return tuple(diagnostics)


def _check_import_inertness(root: Path) -> tuple[ProjectDiagnostic, ...]:
    diagnostics: list[ProjectDiagnostic] = []
    try:
        declaration_proofs = _project_declaration_proofs(root)
    except _DeclarationGraphLimit:
        return (
            _diagnostic(
                code="SCAFFOLD_MODULE_SOURCE_INVALID",
                path=".",
                message="The explicit source dependency graph exceeds the bounded declaration proof.",
                hint="Reduce the explicit dependency graph before importing the application.",
                parameters={
                    "reason": "declaration_graph_limit",
                    "max_modules": _MAX_DECLARATION_MODULES,
                },
                severity=DiagnosticSeverity.ERROR,
                tags=_IMPORT_SAFETY_TAGS,
            ),
        )
    for relative in _APPLICATION_SOURCE_DIRECTORIES:
        package = root / relative
        if package.is_symlink():
            diagnostics.append(
                _diagnostic(
                    code="SCAFFOLD_MODULE_SOURCE_INVALID",
                    path=relative,
                    message="A declared application package is a symbolic link.",
                    hint="Replace the link with reviewed ordinary Python source in the project.",
                    parameters={"reason": "symbolic_link"},
                    severity=DiagnosticSeverity.ERROR,
                    tags=_IMPORT_SAFETY_TAGS,
                )
            )
    for path in _application_module_paths(root):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            diagnostics.append(
                _diagnostic(
                    code="SCAFFOLD_MODULE_SOURCE_INVALID",
                    path=relative,
                    message="A declared application module is a symbolic link.",
                    hint="Replace the link with reviewed ordinary Python source in the project.",
                    parameters={"reason": "symbolic_link"},
                    severity=DiagnosticSeverity.ERROR,
                    tags=_IMPORT_SAFETY_TAGS,
                )
            )
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            diagnostics.append(
                _diagnostic(
                    code="SCAFFOLD_MODULE_SOURCE_INVALID",
                    path=relative,
                    message=f"Declared application module source is not safely parseable ({type(exc).__name__}).",
                    hint="Restore parseable UTF-8 Python source before importing the application.",
                    parameters={"reason": type(exc).__name__},
                    severity=DiagnosticSeverity.ERROR,
                    tags=_IMPORT_SAFETY_TAGS,
                )
            )
            continue
        import_bindings = _import_bindings(tree)
        rebound_names = _module_rebound_names(tree)
        declarative_identities = _declarative_identity_names(
            tree,
            import_bindings=import_bindings,
            rebound_names=rebound_names,
        )
        declarative_subscriptions = _declarative_subscription_names(root, tree)
        declarative_bases = declaration_proofs.proven.get(path, {})
        for node in tree.body:
            unsafe_import = next(
                (
                    import_node
                    for import_node in _import_time_import_from_nodes(node)
                    if not _project_local_import_from_is_declarative(
                        root,
                        importing_path=path,
                        node=import_node,
                    )
                ),
                None,
            )
            unsafe_node: ast.expr | ast.stmt | None = unsafe_import
            if unsafe_node is None:
                unsafe_node = _unsafe_import_time_expression(
                    node,
                    import_bindings=import_bindings,
                    rebound_names=rebound_names,
                    declarative_identities=declarative_identities,
                    declarative_subscriptions=declarative_subscriptions,
                    project_root=root,
                    declarative_bases=declarative_bases,
                )
            if unsafe_node is None:
                continue
            if any(unsafe_node is base for base, _scope in _import_time_class_base_uses(node)):
                primary = _unsafe_import_time_expression(
                    node,
                    import_bindings=import_bindings,
                    rebound_names=rebound_names,
                    declarative_identities=declarative_identities,
                    declarative_subscriptions=declarative_subscriptions,
                    project_root=root,
                    declarative_bases=declaration_proofs.structural.get(path, {}),
                )
                if primary is not None:
                    unsafe_node = primary
            unproven_base = any(
                unsafe_node is base for base, _scope in _import_time_class_base_uses(node)
            )
            diagnostics.append(
                _import_side_effect_diagnostic(
                    relative,
                    unsafe_node,
                    unproven_base=unproven_base,
                    blocking_modules=tuple(
                        sorted(
                            cause.relative_to(root).as_posix()
                            for cause in declaration_proofs.blockers.get(path, ())
                        )
                    )
                    if unproven_base
                    else (),
                )
            )
    # Include primary failures in explicitly imported helpers outside the normal
    # concern directories. A missing inheritance proof must not hide the cause.
    reported = {diagnostic.path for diagnostic in diagnostics}
    for path, failures in sorted(declaration_proofs.failures.items()):
        relative = path.relative_to(root).as_posix()
        for unsafe_node in failures:
            location = f"{relative}:{unsafe_node.lineno}"
            if location not in reported:
                diagnostics.append(_import_side_effect_diagnostic(relative, unsafe_node))
                reported.add(location)
    for path in sorted(declaration_proofs.blockers.keys() - declaration_proofs.structural.keys()):
        relative = path.relative_to(root).as_posix()
        if relative not in reported:
            diagnostics.append(
                _diagnostic(
                    code="SCAFFOLD_MODULE_SOURCE_INVALID",
                    path=relative,
                    message="An explicitly imported local dependency has unavailable or unsafe source.",
                    hint="Restore ordinary parseable source before importing the application.",
                    parameters={"reason": "dependency_source_invalid"},
                    severity=DiagnosticSeverity.ERROR,
                    tags=_IMPORT_SAFETY_TAGS,
                )
            )
    return tuple(diagnostics)


def _import_side_effect_diagnostic(
    relative: str,
    unsafe_node: ast.expr | ast.stmt,
    *,
    unproven_base: bool = False,
    blocking_modules: tuple[str, ...] = (),
) -> ProjectDiagnostic:
    return _diagnostic(
        code="SCAFFOLD_IMPORT_SIDE_EFFECT",
        path=f"{relative}:{unsafe_node.lineno}",
        message=(
            "Class inheritance cannot be proven declarative; the base or its "
            "import dependency closure is unproven."
            if unproven_base
            else "Import-time execution in a declared application module "
            "cannot be proven declarative."
        ),
        hint=(
            "Inspect unsupported_expression findings in the base's defining "
            "module and dependencies before changing the dependent class."
            if unproven_base
            else "Move external or lifecycle work behind an explicit builder or "
            "lifecycle entry point."
        ),
        parameters={
            "reason": ("unproven_class_base" if unproven_base else "unsupported_expression"),
            "symbol": (
                _expression_root_name(
                    unsafe_node.func if isinstance(unsafe_node, ast.Call) else unsafe_node
                )
                if isinstance(unsafe_node, ast.expr)
                else None
            ),
            "expression_kind": type(unsafe_node).__name__,
            "line": unsafe_node.lineno,
            "module": relative,
            "blocking_modules": list(blocking_modules[:8]),
            "blocking_module_count": len(blocking_modules),
        },
        severity=DiagnosticSeverity.ERROR,
        tags=_IMPORT_SAFETY_TAGS,
    )


def _application_module_paths(root: Path) -> tuple[Path, ...]:
    paths = {path for path in root.glob("*.py") if path.is_file() or path.is_symlink()}
    for relative in _APPLICATION_SOURCE_DIRECTORIES:
        package = root / relative
        if not package.is_dir() or package.is_symlink():
            continue
        paths.update(path for path in package.rglob("*.py") if path.is_file() or path.is_symlink())
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def _import_time_import_from_nodes(node: ast.stmt) -> tuple[ast.ImportFrom, ...]:
    return tuple(
        item for item in _import_time_import_nodes(node) if isinstance(item, ast.ImportFrom)
    )


def _import_time_import_nodes(node: ast.stmt) -> tuple[ast.Import | ast.ImportFrom, ...]:
    imports: list[ast.Import | ast.ImportFrom] = []

    def add_suite(statements: list[ast.stmt]) -> None:
        for statement in statements:
            imports.extend(_import_time_import_nodes(statement))

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports.append(node)
    elif isinstance(node, ast.ClassDef):
        add_suite(node.body)
    elif isinstance(node, ast.If):
        add_suite(node.orelse if _is_main_guard(node.test) else node.body + node.orelse)
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        add_suite(node.body)
        add_suite(node.orelse)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        add_suite(node.body)
    elif isinstance(node, (ast.Try, ast.TryStar)):
        add_suite(node.body)
        for handler in node.handlers:
            add_suite(handler.body)
        add_suite(node.orelse)
        add_suite(node.finalbody)
    elif isinstance(node, ast.Match):
        for case in node.cases:
            add_suite(case.body)
    return tuple(imports)


def _project_local_import_from_is_declarative(
    root: Path,
    *,
    importing_path: Path,
    node: ast.ImportFrom,
) -> bool:
    is_project_local, module_path = _project_local_import_module_path(
        root,
        importing_path=importing_path,
        module=node.module,
        level=node.level,
    )
    if any(alias.name == "*" for alias in node.names):
        return False
    if not is_project_local:
        return True
    if module_path is None or module_path.is_symlink():
        return False
    if module_path.is_dir():
        # Namespace packages have no source-level exports. Every requested name
        # must resolve to an explicit child; the graph checks that child's source.
        return all(
            _project_local_import_module_path(
                root,
                importing_path=importing_path,
                module=f"{node.module}.{alias.name}" if node.module else alias.name,
                level=node.level,
            )[1]
            is not None
            for alias in node.names
        )
    if _module_may_bind_name(module_path, "__getattr__"):
        return False
    return all(alias.name != "*" for alias in node.names)


def _project_local_import_module_path(
    root: Path,
    *,
    importing_path: Path,
    module: str | None,
    level: int,
) -> tuple[bool, Path | None]:
    """Resolve local source or a namespace directory without importing it."""
    if level:
        relative = importing_path.relative_to(root)
        package_parts = relative.parent.parts
        parents = level - 1
        if parents > len(package_parts):
            return True, None
        prefix = package_parts[: len(package_parts) - parents]
        module_parts = () if module is None else tuple(module.split("."))
        parts = (*prefix, *module_parts)
        is_project_local = True
    else:
        if module is None:
            return False, None
        parts = tuple(module.split("."))
        is_project_local = _import_binding_is_project_local(root, (module, None))

    # A same-named module shadows a namespace directory and cannot contain
    # imported children. Do not prove paths that Python cannot traverse.
    for length in range(1, len(parts)):
        parent = root.joinpath(*parts[:length])
        initializer = parent / "__init__.py"
        if not (initializer.is_file() or initializer.is_symlink()) and (
            parent.with_suffix(".py").is_file() or parent.with_suffix(".py").is_symlink()
        ):
            return True, None
    candidate = root.joinpath(*parts)
    # Python prefers a regular package over a same-named module file.
    package_init = candidate / "__init__.py"
    if package_init.is_file() or package_init.is_symlink():
        return True, package_init
    module_file = candidate.with_suffix(".py")
    if module_file.is_file() or module_file.is_symlink():
        return True, module_file
    if candidate.is_dir() or candidate.is_symlink():
        return True, candidate
    return is_project_local, None


def _module_may_bind_name(path: Path, name: str) -> bool:
    tree = _parsed_module(path)
    return tree is not None and any(
        _import_time_statement_may_bind_name(node, name) for node in tree.body
    )


def _import_time_statement_may_bind_name(node: ast.stmt, name: str) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name == name or any(
            _expression_may_bind_name(expression, name)
            for expression in (
                *node.decorator_list,
                *node.args.defaults,
                *node.args.kw_defaults,
            )
        )
    if isinstance(node, ast.ClassDef):
        return node.name == name or any(
            _expression_may_bind_name(expression, name)
            for expression in (
                *node.decorator_list,
                *node.bases,
                *(keyword.value for keyword in node.keywords),
            )
        )
    if isinstance(node, ast.Assign):
        return any(
            _assignment_target_binds_name(target, name) for target in node.targets
        ) or _expression_may_bind_name(node.value, name)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return _assignment_target_binds_name(node.target, name) or _expression_may_bind_name(
            node.value, name
        )
    if isinstance(node, ast.Import):
        return any((alias.asname or alias.name.partition(".")[0]) == name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return any(
            alias.name != "*" and (alias.asname or alias.name) == name for alias in node.names
        )
    if isinstance(node, ast.Expr):
        return _expression_may_bind_name(node.value, name)

    suites: list[list[ast.stmt]] = []
    binds_in_header = False
    if isinstance(node, ast.If):
        binds_in_header = _expression_may_bind_name(node.test, name)
        suites.extend((node.body, node.orelse))
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        binds_in_header = _assignment_target_binds_name(
            node.target, name
        ) or _expression_may_bind_name(node.iter, name)
        suites.extend((node.body, node.orelse))
    elif isinstance(node, ast.While):
        binds_in_header = _expression_may_bind_name(node.test, name)
        suites.extend((node.body, node.orelse))
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        binds_in_header = any(
            _expression_may_bind_name(item.context_expr, name)
            or (
                item.optional_vars is not None
                and _assignment_target_binds_name(item.optional_vars, name)
            )
            for item in node.items
        )
        suites.append(node.body)
    elif isinstance(node, (ast.Try, ast.TryStar)):
        suites.append(node.body)
        suites.extend(handler.body for handler in node.handlers)
        suites.extend((node.orelse, node.finalbody))
    elif isinstance(node, ast.Match):
        binds_in_header = _expression_may_bind_name(node.subject, name) or any(
            _match_pattern_may_bind_name(case.pattern, name)
            or _expression_may_bind_name(case.guard, name)
            for case in node.cases
        )
        suites.extend(case.body for case in node.cases)
    else:
        binds_in_header = _expression_may_bind_name(node, name)
    return binds_in_header or any(
        _import_time_statement_may_bind_name(statement, name)
        for suite in suites
        for statement in suite
    )


def _expression_may_bind_name(expression: ast.AST | None, name: str) -> bool:
    return expression is not None and any(
        isinstance(node, ast.NamedExpr) and _assignment_target_binds_name(node.target, name)
        for node in ast.walk(expression)
    )


def _match_pattern_may_bind_name(pattern: ast.pattern, name: str) -> bool:
    return any(
        isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name
        for node in ast.walk(pattern)
    )


def _assignment_target_binds_name(target: ast.expr, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
        for node in ast.walk(target)
    )


@dataclass(frozen=True)
class _DeclarativeBinding:
    line: int
    kind: Literal["class", "model", "enum", "literal", "annotation", "function"] = "class"
    enum_members: frozenset[str] | None = None


def _is_pydantic_base(
    base: ast.expr,
    imports: Mapping[str, tuple[str, str | None]],
    bases: Mapping[str, _DeclarativeBinding],
) -> bool:
    return isinstance(base, ast.Name) and (
        imports.get(base.id) == ("pydantic", "BaseModel")
        or (base.id in bases and bases[base.id].kind == "model")
    )


def _trusted_symbol(
    expression: ast.expr,
    imports: Mapping[str, tuple[str, str | None]],
    rebound: frozenset[str],
    root: Path,
) -> tuple[str, str | None] | None:
    if isinstance(expression, ast.Name):
        name = expression.id
        binding = imports.get(name)
    elif isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        name = expression.value.id
        module, symbol = imports.get(name, ("", None))
        binding = (module, expression.attr) if symbol is None else None
    else:
        return None
    if name in rebound or binding is None or _import_binding_is_project_local(root, binding):
        return None
    return binding


def _stdlib_declaration_is_inert(
    call: ast.Call,
    imports: Mapping[str, tuple[str, str | None]],
    rebound: frozenset[str],
    root: Path,
    proofs: Mapping[str, _DeclarativeBinding],
) -> bool:
    """Admit identity declarations with literal data, never arbitrary factories."""
    binding = _trusted_symbol(call.func, imports, rebound, root)
    if (
        isinstance(call.func, ast.Name)
        and call.func.id == "set"
        and call.func.id not in imports
        and call.func.id not in rebound
    ):
        return not call.args and not call.keywords
    if (
        isinstance(call.func, ast.Name)
        and call.func.id == "len"
        and "len" not in imports
        and "len" not in rebound
    ):
        return (
            len(call.args) == 1
            and not call.keywords
            and isinstance(call.args[0], ast.Name)
            and (proof := proofs.get(call.args[0].id)) is not None
            and proof.enum_members is not None
            and proof.line < call.lineno
        )
    if binding == ("datetime", "timedelta"):
        return (
            len(call.args) <= 3
            and all(
                isinstance(arg, ast.Constant) and type(arg.value) in {int, float}
                for arg in call.args
            )
            and all(
                keyword.arg
                in {"days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks"}
                and isinstance(keyword.value, ast.Constant)
                and type(keyword.value.value) in {int, float}
                for keyword in call.keywords
            )
        )
    if binding == ("cayu", "RetryPolicy"):
        return not call.args and all(
            keyword.arg
            in {
                "max_attempts",
                "max_unknown_attempts",
                "initial_delay_s",
                "max_delay_s",
                "backoff_multiplier",
                "jitter_s",
                "retry_on_status_codes",
                "retry_on_timeout",
                "retry_on_connection_error",
                "retry_on_rate_limit",
            }
            and _literal_collection_is_data_only(keyword.value)
            for keyword in call.keywords
        )
    if binding not in {
        ("typing", "TypeVar"),
        ("typing", "ParamSpec"),
        ("typing", "TypeVarTuple"),
        ("contextvars", "ContextVar"),
    }:
        return False
    if (
        len(call.args) != 1
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
    ):
        return False
    options = {keyword.arg: keyword.value for keyword in call.keywords}
    if len(options) != len(call.keywords):
        return False
    if binding == ("contextvars", "ContextVar"):
        return set(options) <= {"default"} and all(
            _literal_collection_is_data_only(value) for value in options.values()
        )
    if "bound" in options and binding == ("typing", "TypeVar"):
        bound = options.pop("bound")
        if not _pydantic_annotation_is_declarative(
            bound,
            import_bindings=imports,
            rebound_names=rebound,
            project_root=root,
            declarative_bases=proofs,
        ):
            return False
    return set(options) <= (
        {"covariant", "contravariant", "infer_variance"} if binding[1] != "TypeVarTuple" else set()
    ) and all(
        isinstance(value, ast.Constant) and type(value.value) is bool for value in options.values()
    )


def _enum_value_is_declarative(
    expression: ast.expr,
    proofs: Mapping[str, _DeclarativeBinding],
) -> bool:
    member = expression
    if isinstance(member, ast.Attribute) and member.attr == "value":
        member = member.value
    if not isinstance(member, ast.Attribute) or not isinstance(member.value, ast.Name):
        return False
    proof = proofs.get(member.value.id)
    return (
        proof is not None
        and proof.line < expression.lineno
        and proof.enum_members is not None
        and member.attr in proof.enum_members
    )


def _bounded_integer_product(expression: ast.expr, depth: int = 0) -> int | None:
    if depth > 8:
        return None
    if isinstance(expression, ast.Constant) and type(expression.value) is int:
        return expression.value if abs(expression.value) <= 2**31 else None
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
        left = _bounded_integer_product(expression.left, depth + 1)
        right = _bounded_integer_product(expression.right, depth + 1)
        if left is not None and right is not None:
            value = left * right
            return value if abs(value) < 2**63 else None
    return None


def _declaration_value_is_inert(
    expression: ast.expr | None,
    proofs: Mapping[str, _DeclarativeBinding],
) -> bool:
    if _literal_collection_is_data_only(expression):
        return True
    if isinstance(expression, ast.Name):
        proof = proofs.get(expression.id)
        return proof is not None and proof.kind == "literal" and proof.line < expression.lineno
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
        return _bounded_integer_product(expression) is not None
    if isinstance(expression, ast.JoinedStr):
        return all(
            isinstance(item, ast.Constant)
            or (
                isinstance(item, ast.FormattedValue)
                and item.format_spec is None
                and _declaration_value_is_inert(item.value, proofs)
            )
            for item in expression.values
        )
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, (ast.UAdd, ast.USub)):
        return isinstance(expression.operand, ast.Constant) and type(expression.operand.value) in {
            int,
            float,
        }
    if isinstance(expression, ast.Attribute):
        return _enum_value_is_declarative(expression, proofs)
    if isinstance(expression, (ast.Tuple, ast.List, ast.Set)):
        return all(_declaration_value_is_inert(item, proofs) for item in expression.elts)
    if isinstance(expression, ast.Dict):
        return all(
            key is not None
            and _declaration_value_is_inert(key, proofs)
            and _declaration_value_is_inert(value, proofs)
            for key, value in zip(expression.keys, expression.values, strict=True)
        )
    return False


def _pydantic_field_argument_is_declarative(
    keyword: ast.keyword,
    imports: Mapping[str, tuple[str, str | None]],
    rebound: frozenset[str],
    proofs: Mapping[str, _DeclarativeBinding],
) -> bool:
    """Field stores a factory; it does not call it while declaring a model.

    Plain lambda/function bodies are deferred; lambda defaults are evaluated.
    Other references require an exact builtin or model proof. Arbitrary callable
    objects can expose effectful signature descriptors to Pydantic, so a mere
    name lookup is not sufficient evidence for those.
    """
    if keyword.arg is None:
        return False
    value = keyword.value
    if keyword.arg == "exclude_if" and isinstance(value, ast.Lambda):
        return (
            len(value.args.args) == 1
            and not value.args.posonlyargs
            and not value.args.kwonlyargs
            and not value.args.defaults
            and value.args.vararg is None
            and value.args.kwarg is None
            and (
                (
                    isinstance(value.body, ast.UnaryOp)
                    and isinstance(value.body.op, ast.Not)
                    and isinstance(value.body.operand, ast.Name)
                    and value.body.operand.id == value.args.args[0].arg
                )
                or (
                    isinstance(value.body, ast.Compare)
                    and len(value.body.ops) == 1
                    and isinstance(value.body.ops[0], (ast.Is, ast.IsNot))
                    and isinstance(value.body.left, ast.Name)
                    and value.body.left.id == value.args.args[0].arg
                    and _declaration_value_is_inert(
                        value.body.comparators[0],
                        {
                            name: proof
                            for name, proof in proofs.items()
                            if name != value.args.args[0].arg
                        },
                    )
                )
            )
        )
    if keyword.arg != "default_factory":
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "len"
        ):
            return (
                "len" not in imports
                and "len" not in rebound
                and len(value.args) == 1
                and not value.keywords
                and isinstance(value.args[0], ast.Name)
                and (proof := proofs.get(value.args[0].id)) is not None
                and proof.enum_members is not None
                and proof.line < value.lineno
            )
        return _declaration_value_is_inert(value, proofs)
    if isinstance(value, ast.Name):
        proof = proofs.get(value.id)
        if proof is not None and proof.kind in {"model", "function"} and proof.line < value.lineno:
            return True
        return (
            value.id in {"list", "dict", "tuple", "set", "frozenset"}
            and value.id not in imports
            and value.id not in rebound
        )
    if isinstance(value, ast.Lambda):
        return (
            not value.args.posonlyargs
            and not value.args.args
            and not value.args.kwonlyargs
            and value.args.vararg is None
            and value.args.kwarg is None
            and not value.args.defaults
            and not value.args.kw_defaults
        )
    return isinstance(value, ast.Constant) and value.value is None


def _unsafe_import_time_expression(
    node: ast.stmt,
    *,
    import_bindings: Mapping[str, tuple[str, str | None]],
    rebound_names: frozenset[str],
    declarative_identities: frozenset[str],
    declarative_subscriptions: frozenset[str],
    project_root: Path,
    declarative_bases: Mapping[str, _DeclarativeBinding],
) -> ast.expr | ast.stmt | None:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return (
            None
            if _private_collection_update_is_inert(
                node.value,
                declarative_collections=declarative_subscriptions,
            )
            else node.value
        )
    declarative_values: set[ast.expr] = set()
    for decorator, class_rebound_names in _import_time_decorator_uses(node):
        decorator_rebound_names = rebound_names | class_rebound_names
        if _trusted_symbol(decorator, import_bindings, decorator_rebound_names, project_root) in {
            ("contextlib", "contextmanager"),
            ("contextlib", "asynccontextmanager"),
        } and any(
            isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
            and decorator in definition.decorator_list
            for definition in _import_time_definition_nodes(node)
        ):
            continue
        if _trusted_symbol(decorator, import_bindings, decorator_rebound_names, project_root) == (
            "pydantic",
            "computed_field",
        ) and any(
            isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
            and decorator in definition.decorator_list
            and definition.returns is not None
            and _pydantic_annotation_is_declarative(
                definition.returns,
                import_bindings=import_bindings,
                rebound_names=decorator_rebound_names,
                project_root=project_root,
                declarative_bases={
                    name: proof
                    for name, proof in declarative_bases.items()
                    if name not in class_rebound_names
                },
            )
            for definition in _import_time_definition_nodes(node)
        ):
            continue
        if isinstance(decorator, ast.Call) and _pydantic_validator_is_declarative(
            decorator,
            import_bindings=import_bindings,
            rebound_names=decorator_rebound_names,
            project_root=project_root,
        ):
            declarative_values.add(decorator)
            continue
        for call in _evaluated_calls(decorator):
            if not _import_call_is_declarative(
                call,
                import_bindings=import_bindings,
                rebound_names=decorator_rebound_names,
                declarative_identities=declarative_identities,
            ):
                return call
        if isinstance(decorator, ast.Call):
            continue
        if not _bare_decorator_is_declarative(
            decorator,
            import_bindings=import_bindings,
            rebound_names=decorator_rebound_names,
        ):
            return decorator
    for metaclass in _import_time_metaclasses(node):
        return metaclass
    for base, class_rebound_names in _import_time_class_base_uses(node):
        if not _class_base_is_declarative(
            base,
            project_root=project_root,
            import_bindings=import_bindings,
            rebound_names=rebound_names | class_rebound_names,
            declarative_bases=declarative_bases,
            class_rebound_names=class_rebound_names,
        ):
            return base
    # A newly admitted local base must not allow an opaque descriptor into the
    # subclass namespace: type.__new__ would execute its __set_name__ hook.
    for definition in _import_time_class_nodes(node):
        if (
            any(
                _trusted_symbol(base, import_bindings, rebound_names, project_root)
                in {("enum", "Enum"), ("enum", "IntEnum"), ("enum", "StrEnum")}
                for base in definition.bases
            )
            and _enum_declaration_members(definition, import_bindings, rebound_names, project_root)
            is None
        ):
            return definition
        if any(
            _is_pydantic_base(base, import_bindings, declarative_bases) for base in definition.bases
        ):
            # An ordinary mixin may carry schema hooks that Pydantic activates.
            # Only model-proven parents can participate in this bounded proof.
            for base in definition.bases:
                if not _is_pydantic_base(base, import_bindings, declarative_bases):
                    return base
            class_bindings = _scope_rebound_names(definition.body, include_imports=True)
            class_scope = rebound_names | class_bindings
            class_proofs = {
                name: proof
                for name, proof in declarative_bases.items()
                if name not in class_bindings
            }
            for statement in definition.body:
                if (
                    isinstance(statement, (ast.Assign, ast.AnnAssign))
                    and statement.value is not None
                    and _declaration_value_is_inert(statement.value, class_proofs)
                ):
                    declarative_values.add(statement.value)
                if isinstance(statement, ast.AnnAssign) and not _pydantic_annotation_is_declarative(
                    statement.annotation,
                    import_bindings=import_bindings,
                    rebound_names=class_scope,
                    project_root=project_root,
                    declarative_bases=class_proofs,
                ):
                    return statement.annotation
                if isinstance(statement, (ast.Assign, ast.AnnAssign)) and isinstance(
                    statement.value, ast.Call
                ):
                    call = statement.value
                    if (
                        isinstance(call.func, ast.Name)
                        and call.func.id not in class_scope
                        and import_bindings.get(call.func.id)
                        in {("pydantic", "ConfigDict"), ("pydantic", "Field")}
                        and not _import_binding_is_project_local(
                            project_root, import_bindings[call.func.id]
                        )
                        and all(_literal_collection_is_data_only(arg) for arg in call.args)
                        and all(
                            (
                                _pydantic_field_argument_is_declarative(
                                    keyword,
                                    import_bindings,
                                    class_scope,
                                    class_proofs,
                                )
                                if import_bindings.get(call.func.id) == ("pydantic", "Field")
                                else keyword.arg is not None
                                and _literal_collection_is_data_only(keyword.value)
                            )
                            for keyword in call.keywords
                        )
                    ):
                        declarative_values.add(call)
            # Pydantic evaluates explicit annotation dictionaries and invokes
            # Python 3.14 annotation hooks and schema hooks during construction.
            for statement in definition.body:
                if _scope_rebound_names([statement], include_imports=True) & {
                    "__annotations__",
                    "__annotate__",
                    "__annotate_func__",
                    "__get_pydantic_core_schema__",
                    "__get_pydantic_json_schema__",
                    "__pydantic_init_subclass__",
                    "__pydantic_on_complete__",
                }:
                    return statement
        if any(
            isinstance(base, ast.Name) and base.id in declarative_bases for base in definition.bases
        ):
            # Multiple inheritance can introduce an effectful metaclass even
            # when the application mixin itself has an ordinary, inert type.
            for base in definition.bases:
                if not _class_base_is_declarative(
                    base,
                    project_root=project_root,
                    import_bindings=import_bindings,
                    rebound_names=rebound_names,
                    declarative_bases=declarative_bases,
                    require_inert_metaclass=True,
                ):
                    return base
            if definition.keywords:
                return definition.keywords[0].value
    # type.__new__ invokes descriptors even for a leaf class with no local
    # subclasses. Every newly admitted namespace needs the same value proof.
    for definition in _import_time_class_nodes(node):
        class_bindings = _scope_rebound_names(definition.body, include_imports=True)
        class_proofs = {
            name: proof for name, proof in declarative_bases.items() if name not in class_bindings
        }
        values = declarative_values | {
            statement.value
            for statement in definition.body
            if isinstance(statement, (ast.Assign, ast.AnnAssign))
            and statement.value is not None
            and (
                _declaration_value_is_inert(statement.value, class_proofs)
                or _runtime_namespace_value_is_inert(
                    statement.value, import_bindings, rebound_names | class_bindings, project_root
                )
            )
        }
        unsafe_namespace = _unsafe_class_namespace(definition, declarative_values=values)
        if unsafe_namespace is not None:
            return unsafe_namespace
    for expression, class_rebound_names in _import_time_expression_uses(node):
        expression_rebound_names = rebound_names | class_rebound_names
        expression_proofs = {
            name: proof
            for name, proof in declarative_bases.items()
            if name not in class_rebound_names
        }
        if _pydantic_annotation_is_declarative(
            expression,
            import_bindings=import_bindings,
            rebound_names=expression_rebound_names,
            project_root=project_root,
            declarative_bases=expression_proofs,
        ):
            declarative_values.update(_evaluated_calls(expression))
        for mutation_target in ast.walk(expression):
            if isinstance(mutation_target, (ast.Attribute, ast.Subscript)) and isinstance(
                mutation_target.ctx, (ast.Store, ast.Del)
            ):
                return mutation_target
        for attribute in _evaluated_attributes(expression):
            if _enum_value_is_declarative(
                attribute,
                {
                    name: proof
                    for name, proof in declarative_bases.items()
                    if name not in class_rebound_names
                },
            ):
                continue
            root_name = _expression_root_name(attribute)
            if root_name in expression_rebound_names and root_name not in declarative_identities:
                return attribute
            binding = import_bindings.get(root_name) if root_name is not None else None
            if binding is not None and _import_binding_is_project_local(project_root, binding):
                return attribute
        for subscript in _evaluated_subscripts(expression):
            if _trusted_symbol(
                subscript.value, import_bindings, expression_rebound_names, project_root
            ) in {
                ("contextvars", "ContextVar"),
                ("asyncio", "Future"),
            } and _pydantic_annotation_is_declarative(
                subscript.slice,
                import_bindings=import_bindings,
                rebound_names=expression_rebound_names,
                project_root=project_root,
            ):
                continue
            root_name = _expression_root_name(subscript.value)
            binding = import_bindings.get(root_name) if root_name is not None else None
            if root_name is not None and (
                (
                    root_name in class_rebound_names
                    or (root_name in rebound_names and root_name not in declarative_subscriptions)
                )
                or (
                    binding is not None
                    and not _imported_subscription_is_declarative(project_root, binding)
                )
            ):
                return subscript
        for call in _evaluated_calls(expression):
            if call in declarative_values or _stdlib_declaration_is_inert(
                call, import_bindings, expression_rebound_names, project_root, expression_proofs
            ):
                continue
            if not _import_call_is_declarative(
                call,
                import_bindings=import_bindings,
                rebound_names=expression_rebound_names,
                declarative_identities=declarative_identities,
            ):
                return call
    return None


def _pydantic_annotation_is_declarative(
    annotation: ast.expr,
    *,
    import_bindings: Mapping[str, tuple[str, str | None]],
    rebound_names: frozenset[str],
    project_root: Path,
    declarative_bases: Mapping[str, _DeclarativeBinding] = {},
) -> bool:
    """Model construction evaluates annotations and invokes custom schema hooks."""

    def safe(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return node.value is None or node.value is Ellipsis
        if isinstance(node, ast.Name):
            proof = declarative_bases.get(node.id)
            if proof is not None and proof.line < node.lineno:
                return proof.kind in {"model", "enum", "annotation"}
        if isinstance(node, ast.Name) and node.id not in rebound_names:
            binding = import_bindings.get(node.id)
            if binding is None:
                return node.id in {
                    "str",
                    "int",
                    "float",
                    "bool",
                    "bytes",
                    "list",
                    "dict",
                    "tuple",
                    "set",
                    "frozenset",
                }
            if _import_binding_is_project_local(project_root, binding):
                return False
            module, name = binding
            return (
                binding
                in {
                    ("pathlib", "Path"),
                    ("pathlib", "PurePosixPath"),
                    ("datetime", "datetime"),
                    ("datetime", "date"),
                    ("datetime", "timedelta"),
                    ("pydantic", "BaseModel"),
                    ("pydantic", "StrictInt"),
                    ("pydantic", "StrictStr"),
                    ("pydantic", "StrictBool"),
                    ("pydantic", "StrictFloat"),
                    ("pydantic", "StrictBytes"),
                }
                or module == "cayu"
                or module.startswith("cayu.")
                or (
                    module == "typing"
                    and name in {"Any", "Optional", "Union", "Literal", "ClassVar", "Annotated"}
                )
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return safe(node.left) and safe(node.right)
        if isinstance(node, ast.Subscript):
            arguments = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if isinstance(node.value, ast.Name) and import_bindings.get(node.value.id) == (
                "typing",
                "Literal",
            ):
                return safe(node.value) and all(
                    isinstance(arg, ast.Constant)
                    or _enum_value_is_declarative(arg, declarative_bases)
                    for arg in arguments
                )
            if _trusted_symbol(node.value, import_bindings, rebound_names, project_root) == (
                "typing",
                "Annotated",
            ):
                return (
                    len(arguments) >= 2
                    and safe(arguments[0])
                    and all(
                        isinstance(arg, ast.Call)
                        and _trusted_symbol(arg.func, import_bindings, rebound_names, project_root)
                        == ("pydantic", "Field")
                        and all(
                            _declaration_value_is_inert(item, declarative_bases)
                            for item in arg.args
                        )
                        and all(
                            _pydantic_field_argument_is_declarative(
                                keyword, import_bindings, rebound_names, declarative_bases
                            )
                            for keyword in arg.keywords
                        )
                        for arg in arguments[1:]
                    )
                )
            return safe(node.value) and all(safe(arg) for arg in arguments)
        return False

    return safe(annotation)


def _pydantic_validator_is_declarative(
    call: ast.Call,
    *,
    import_bindings: Mapping[str, tuple[str, str | None]],
    rebound_names: frozenset[str],
    project_root: Path,
) -> bool:
    """Recognize registration only, with literal options and verified import identity."""
    if isinstance(call.func, ast.Name):
        root = call.func.id
        module, helper = import_bindings.get(root, ("", None))
    elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        root = call.func.value.id
        module, imported = import_bindings.get(root, ("", None))
        if imported is not None:
            return False
        helper = call.func.attr
    else:
        return False
    if (
        root in rebound_names
        or module != "pydantic"
        or helper not in {"field_validator", "model_validator"}
        or _import_binding_is_project_local(project_root, (module, helper))
    ):
        return False
    # No unpacking, lookups, operators or calls: even an apparently innocuous
    # option can invoke application code via iteration or a descriptor.
    if any(
        not isinstance(arg, ast.Constant) or not isinstance(arg.value, str) for arg in call.args
    ):
        return False
    if (helper == "field_validator" and not call.args) or (
        helper == "model_validator" and call.args
    ):
        return False
    options: dict[str, object] = {}
    for keyword in call.keywords:
        if (
            keyword.arg is None
            or keyword.arg in options
            or not isinstance(keyword.value, ast.Constant)
        ):
            return False
        options[keyword.arg] = keyword.value.value
    modes = {"before", "after", "wrap"}
    if helper == "field_validator":
        modes.add("plain")
        if set(options) - {"mode", "check_fields"}:
            return False
        if type(options.get("check_fields")) not in {bool, type(None)}:
            return False
    elif set(options) != {"mode"}:
        return False
    mode = options.get("mode", "after")
    return isinstance(mode, str) and mode in modes


def _bare_decorator_is_declarative(
    decorator: ast.expr,
    *,
    import_bindings: Mapping[str, tuple[str, str | None]],
    rebound_names: frozenset[str],
) -> bool:
    if not isinstance(decorator, ast.Name) or decorator.id in rebound_names:
        return False
    name = decorator.id
    if name in {"classmethod", "property", "staticmethod"}:
        return name not in import_bindings
    expected_imports = {
        "abstractmethod": ("abc", "abstractmethod"),
        "cached_property": ("functools", "cached_property"),
        "dataclass": ("dataclasses", "dataclass"),
        "final": ("typing", "final"),
        "overload": ("typing", "overload"),
        "runtime_checkable": ("typing", "runtime_checkable"),
    }
    return name in expected_imports and import_bindings.get(name) == expected_imports[name]


def _expression_root_name(expression: ast.expr) -> str | None:
    while isinstance(expression, (ast.Attribute, ast.Subscript)):
        expression = expression.value
    return expression.id if isinstance(expression, ast.Name) else None


# These builtins have no application-owned metaclass or subclass hook. Unknown
# unbound names are not evidence of safety (nor are calls or local subscriptions).
_INERT_BUILTIN_BASES = frozenset(
    {"object"}
    | {
        name
        for name, value in vars(builtins).items()
        if isinstance(value, type) and issubclass(value, BaseException)
    }
)


# Permission to declare a class using an external base does not establish that
# its metaclass can safely create further application subclasses. EnumType, for
# example, calls inherited __init__/__new__ methods while creating members.
# Only these reviewed external roots propagate a local inheritance proof.
# BaseModel additionally requires the model annotation/namespace/hook checks at
# every generation; _DeclarativeBinding carries that obligation through imports.
_INERT_METACLASS_BASE_IMPORTS = frozenset(
    {
        ("abc", "ABC"),
        ("cayu", "Tool"),
        ("cayu.core.tools", "Tool"),
        ("pydantic", "BaseModel"),
        ("http.server", "ThreadingHTTPServer"),
        ("http.server", "BaseHTTPRequestHandler"),
    }
)


def _class_base_is_declarative(
    base: ast.expr,
    *,
    project_root: Path,
    import_bindings: Mapping[str, tuple[str, str | None]],
    rebound_names: frozenset[str],
    declarative_bases: Mapping[str, _DeclarativeBinding],
    class_rebound_names: frozenset[str] = frozenset(),
    require_inert_metaclass: bool = False,
) -> bool:
    if isinstance(base, ast.Name) and base.id not in class_rebound_names:
        proof = declarative_bases.get(base.id)
        if proof is not None and proof.line < base.lineno:
            return proof.kind in {"class", "model"}
    name = _expression_root_name(base)
    if name is None or name in rebound_names:
        return False
    binding = import_bindings.get(name)
    if binding is not None:
        # Only direct imported names enter the model safeguards above. An
        # attribute or subscription rooted at BaseModel is not the same proof.
        if binding == ("pydantic", "BaseModel") and not isinstance(base, ast.Name):
            return False
        if require_inert_metaclass and (
            not isinstance(base, ast.Name) or binding not in _INERT_METACLASS_BASE_IMPORTS
        ):
            return False
        return _imported_class_base_is_declarative(project_root, binding)
    return isinstance(base, ast.Name) and name in _INERT_BUILTIN_BASES


def _runtime_namespace_value_is_inert(
    value: ast.expr,
    imports: Mapping[str, tuple[str, str | None]],
    rebound: frozenset[str],
    root: Path,
) -> bool:
    """Reviewed immutable Runtime metadata cannot supply descriptor hooks."""
    if isinstance(value, ast.Attribute):
        return _trusted_symbol(value.value, imports, rebound, root) == (
            "cayu.server",
            "ServiceIdentityStoreKind",
        ) and value.attr in {"DEVELOPMENT", "DURABLE"}
    return (
        isinstance(value, ast.Call)
        and _trusted_symbol(value.func, imports, rebound, root)
        in {
            ("cayu", "ToolSpec"),
            ("cayu", "ExecutionProfileBehaviorIdentity"),
        }
        and all(_literal_collection_is_data_only(arg) for arg in value.args)
        and all(
            keyword.arg is not None and _literal_collection_is_data_only(keyword.value)
            for keyword in value.keywords
        )
    )


def _unsafe_class_namespace(
    node: ast.ClassDef,
    *,
    declarative_values: set[ast.expr] | frozenset[ast.expr] = frozenset(),
) -> ast.expr | ast.stmt | None:
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Pass)):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value = statement.value
            if (
                value is None
                or _literal_collection_is_data_only(value)
                or value in declarative_values
            ):
                continue
            return value
        # Conditional/dynamic namespaces need a richer proof than this bounded
        # local inheritance analysis.
        return statement
    return None


@dataclass(frozen=True)
class _DeclarationGraph:
    proven: dict[Path, dict[str, _DeclarativeBinding]]
    structural: dict[Path, dict[str, _DeclarativeBinding]]
    failures: dict[Path, tuple[ast.expr | ast.stmt, ...]]
    blockers: dict[Path, frozenset[Path]]


def _project_declaration_proofs(root: Path) -> _DeclarationGraph:
    """Prove ordered bindings to a fixed point, then invalidate unsafe closures.

    No application code is imported. Cyclic imports can re-export already proven
    identities; cyclic inheritance cannot bootstrap a proof. State belongs to
    one scan, so subsequent scans cannot reuse stale source or trust decisions.
    """
    trees: dict[Path, ast.Module] = {}
    dependencies: dict[Path, set[Path]] = {}
    invalid: set[Path] = set()
    invalid_imports: dict[Path, list[ast.stmt]] = {}
    pending = list(_application_module_paths(root))
    while pending:
        path = pending.pop()
        if path in trees or path in invalid:
            continue
        if len(trees.keys() | invalid) >= _MAX_DECLARATION_MODULES:
            raise _DeclarationGraphLimit
        if any(part.is_symlink() for part in (path, *path.parents) if part != root):
            invalid.add(path)
            continue
        # A namespace directory contributes no executable source, but its
        # explicit children and regular parent initializers still enter the graph.
        tree = ast.Module(body=[], type_ignores=[]) if path.is_dir() else _parsed_module(path)
        if tree is None:
            invalid.add(path)
            continue
        trees[path] = tree
        edges = dependencies[path] = set()
        for parent in path.parents:
            if parent == root:
                break
            initializer = parent / "__init__.py"
            if initializer.exists() or initializer.is_symlink():
                edges.add(initializer)
        for statement in tree.body:
            for imported in _import_time_import_nodes(statement):
                modules = (
                    [(alias.name, 0) for alias in imported.names]
                    if isinstance(imported, ast.Import)
                    else [(imported.module, imported.level)]
                )
                for module, level in modules:
                    local, target = _project_local_import_module_path(
                        root, importing_path=path, module=module, level=level
                    )
                    if target is not None:
                        edges.add(target)
                    elif local:
                        invalid.add(path)
                        invalid_imports.setdefault(path, []).append(imported)
                    if isinstance(imported, ast.ImportFrom):
                        for alias in imported.names:
                            child_module = f"{module}.{alias.name}" if module else alias.name
                            _local, child = _project_local_import_module_path(
                                root, importing_path=path, module=child_module, level=level
                            )
                            if child is not None:
                                edges.add(child)
        pending.extend(edges - trees.keys() - invalid)

    imports = {path: _import_bindings(tree) for path, tree in trees.items()}
    rebounds = {path: _module_rebound_names(tree) for path, tree in trees.items()}
    unique: dict[Path, set[str]] = {}
    for path, tree in trees.items():
        counts: dict[str, int] = {}
        for statement in tree.body:
            names = set(_scope_rebound_names([statement], include_imports=True))
            for expression, _scope in _import_time_expression_uses(statement):
                names.update(
                    item.target.id
                    for item in ast.walk(expression)
                    if isinstance(item, ast.NamedExpr) and isinstance(item.target, ast.Name)
                )
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        unique[path] = {name for name, count in counts.items() if count == 1}
        if any(
            alias.name == "*"
            for statement in tree.body
            for imported in _import_time_import_from_nodes(statement)
            for alias in imported.names
        ):
            unique[path] = set()
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                names = [alias.asname or alias.name for alias in statement.names]
                unique[path].difference_update(name for name in names if names.count(name) > 1)

    proofs: dict[Path, dict[str, _DeclarativeBinding]] = {path: {} for path in trees}
    inputs: dict[Path, tuple[tuple[Path, tuple[tuple[str, _DeclarativeBinding], ...]], ...]] = {}
    # Every successful round establishes at least one source binding. The bound
    # is derived from this finite graph, not from a timeout or optimistic cycle.
    for _ in range(1 + sum(len(names) for names in unique.values())):
        changed = False
        for path, tree in trees.items():
            incoming = tuple(
                (target, tuple(proofs.get(target, {}).items()))
                for target in sorted(dependencies[path])
                if target != path
            )
            if path in inputs and inputs[path] == incoming:
                continue
            inputs[path] = incoming
            proven: dict[str, _DeclarativeBinding] = {}
            for node in tree.body:
                if isinstance(node, ast.ImportFrom):
                    _local, target = _project_local_import_module_path(
                        root, importing_path=path, module=node.module, level=node.level
                    )
                    if target is None:
                        continue
                    imported_proofs = proven if target == path else proofs.get(target, {})
                    for alias in node.names:
                        name = alias.asname or alias.name
                        proof = imported_proofs.get(alias.name)
                        if name in unique[path] and proof is not None:
                            proven[name] = replace(proof, line=node.lineno)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    target = (
                        node.targets[0]
                        if isinstance(node, ast.Assign) and len(node.targets) == 1
                        else (node.target if isinstance(node, ast.AnnAssign) else None)
                    )
                    if (
                        isinstance(target, ast.Name)
                        and target.id in unique[path]
                        and node.value is not None
                    ):
                        value = node.value
                        # Mutable collections are never exported as trusted constants.
                        if isinstance(
                            value, (ast.Constant, ast.Tuple, ast.BinOp, ast.UnaryOp)
                        ) and _declaration_value_is_inert(value, proven):
                            if not any(
                                isinstance(item, (ast.List, ast.Dict, ast.Set))
                                for item in ast.walk(value)
                            ):
                                proven[target.id] = _DeclarativeBinding(node.lineno, kind="literal")
                        elif isinstance(
                            value, (ast.Subscript, ast.BinOp)
                        ) and _pydantic_annotation_is_declarative(
                            value,
                            import_bindings=imports[path],
                            rebound_names=rebounds[path],
                            project_root=root,
                            declarative_bases=proven,
                        ):
                            proven[target.id] = _DeclarativeBinding(node.lineno, kind="annotation")
                elif isinstance(node, ast.FunctionDef) and node.name in unique[path]:
                    if (
                        not node.decorator_list
                        and _unsafe_import_time_expression(
                            node,
                            import_bindings=imports[path],
                            rebound_names=rebounds[path],
                            declarative_identities=frozenset(),
                            declarative_subscriptions=frozenset(),
                            project_root=root,
                            declarative_bases=proven,
                        )
                        is None
                    ):
                        proven[node.name] = _DeclarativeBinding(node.lineno, kind="function")
                elif isinstance(node, ast.ClassDef) and node.name in unique[path]:
                    proof = _class_declaration_proof(
                        root, node, imports[path], rebounds[path], proven
                    )
                    if proof is not None:
                        proven[node.name] = proof
            if proven != proofs[path]:
                proofs[path] = proven
                changed = True
        if not changed:
            break
    else:
        raise _DeclarationGraphLimit

    failures = {
        path: failed
        for path, tree in trees.items()
        if (failed := _module_declaration_failures(root, path, tree, proofs[path]))
    }
    for path, imports in invalid_imports.items():
        failures[path] = tuple(dict.fromkeys((*failures.get(path, ()), *imports)))
    invalid.update(failures)
    blockers = {path: {path} if path in invalid else set() for path in trees.keys() | invalid}
    while True:
        changed = False
        for path, edges in dependencies.items():
            inherited = set().union(*(blockers.get(edge, {edge}) for edge in edges))
            if not inherited <= blockers[path]:
                blockers[path].update(inherited)
                changed = True
        if not changed:
            break
    return _DeclarationGraph(
        {path: proof for path, proof in proofs.items() if not blockers[path]},
        proofs,
        failures,
        {path: frozenset(causes) for path, causes in blockers.items() if causes},
    )


def _class_declaration_proof(
    root: Path,
    node: ast.ClassDef,
    imports: Mapping[str, tuple[str, str | None]],
    rebound: frozenset[str],
    proven: Mapping[str, _DeclarativeBinding],
) -> _DeclarativeBinding | None:
    if node.decorator_list or node.keywords:
        return None
    if any(_import_time_statement_may_bind_name(item, "__init_subclass__") for item in node.body):
        return None
    enum_members = _enum_declaration_members(node, imports, rebound, root)
    pydantic = any(_is_pydantic_base(base, imports, proven) for base in node.bases)
    if enum_members is None and not all(
        _class_base_is_declarative(
            base,
            project_root=root,
            import_bindings=imports,
            rebound_names=rebound,
            declarative_bases=proven,
            require_inert_metaclass=True,
        )
        for base in node.bases
    ):
        return None
    if (
        _unsafe_import_time_expression(
            node,
            import_bindings=imports,
            rebound_names=rebound,
            declarative_identities=frozenset(),
            declarative_subscriptions=frozenset(),
            project_root=root,
            declarative_bases=proven,
        )
        is not None
    ):
        return None
    return _DeclarativeBinding(
        node.lineno,
        "model" if pydantic else "enum" if enum_members is not None else "class",
        enum_members,
    )


def _enum_declaration_members(
    node: ast.ClassDef,
    imports: Mapping[str, tuple[str, str | None]],
    rebound: frozenset[str],
    root: Path,
) -> frozenset[str] | None:
    if len(node.bases) != 1 or _trusted_symbol(node.bases[0], imports, rebound, root) not in {
        ("enum", "Enum"),
        ("enum", "StrEnum"),
        ("enum", "IntEnum"),
    }:
        return None
    members: set[str] = set()
    for statement in node.body:
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
        ):
            continue
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and not statement.targets[0].id.startswith("_")
            and isinstance(statement.value, ast.Constant)
            and type(statement.value.value) in {str, int}
        ):
            return None
        members.add(statement.targets[0].id)
    return frozenset(members)


def _module_declaration_failures(
    root: Path,
    path: Path,
    tree: ast.Module,
    bases: Mapping[str, _DeclarativeBinding],
) -> tuple[ast.expr | ast.stmt, ...]:
    bindings = _import_bindings(tree)
    rebound = _module_rebound_names(tree)
    identities = _declarative_identity_names(tree, import_bindings=bindings, rebound_names=rebound)
    subscriptions = _declarative_subscription_names(root, tree)
    failures: list[ast.expr | ast.stmt] = []
    for statement in tree.body:
        if _import_time_statement_may_bind_name(statement, "__getattr__"):
            failures.append(statement)
            continue
        unsafe: ast.expr | ast.stmt | None = next(
            (
                imported
                for imported in _import_time_import_from_nodes(statement)
                if not _project_local_import_from_is_declarative(
                    root, importing_path=path, node=imported
                )
            ),
            None,
        )
        if unsafe is None:
            unsafe = _unsafe_import_time_expression(
                statement,
                import_bindings=bindings,
                rebound_names=rebound,
                declarative_identities=identities,
                declarative_subscriptions=subscriptions,
                project_root=root,
                declarative_bases=bases,
            )
        if unsafe is not None:
            failures.append(unsafe)
    return tuple(failures)


def _imported_class_base_is_declarative(
    root: Path,
    binding: tuple[str, str | None],
) -> bool:
    module = binding[0]
    if _import_binding_is_project_local(root, binding):
        return False
    if module == "cayu" or module.startswith("cayu."):
        return True
    return binding in {
        ("abc", "ABC"),
        ("http.server", "ThreadingHTTPServer"),
        ("http.server", "BaseHTTPRequestHandler"),
        ("pydantic", "BaseModel"),
        ("enum", "Enum"),
        ("enum", "IntEnum"),
        ("enum", "StrEnum"),
        ("typing", "Protocol"),
        ("typing_extensions", "Protocol"),
    }


def _import_binding_is_project_local(
    root: Path,
    binding: tuple[str, str | None],
) -> bool:
    module = binding[0]
    if module.startswith("."):
        return True
    parts = module.split(".")
    return any(
        candidate.is_dir()
        or candidate.is_symlink()
        or candidate.with_suffix(".py").is_file()
        or candidate.with_suffix(".py").is_symlink()
        for length in range(1, len(parts) + 1)
        for candidate in (root.joinpath(*parts[:length]),)
    )


def _imported_subscription_is_declarative(
    root: Path,
    binding: tuple[str, str | None],
) -> bool:
    if not _imported_class_base_is_declarative(root, binding):
        module = binding[0]
        return module in {
            "collections",
            "collections.abc",
            "os",
            "re",
            "subprocess",
            "typing",
        } or module.startswith("typing.")
    return True


def _declarative_subscription_names(root: Path, tree: ast.Module) -> frozenset[str]:
    import_bindings = _import_bindings(tree)
    rebound_names = _module_rebound_names(tree)
    candidates: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        else:
            continue
        literal_initializer = isinstance(
            value, (ast.Constant, ast.Dict, ast.List, ast.Set, ast.Tuple)
        )
        copied_literal_export = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "list"
            and "list" not in import_bindings
            and "list" not in rebound_names
            and len(value.args) == 1
            and not value.keywords
            and isinstance(value.args[0], ast.Name)
            and _imported_literal_collection_export(root, import_bindings.get(value.args[0].id))
        )
        if isinstance(target, ast.Name) and (literal_initializer or copied_literal_export):
            candidates.add(target.id)
    store_counts = {
        name: sum(
            isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id == name
            for item in ast.walk(tree)
        )
        for name in candidates
    }
    other_bindings = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    other_bindings.update(_import_bindings(tree))
    return frozenset(
        name for name in candidates if store_counts[name] == 1 and name not in other_bindings
    )


def _imported_literal_collection_export(
    root: Path,
    binding: tuple[str, str | None] | None,
) -> bool:
    if binding is None or binding[1] is None or binding[0].startswith("."):
        return False
    path = root.joinpath(*binding[0].split(".")).with_suffix(".py")
    if not path.is_file() or path.is_symlink():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return False
    symbol = binding[1]
    definitions: list[ast.expr | None] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == symbol for target in node.targets):
                definitions.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == symbol:
                definitions.append(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                definitions.append(None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for imported in node.names:
                if (imported.asname or imported.name.split(".")[0]) == symbol:
                    definitions.append(None)
    return len(definitions) == 1 and _literal_collection_is_data_only(definitions[0])


def _literal_collection_is_data_only(expression: ast.expr | None) -> bool:
    if isinstance(expression, ast.Constant):
        return True
    if isinstance(expression, (ast.List, ast.Set, ast.Tuple)):
        return all(_literal_collection_is_data_only(item) for item in expression.elts)
    if isinstance(expression, ast.Dict):
        return all(
            key is not None
            and _literal_collection_is_data_only(key)
            and _literal_collection_is_data_only(value)
            for key, value in zip(expression.keys, expression.values, strict=True)
        )
    return False


def _import_time_definition_nodes(
    node: ast.stmt,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, ...]:
    definitions: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = []

    def add_suite(statements: list[ast.stmt]) -> None:
        for statement in statements:
            definitions.extend(_import_time_definition_nodes(statement))

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        definitions.append(node)
    elif isinstance(node, ast.ClassDef):
        definitions.append(node)
        add_suite(node.body)
    elif isinstance(node, ast.If):
        add_suite(node.orelse if _is_main_guard(node.test) else node.body + node.orelse)
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        add_suite(node.body)
        add_suite(node.orelse)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        add_suite(node.body)
    elif isinstance(node, (ast.Try, ast.TryStar)):
        add_suite(node.body)
        for handler in node.handlers:
            add_suite(handler.body)
        add_suite(node.orelse)
        add_suite(node.finalbody)
    elif isinstance(node, ast.Match):
        for case in node.cases:
            add_suite(case.body)
    return tuple(definitions)


def _import_time_class_nodes(node: ast.stmt) -> tuple[ast.ClassDef, ...]:
    return tuple(
        definition
        for definition in _import_time_definition_nodes(node)
        if isinstance(definition, ast.ClassDef)
    )


def _import_time_decorator_uses(
    node: ast.stmt,
    *,
    class_rebound_names: frozenset[str] = frozenset(),
) -> tuple[tuple[ast.expr, frozenset[str]], ...]:
    uses: list[tuple[ast.expr, frozenset[str]]] = []

    def add_suite(statements: list[ast.stmt], scope: frozenset[str]) -> None:
        for statement in statements:
            uses.extend(
                _import_time_decorator_uses(
                    statement,
                    class_rebound_names=scope,
                )
            )

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        uses.extend((decorator, class_rebound_names) for decorator in node.decorator_list)
    elif isinstance(node, ast.ClassDef):
        uses.extend((decorator, class_rebound_names) for decorator in node.decorator_list)
        class_scope = _scope_rebound_names(node.body, include_imports=True)
        add_suite(node.body, class_scope)
    elif isinstance(node, ast.If):
        add_suite(
            node.orelse if _is_main_guard(node.test) else node.body + node.orelse,
            class_rebound_names,
        )
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        add_suite(node.body, class_rebound_names)
        add_suite(node.orelse, class_rebound_names)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        add_suite(node.body, class_rebound_names)
    elif isinstance(node, (ast.Try, ast.TryStar)):
        add_suite(node.body, class_rebound_names)
        for handler in node.handlers:
            add_suite(handler.body, class_rebound_names)
        add_suite(node.orelse, class_rebound_names)
        add_suite(node.finalbody, class_rebound_names)
    elif isinstance(node, ast.Match):
        for case in node.cases:
            add_suite(case.body, class_rebound_names)
    return tuple(uses)


def _import_time_metaclasses(node: ast.stmt) -> tuple[ast.expr, ...]:
    return tuple(
        keyword.value
        for class_node in _import_time_class_nodes(node)
        for keyword in class_node.keywords
        if keyword.arg == "metaclass"
    )


def _import_time_class_base_uses(
    node: ast.stmt,
    *,
    class_rebound_names: frozenset[str] = frozenset(),
) -> tuple[tuple[ast.expr, frozenset[str]], ...]:
    uses: list[tuple[ast.expr, frozenset[str]]] = []

    def add_suite(statements: list[ast.stmt], scope: frozenset[str]) -> None:
        for statement in statements:
            uses.extend(
                _import_time_class_base_uses(
                    statement,
                    class_rebound_names=scope,
                )
            )

    if isinstance(node, ast.ClassDef):
        uses.extend((base, class_rebound_names) for base in node.bases)
        add_suite(node.body, _scope_rebound_names(node.body, include_imports=True))
    elif isinstance(node, ast.If):
        add_suite(
            node.orelse if _is_main_guard(node.test) else node.body + node.orelse,
            class_rebound_names,
        )
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        add_suite(node.body, class_rebound_names)
        add_suite(node.orelse, class_rebound_names)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        add_suite(node.body, class_rebound_names)
    elif isinstance(node, (ast.Try, ast.TryStar)):
        add_suite(node.body, class_rebound_names)
        for handler in node.handlers:
            add_suite(handler.body, class_rebound_names)
        add_suite(node.orelse, class_rebound_names)
        add_suite(node.finalbody, class_rebound_names)
    elif isinstance(node, ast.Match):
        for case in node.cases:
            add_suite(case.body, class_rebound_names)
    return tuple(uses)


def _import_bindings(tree: ast.Module) -> dict[str, tuple[str, str | None]]:
    bindings: dict[str, tuple[str, str | None]] = {}
    ambiguous: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.partition(".")[0]
                if local_name in bindings:
                    ambiguous.add(local_name)
                bindings[local_name] = (alias.name, None)
        elif isinstance(node, ast.ImportFrom) and (node.module is not None or node.level > 0):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                if local_name in bindings:
                    ambiguous.add(local_name)
                bindings[local_name] = (module, alias.name)
    if any(
        isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(tree)
    ):
        ambiguous.update(bindings)
        ambiguous.update(vars(builtins))
    # Retain an opaque binding instead of making an ambiguous import look like
    # an unshadowed builtin (for example tuple or classmethod).
    return {**bindings, **{name: ("<ambiguous>", None) for name in ambiguous}}


def _module_rebound_names(tree: ast.Module) -> frozenset[str]:
    return _scope_rebound_names(tree.body, include_imports=False) | _nested_import_names(tree.body)


def _nested_import_names(statements: list[ast.stmt]) -> frozenset[str]:
    names: set[str] = set()

    def add_suite(suite: list[ast.stmt]) -> None:
        for statement in suite:
            if isinstance(statement, ast.Import):
                names.update(
                    alias.asname or alias.name.partition(".")[0] for alias in statement.names
                )
            elif isinstance(statement, ast.ImportFrom):
                names.update(
                    alias.asname or alias.name for alias in statement.names if alias.name != "*"
                )
            elif isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                add_suite(statement.body)
                add_suite(statement.orelse)
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                add_suite(statement.body)
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                add_suite(statement.body)
                for handler in statement.handlers:
                    add_suite(handler.body)
                add_suite(statement.orelse)
                add_suite(statement.finalbody)
            elif isinstance(statement, ast.Match):
                for case in statement.cases:
                    add_suite(case.body)

    # Direct imports are represented precisely by _import_bindings. Imports in
    # module-level control flow are conditional/rebindable and must fail closed.
    for statement in statements:
        if isinstance(statement, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            add_suite(statement.body)
            add_suite(statement.orelse)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            add_suite(statement.body)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            add_suite(statement.body)
            for handler in statement.handlers:
                add_suite(handler.body)
            add_suite(statement.orelse)
            add_suite(statement.finalbody)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                add_suite(case.body)
    return frozenset(names)


def _scope_rebound_names(
    statements: list[ast.stmt],
    *,
    include_imports: bool,
) -> frozenset[str]:
    names: set[str] = set()

    def add_target(target: ast.expr) -> None:
        names.update(
            item.id
            for item in ast.walk(target)
            if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del))
        )

    def add_expression(expression: ast.expr | None) -> None:
        if expression is None:
            return
        for item in ast.walk(expression):
            if isinstance(item, ast.NamedExpr):
                add_target(item.target)

    def add_suite(suite: list[ast.stmt]) -> None:
        for statement in suite:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(statement.name)
                for decorator in statement.decorator_list:
                    add_expression(decorator)
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for default in (*statement.args.defaults, *statement.args.kw_defaults):
                        add_expression(default)
                else:
                    for base in statement.bases:
                        add_expression(base)
                    for keyword in statement.keywords:
                        add_expression(keyword.value)
            elif include_imports and isinstance(statement, ast.Import):
                names.update(
                    alias.asname or alias.name.partition(".")[0] for alias in statement.names
                )
            elif include_imports and isinstance(statement, ast.ImportFrom):
                names.update(
                    alias.asname or alias.name for alias in statement.names if alias.name != "*"
                )
            elif isinstance(statement, (ast.Assign, ast.Delete)):
                for target in statement.targets:
                    add_target(target)
                if isinstance(statement, ast.Assign):
                    add_expression(statement.value)
            elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
                add_target(statement.target)
                add_expression(statement.value)
            elif isinstance(statement, (ast.For, ast.AsyncFor)):
                add_target(statement.target)
                add_expression(statement.iter)
                add_suite(statement.body)
                add_suite(statement.orelse)
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                for item in statement.items:
                    add_expression(item.context_expr)
                    if item.optional_vars is not None:
                        add_target(item.optional_vars)
                add_suite(statement.body)
            elif isinstance(statement, (ast.If, ast.While)):
                add_expression(statement.test)
                add_suite(statement.body)
                add_suite(statement.orelse)
            elif isinstance(statement, (ast.Try, ast.TryStar)):
                add_suite(statement.body)
                for handler in statement.handlers:
                    if handler.name is not None:
                        names.add(handler.name)
                    add_suite(handler.body)
                add_suite(statement.orelse)
                add_suite(statement.finalbody)
            elif isinstance(statement, ast.Match):
                add_expression(statement.subject)
                for case in statement.cases:
                    names.update(
                        item.name
                        for item in ast.walk(case.pattern)
                        if isinstance(item, (ast.MatchAs, ast.MatchStar)) and item.name is not None
                    )
                    names.update(
                        item.rest
                        for item in ast.walk(case.pattern)
                        if isinstance(item, ast.MatchMapping) and item.rest is not None
                    )
                    add_expression(case.guard)
                    add_suite(case.body)
            elif isinstance(statement, ast.Expr):
                add_expression(statement.value)
            elif isinstance(statement, ast.Assert):
                add_expression(statement.test)
                add_expression(statement.msg)
            elif isinstance(statement, ast.Raise):
                add_expression(statement.exc)
                add_expression(statement.cause)

    add_suite(statements)
    return frozenset(names)


def _declarative_identity_names(
    tree: ast.Module,
    *,
    import_bindings: Mapping[str, tuple[str, str | None]],
    rebound_names: frozenset[str],
) -> frozenset[str]:
    if (
        import_bindings.get("ExecutionProfileBehaviorIdentity")
        != ("cayu", "ExecutionProfileBehaviorIdentity")
        or "ExecutionProfileBehaviorIdentity" in rebound_names
    ):
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        targets: tuple[ast.expr, ...] = ()
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
            value = node.value
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "ExecutionProfileBehaviorIdentity"
        ):
            continue
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return frozenset(names)


def _import_time_expression_uses(
    node: ast.stmt,
    *,
    class_rebound_names: frozenset[str] = frozenset(),
) -> tuple[tuple[ast.expr, frozenset[str]], ...]:
    expressions: list[tuple[ast.expr, frozenset[str]]] = []

    def add(expression: ast.expr) -> None:
        expressions.append((expression, class_rebound_names))

    def add_suite(statements: list[ast.stmt]) -> None:
        for statement in statements:
            expressions.extend(
                _import_time_expression_uses(
                    statement,
                    class_rebound_names=class_rebound_names,
                )
            )

    if isinstance(node, ast.Expr):
        add(node.value)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            add(target)
        add(node.value)
    elif isinstance(node, ast.AugAssign):
        add(node.target)
        add(node.value)
    elif isinstance(node, ast.AnnAssign):
        add(node.target)
        add(node.annotation)
        if node.value is not None:
            add(node.value)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for expression in node.decorator_list:
            add(expression)
        for expression in node.args.defaults:
            add(expression)
        for expression in node.args.kw_defaults:
            if expression is not None:
                add(expression)
        for expression in _argument_annotations(node.args):
            add(expression)
        if node.returns is not None:
            add(node.returns)
    elif isinstance(node, ast.ClassDef):
        for expression in (*node.decorator_list, *node.bases):
            add(expression)
        for keyword in node.keywords:
            add(keyword.value)
        class_scope = _scope_rebound_names(node.body, include_imports=True)
        for statement in node.body:
            expressions.extend(
                _import_time_expression_uses(
                    statement,
                    class_rebound_names=class_scope,
                )
            )
    elif isinstance(node, ast.If):
        add(node.test)
        if _is_main_guard(node.test):
            add_suite(node.orelse)
        else:
            add_suite(node.body)
            add_suite(node.orelse)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        add(node.target)
        add(node.iter)
        add_suite(node.body)
        add_suite(node.orelse)
    elif isinstance(node, ast.While):
        add(node.test)
        add_suite(node.body)
        add_suite(node.orelse)
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            add(item.context_expr)
            if item.optional_vars is not None:
                add(item.optional_vars)
        add_suite(node.body)
    elif isinstance(node, (ast.Try, ast.TryStar)):
        add_suite(node.body)
        for handler in node.handlers:
            if handler.type is not None:
                add(handler.type)
            add_suite(handler.body)
        add_suite(node.orelse)
        add_suite(node.finalbody)
    elif isinstance(node, ast.Match):
        add(node.subject)
        for case in node.cases:
            if case.guard is not None:
                add(case.guard)
            add_suite(case.body)
    elif isinstance(node, ast.Assert):
        add(node.test)
        if node.msg is not None:
            add(node.msg)
    elif isinstance(node, ast.Raise):
        if node.exc is not None:
            add(node.exc)
        if node.cause is not None:
            add(node.cause)
    elif isinstance(node, ast.Delete):
        for target in node.targets:
            add(target)
    return tuple(expressions)


def _argument_annotations(arguments: ast.arguments) -> tuple[ast.expr, ...]:
    annotations = [
        argument.annotation
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
        if argument.annotation is not None
    ]
    if arguments.vararg is not None and arguments.vararg.annotation is not None:
        annotations.append(arguments.vararg.annotation)
    if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
        annotations.append(arguments.kwarg.annotation)
    return tuple(annotations)


def _is_main_guard(expression: ast.expr) -> bool:
    return (
        isinstance(expression, ast.Compare)
        and isinstance(expression.left, ast.Name)
        and expression.left.id == "__name__"
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], ast.Eq)
        and len(expression.comparators) == 1
        and isinstance(expression.comparators[0], ast.Constant)
        and expression.comparators[0].value == "__main__"
    )


class _EvaluatedCallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)


def _evaluated_calls(expression: ast.expr) -> tuple[ast.Call, ...]:
    collector = _EvaluatedCallCollector()
    collector.visit(expression)
    return tuple(collector.calls)


class _EvaluatedAttributeCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.attributes: list[ast.Attribute] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.attributes.append(node)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)


def _evaluated_attributes(expression: ast.expr) -> tuple[ast.Attribute, ...]:
    collector = _EvaluatedAttributeCollector()
    collector.visit(expression)
    return tuple(collector.attributes)


class _EvaluatedSubscriptCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.subscripts: list[ast.Subscript] = []

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.subscripts.append(node)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)


def _evaluated_subscripts(expression: ast.expr) -> tuple[ast.Subscript, ...]:
    collector = _EvaluatedSubscriptCollector()
    collector.visit(expression)
    return tuple(collector.subscripts)


def _private_collection_update_is_inert(
    call: ast.Call,
    *,
    declarative_collections: frozenset[str],
) -> bool:
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "append"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in _INERT_GENERATED_COLLECTION_NAMES
        and call.func.value.id in declarative_collections
        and not call.keywords
        and len(call.args) == 1
    ):
        return False
    # These are machine-owned list seams. Keep the exception intentionally
    # narrower than Python's expression grammar: a plain name lookup or literal
    # cannot invoke user code, while attributes, formatting, operators,
    # subscriptions, unpacking, and calls all can.
    return isinstance(call.args[0], (ast.Constant, ast.Name))


def _import_call_is_declarative(
    call: ast.Call,
    *,
    import_bindings: Mapping[str, tuple[str, str | None]],
    rebound_names: frozenset[str],
    declarative_identities: frozenset[str],
) -> bool:
    if isinstance(call.func, ast.Name):
        name = call.func.id
        if name in rebound_names:
            return False
        if name in {"frozenset", "list", "tuple"}:
            return name not in import_bindings
        expected_imports = {
            "AgentSpec": ("cayu", "AgentSpec"),
            "ExecutionProfileBehaviorIdentity": (
                "cayu",
                "ExecutionProfileBehaviorIdentity",
            ),
            "Path": ("pathlib", "Path"),
            "ToolSpec": ("cayu", "ToolSpec"),
            "dataclass": ("dataclasses", "dataclass"),
        }
        if name in expected_imports:
            return import_bindings.get(name) == expected_imports[name]
        if name in {"configured_model", "configured_provider_name"}:
            return import_bindings.get(name) in {
                ("configuration", name),
                ("configuration.settings", name),
            }
        return False
    if not isinstance(call.func, ast.Attribute):
        return False
    if (
        call.func.attr == "compile"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "re"
    ):
        return import_bindings.get("re") == ("re", None) and "re" not in rebound_names
    if call.func.attr == "resolve" and isinstance(call.func.value, ast.Call):
        constructor = call.func.value.func
        return (
            isinstance(constructor, ast.Name)
            and constructor.id == "Path"
            and import_bindings.get("Path") == ("pathlib", "Path")
            and "Path" not in rebound_names
        )
    if call.func.attr == "join":
        return isinstance(call.func.value, ast.Constant) and isinstance(call.func.value.value, str)
    if call.func.attr == "model_dump":
        return (
            isinstance(call.func.value, ast.Name) and call.func.value.id in declarative_identities
        )
    return False


def _check_registration_provenance(
    manifest: AppManifest,
) -> tuple[ProjectDiagnostic, ...]:
    diagnostics: list[ProjectDiagnostic] = []
    for agent in manifest.agents:
        provenance = agent.registration_provenance
        if provenance.location != "agents/registration.py":
            diagnostics.append(
                _diagnostic(
                    code="SCAFFOLD_REGISTRATION_PROVENANCE_DRIFT",
                    path=f"agents.{agent.name}.registration_provenance",
                    message=(
                        f"Agent {agent.name!r} was registered from "
                        f"{provenance.location or provenance.kind!r}, not "
                        "'agents/registration.py'."
                    ),
                    hint=(
                        "Perform explicit register_agent(...) calls in "
                        "agents/registration.py and call that seam from app.py."
                    ),
                    parameters={
                        "agent": agent.name,
                        "observed_location": provenance.location,
                    },
                )
            )
    return tuple(diagnostics)


def _diagnostic(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
    parameters: Mapping[str, object],
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
    tags: tuple[str, ...] = ("authoring", "configuration", "deploy"),
) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        code=code,
        severity=severity,
        subject="scaffold",
        path=path,
        message=message,
        hint=hint,
        tags=tags,
        parameters=parameters,
        documentation_anchor=_DOCS,
        verification_command=_VERIFY,
    )


def _filter(
    diagnostics: tuple[ProjectDiagnostic, ...],
    *,
    tags: frozenset[str],
    deploy_only: bool,
) -> tuple[ProjectDiagnostic, ...]:
    selected = diagnostics
    if deploy_only:
        selected = tuple(item for item in selected if "deploy" in item.tags)
    if tags:
        selected = tuple(item for item in selected if tags.intersection(item.tags))
    return selected
