"""Project improvement declarations: objectives, improvement evals and evaluator-only paths.

A Cayu project declares what an improvement loop may optimize next to its
``eval_target``::

    [tool.cayu.improvement]
    objectives = "cayu-objectives.toml"
    corpora = ["evals/improvement.json"]
    evaluator_only_paths = ["evals/reference"]

The objectives live in their own versioned file so they stay outside every
candidate's mutable paths. Each part here is usable alone: the schema models, the
objectives parser and fingerprint, the ``[tool.cayu.improvement]`` model, the
project loader and the file-level validator.

The Runtime defines, loads, fingerprints and validates this declaration; it does not
enforce it. Loading imports no project code, runs no eval and grants no authority.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.machinery import PathFinder
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import copy_durable_json_object
from cayu.evals.benchmark_package import (
    _benchmark_manifest_path,
    _contains_symlink,
    _is_normalized_relative_posix_path,
    load_benchmark_package,
)
from cayu.evals.corpus import (
    _CURRENCY_PATTERN,
    _bounded_durable_text,
    _canonical_decimal_text,
    _content_revision,
    _model_content_revision,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _SchemaV1PortableModel,
    load_eval_corpus,
)
from cayu.evals.execution_comparison import CorpusRegressionKind

PROJECT_OBJECTIVES_SCHEMA_VERSION = 1
PROJECT_IMPROVEMENT_DOCUMENT_SCHEMA_VERSION = 1
PROJECT_OBJECTIVES_MAX_BYTES = 1 << 20
SAMPLED_REVIEW_RATE_MAX_BPS = 10_000

_MAX_ENTRIES = 256
_MAX_PATH_CHARS = 512
_MAX_EVAL_TARGET_CHARS = 512
_MAX_NAME_CHARS = 256
_MAX_CURRENCY_CHARS = 16
_MAX_SECONDS = 30 * 24 * 60 * 60
_PYPROJECT = "pyproject.toml"
_EVAL_TARGET_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z",
    re.ASCII,
)


class ProjectImprovementError(ValueError):
    """A project's improvement declaration or objectives file cannot be loaded."""


def _project_path(value: str, field_name: str) -> str:
    _bounded_durable_text(value, field_name, max_chars=_MAX_PATH_CHARS, nonblank=True, clean=True)
    if not _is_normalized_relative_posix_path(value):
        raise ValueError(f"`{field_name}` must be a normalized project-relative POSIX path.")
    return value


def _path_key(value: str) -> str:
    """A comparison key that matches paths the way default macOS and Windows resolve them.

    Those filesystems open the same file for names that differ only in case or in
    Unicode normalization, so the key folds both (canonical caseless matching).
    Windows also ignores trailing dots and spaces in each name.
    """

    trimmed = "/".join(part.rstrip(" .") for part in value.split("/"))
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", trimmed).casefold())


def _unique_paths(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    for value in values:
        _project_path(value, field_name)
    if len({_path_key(value) for value in values}) != len(values):
        raise ValueError(f"`{field_name}` must not repeat a path, ignoring case.")
    return values


def _paths_overlap(first: str, second: str) -> bool:
    left = PurePosixPath(_path_key(first)).parts
    right = PurePosixPath(_path_key(second)).parts
    shorter = min(len(left), len(right))
    return left[:shorter] == right[:shorter]


class CorrectnessFloorV1(_PortableModel):
    """One suite whose compared results must not regress on the named dimensions.

    ``suite_id`` is the portable suite ID the corpus or benchmark package uses. The
    metrics are the regression dimensions ``cayu eval compare`` reports.
    """

    suite_id: StrictStr
    # Spelled out so schema version 1 keeps its meaning if `cayu eval compare` grows
    # a new regression dimension.
    metrics: tuple[CorpusRegressionKind, ...] = (
        CorpusRegressionKind.STATUS,
        CorpusRegressionKind.SCORE,
        CorpusRegressionKind.RELIABILITY,
    )

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls, value: tuple[CorpusRegressionKind, ...]
    ) -> tuple[CorpusRegressionKind, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("Correctness floor metrics must be a non-empty set.")
        order = tuple(CorpusRegressionKind)
        return tuple(sorted(value, key=order.index))


class MandatoryControlV1(_PortableModel):
    """A tool, or one approval gate on it, that is never removed or granted autonomy.

    Without ``approval_gate`` the entry covers every gate on the tool. The field
    names match the tool-authorization grant control, so a grant service can take
    these entries as given.
    """

    tool_name: StrictStr
    approval_gate: StrictStr | None = None

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value, info.field_name, max_chars=_MAX_NAME_CHARS, nonblank=True, clean=True
        )

    @field_validator("approval_gate")
    @classmethod
    def validate_approval_gate(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value, info.field_name, max_chars=_MAX_NAME_CHARS, nonblank=True, clean=True
        )


class CostTargetV1(_PortableModel):
    """A cost goal per verified completion, as a canonical decimal amount."""

    amount: StrictStr
    currency: StrictStr = "USD"

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: str, info) -> str:
        return _canonical_decimal_text(value, info.field_name)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        value = _bounded_durable_text(
            value, info.field_name, max_chars=_MAX_CURRENCY_CHARS, nonblank=True, clean=True
        )
        if _CURRENCY_PATTERN.fullmatch(value) is None:
            raise ValueError("currency must use uppercase ASCII letters, digits, '.', '_', or '-'.")
        return value


class ObjectiveTargetsV1(_PortableModel):
    """What "better" means: goals per verified completion, not enforced limits."""

    cost_per_verified_completion: CostTargetV1 | None = None
    seconds_per_verified_completion: StrictInt | None = Field(default=None, ge=1, le=_MAX_SECONDS)
    # Human-attention requests (approvals, questions) per verified completion.
    human_attention_requests_per_verified_completion: StrictStr | None = None

    @field_validator("human_attention_requests_per_verified_completion")
    @classmethod
    def validate_attention(cls, value: str | None, info) -> str | None:
        return None if value is None else _canonical_decimal_text(value, info.field_name)


class MutableSurfacesV1(_PortableModel):
    """What a candidate change may touch. Everything not listed stays fixed."""

    body_paths: tuple[StrictStr, ...] = Field(default=(), max_length=_MAX_ENTRIES)
    execution_profile: StrictBool = False
    model_per_step: StrictBool = False

    @field_validator("body_paths", mode="before")
    @classmethod
    def validate_body_paths_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("body_paths")
    @classmethod
    def validate_body_paths(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return tuple(sorted(_unique_paths(value, info.field_name)))


ApprovalRequirement = Literal["person"]


class ApprovalRulesV1(_PortableModel):
    """Who must approve each kind of change. Schema version 1 requires a person for all."""

    body_paths: ApprovalRequirement = "person"
    execution_profile: ApprovalRequirement = "person"
    model_per_step: ApprovalRequirement = "person"
    autonomy_grant: ApprovalRequirement = "person"


class ProjectObjectivesV1(_SchemaV1PortableModel):
    """The versioned objectives document a project's ``[tool.cayu.improvement]`` names."""

    correctness_floor: tuple[CorrectnessFloorV1, ...] = Field(min_length=1, max_length=_MAX_ENTRIES)
    mandatory_controls: tuple[MandatoryControlV1, ...] = Field(default=(), max_length=_MAX_ENTRIES)
    targets: ObjectiveTargetsV1 = Field(default_factory=ObjectiveTargetsV1)
    mutable_surfaces: MutableSurfacesV1 = Field(default_factory=MutableSurfacesV1)
    approval_rules: ApprovalRulesV1 = Field(default_factory=ApprovalRulesV1)
    # Share of autonomously granted decisions a person still reviews, in basis
    # points (500 = 5%), so the fingerprinted document never holds a float.
    sampled_review_rate_bps: StrictInt = Field(ge=0, le=SAMPLED_REVIEW_RATE_MAX_BPS)

    @field_validator("correctness_floor", "mandatory_controls", mode="before")
    @classmethod
    def validate_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    # Entries are sorted so the fingerprint does not depend on authoring order.
    @field_validator("correctness_floor")
    @classmethod
    def validate_floor(
        cls, value: tuple[CorrectnessFloorV1, ...]
    ) -> tuple[CorrectnessFloorV1, ...]:
        suite_ids = [entry.suite_id for entry in value]
        if len(set(suite_ids)) != len(suite_ids):
            raise ValueError("Correctness floor entries must name each suite once.")
        return tuple(sorted(value, key=lambda entry: entry.suite_id))

    @field_validator("mandatory_controls")
    @classmethod
    def validate_controls(
        cls, value: tuple[MandatoryControlV1, ...]
    ) -> tuple[MandatoryControlV1, ...]:
        keys = [(entry.tool_name, entry.approval_gate) for entry in value]
        if len(set(keys)) != len(keys):
            raise ValueError("Mandatory controls must not repeat a tool and gate.")
        whole_tools = {tool for tool, gate in keys if gate is None}
        if any(gate is not None and tool in whole_tools for tool, gate in keys):
            raise ValueError(
                "A mandatory control for one gate is redundant with the same tool's "
                "entry without approval_gate."
            )
        return tuple(sorted(value, key=lambda entry: (entry.tool_name, entry.approval_gate or "")))


def project_objectives_fingerprint(objectives: ProjectObjectivesV1) -> str:
    """Return the ``sha256:`` content revision missions and receipts cite.

    The fingerprint covers the validated document, defaults included, in canonical
    JSON. Comments, formatting, key order and entry order do not change it.
    """

    if type(objectives) is not ProjectObjectivesV1:
        raise TypeError("objectives must be a ProjectObjectivesV1.")
    validated = ProjectObjectivesV1.model_validate(objectives.model_dump(mode="json"))
    return _model_content_revision(validated, "project objectives")


def project_objectives_from_toml(source: str) -> ProjectObjectivesV1:
    """Parse and validate one objectives document from TOML text."""

    # A character count above the limit already exceeds it in bytes, so only text
    # that might fit is encoded to measure.
    if (
        type(source) is not str
        or len(source) > PROJECT_OBJECTIVES_MAX_BYTES
        or len(source.encode("utf-8")) > PROJECT_OBJECTIVES_MAX_BYTES
    ):
        raise ValueError("Project objectives must be bounded TOML text.")
    try:
        decoded = tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Project objectives are not valid TOML: {exc}") from exc
    except RecursionError as exc:
        raise ValueError("Project objectives exceed the supported TOML nesting.") from exc
    if type(decoded.get("schema_version")) is not int:
        raise ValueError(f"schema_version must be the integer {PROJECT_OBJECTIVES_SCHEMA_VERSION}.")
    return ProjectObjectivesV1.model_validate(
        copy_durable_json_object(decoded, "project objectives")
    )


def load_project_objectives(path: str | Path) -> ProjectObjectivesV1:
    """Read at most the objectives byte limit, then parse and validate it."""

    with Path(path).open("rb") as stream:
        raw = stream.read(PROJECT_OBJECTIVES_MAX_BYTES + 1)
    if len(raw) > PROJECT_OBJECTIVES_MAX_BYTES:
        raise ValueError(f"Project objectives exceed {PROJECT_OBJECTIVES_MAX_BYTES} bytes.")
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Project objectives must be UTF-8.") from exc
    return project_objectives_from_toml(source)


class ProjectImprovementConfigV1(_PortableModel):
    """The ``[tool.cayu.improvement]`` table of a project's ``pyproject.toml``.

    Paths are project-relative. ``eval_targets`` use the ``module:attribute`` form
    of ``[tool.cayu].eval_target``; ``corpora`` are portable corpus files and
    ``benchmark_packages`` are package directories or manifests. Evaluator-only
    paths hold material a candidate must never see, such as reference answers and
    expected states.
    """

    objectives: StrictStr
    eval_targets: tuple[StrictStr, ...] = Field(default=(), max_length=_MAX_ENTRIES)
    corpora: tuple[StrictStr, ...] = Field(default=(), max_length=_MAX_ENTRIES)
    benchmark_packages: tuple[StrictStr, ...] = Field(default=(), max_length=_MAX_ENTRIES)
    evaluator_only_paths: tuple[StrictStr, ...] = Field(default=(), max_length=_MAX_ENTRIES)

    @field_validator("objectives")
    @classmethod
    def validate_objectives(cls, value: str, info) -> str:
        return _project_path(value, info.field_name)

    @field_validator(
        "eval_targets", "corpora", "benchmark_packages", "evaluator_only_paths", mode="before"
    )
    @classmethod
    def validate_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("eval_targets")
    @classmethod
    def validate_eval_targets(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        for target in value:
            _bounded_durable_text(
                target, info.field_name, max_chars=_MAX_EVAL_TARGET_CHARS, nonblank=True, clean=True
            )
            if _EVAL_TARGET_PATTERN.fullmatch(target) is None:
                raise ValueError("`eval_targets` entries must use module:attribute syntax.")
        if len(set(value)) != len(value):
            raise ValueError("`eval_targets` must not repeat a target.")
        return tuple(sorted(value))

    # Sorted like the objectives' entries, so the declaration fingerprint does not
    # depend on authoring order.
    @field_validator("corpora", "benchmark_packages", "evaluator_only_paths")
    @classmethod
    def validate_paths(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return tuple(sorted(_unique_paths(value, info.field_name)))

    @model_validator(mode="after")
    def validate_evals(self) -> ProjectImprovementConfigV1:
        if not (self.eval_targets or self.corpora or self.benchmark_packages):
            raise ValueError(
                "[tool.cayu.improvement] must name at least one of eval_targets, corpora, "
                "or benchmark_packages."
            )
        return self


def project_improvement_fingerprint(
    config: ProjectImprovementConfigV1, objectives: ProjectObjectivesV1
) -> str:
    """Return the ``sha256:`` content revision of the whole improvement declaration.

    It covers the ``[tool.cayu.improvement]`` table and the objectives fingerprint,
    so changing which evals are declared or which paths stay evaluator-only changes
    it even when the objectives file does not. Entry order does not change it. It
    identifies the declaration, not the content of the files it names: corpora and
    benchmark packages carry their own content revisions.
    """

    if type(config) is not ProjectImprovementConfigV1:
        raise TypeError("config must be a ProjectImprovementConfigV1.")
    return _declaration_fingerprint(config, project_objectives_fingerprint(objectives))


def _declaration_fingerprint(
    config: ProjectImprovementConfigV1, objectives_fingerprint: str
) -> str:
    validated = ProjectImprovementConfigV1.model_validate(config.model_dump(mode="json"))
    return _content_revision(
        {
            "improvement": validated.model_dump(mode="json"),
            "objectives_fingerprint": objectives_fingerprint,
        },
        "project improvement declaration",
    )


def _eval_target_files(root: Path, target: str) -> tuple[Path, ...]:
    """The files importing an eval target's module runs, or ``()`` if it cannot be found.

    Python's path-based finder locates the module without running any code. It
    searches the project root first, as ``cayu eval run`` does, then ``src/``, then
    this interpreter's import path, so a regular package elsewhere wins over a
    namespace directory in the project just as it would at import time. The result
    is the ``__init__.py`` of every enclosing regular package, which an import runs
    first, then the module's own file.
    """

    names = target.split(":", 1)[0].split(".")
    search: list[str] | None = [str(root), str(root / "src"), *sys.path]
    files: list[Path] = []
    for depth in range(1, len(names) + 1):
        spec = None if search is None else PathFinder.find_spec(".".join(names[:depth]), search)
        if spec is None:
            return ()
        if spec.has_location and spec.origin is not None:
            files.append(Path(spec.origin))
        locations = spec.submodule_search_locations
        search = None if locations is None else list(locations)
    return tuple(files)


def _symbolic_links(root: Path, relative: str) -> Iterator[str]:
    """Symbolic links anywhere inside a project directory, as project-relative paths."""

    directory = root / relative
    if directory.is_symlink() or not directory.is_dir():
        return
    for current, directories, files in os.walk(directory):
        directories.sort()
        for name in (*directories, *sorted(files)):
            if (entry := Path(current) / name).is_symlink():
                yield entry.relative_to(root).as_posix()


def _benchmark_package_directory(root: Path, relative: str) -> str:
    """The directory ``load_benchmark_package`` reads a declared package's inputs from.

    A declared manifest file names the directory that contains it, which is ``.`` for
    a manifest at the project root. Any other path, including one that does not exist
    yet, is taken as the package directory itself.
    """

    if (root / relative).is_file():
        return str(PurePosixPath(relative).parent)
    return relative


def _benchmark_package_protected_path(root: Path, relative: str) -> str:
    # A package at the project root would protect everything; the validator reports
    # it, and only its declared manifest is protected.
    directory = _benchmark_package_directory(root, relative)
    return relative if directory == "." else directory


def _protected_paths(root: Path, config: ProjectImprovementConfigV1) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                _PYPROJECT,
                config.objectives,
                *config.corpora,
                *(
                    _benchmark_package_protected_path(root, relative)
                    for relative in config.benchmark_packages
                ),
                *config.evaluator_only_paths,
                *(
                    path.relative_to(root).as_posix()
                    for target in config.eval_targets
                    for path in _eval_target_files(root, target)
                    if path.is_relative_to(root)
                ),
            )
        )
    )


@dataclass(frozen=True)
class ProjectImprovementDeclaration:
    """One project's loaded improvement declaration and its identities.

    ``objectives_fingerprint`` identifies the objectives file alone;
    ``declaration_fingerprint`` also covers the declared evals and evaluator-only paths.
    ``protected_paths`` are the project-relative paths that must stay outside every
    mutable surface: ``pyproject.toml``, the objectives file, every declared corpus,
    benchmark package directory and evaluator-only path, and the module file of each
    eval target found in the checkout. Modules a target imports are not traced.
    """

    root: Path
    pyproject: Path
    config: ProjectImprovementConfigV1
    objectives: ProjectObjectivesV1
    objectives_fingerprint: str
    declaration_fingerprint: str
    protected_paths: tuple[str, ...]


def _contained(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    if _contains_symlink(root, path):
        raise ProjectImprovementError(f"{relative}: project paths must not contain symbolic links.")
    if not path.resolve().is_relative_to(root):
        raise ProjectImprovementError(f"{relative}: path is outside the project.")
    return path


def _declaration(root: Path, pyproject: Path, improvement: object) -> ProjectImprovementDeclaration:
    if not isinstance(improvement, Mapping):
        raise ProjectImprovementError(f"{pyproject}: [tool.cayu.improvement] must be a table.")
    try:
        config = ProjectImprovementConfigV1.model_validate(dict(improvement))
    except ValueError as exc:
        raise ProjectImprovementError(f"{pyproject}: [tool.cayu.improvement]: {exc}") from exc
    objectives_path = _contained(root, config.objectives)
    if not objectives_path.is_file():
        raise ProjectImprovementError(
            f"{pyproject}: objectives file {config.objectives} does not exist."
        )
    try:
        objectives = load_project_objectives(objectives_path)
    except (OSError, ValueError) as exc:
        raise ProjectImprovementError(f"{config.objectives}: {exc}") from exc
    objectives_fingerprint = project_objectives_fingerprint(objectives)
    return ProjectImprovementDeclaration(
        root=root,
        pyproject=pyproject,
        config=config,
        objectives=objectives,
        objectives_fingerprint=objectives_fingerprint,
        declaration_fingerprint=_declaration_fingerprint(config, objectives_fingerprint),
        protected_paths=_protected_paths(root, config),
    )


def load_project_improvement(root: str | Path) -> ProjectImprovementDeclaration:
    """Load the declaration from ``root/pyproject.toml`` without searching elsewhere."""

    root = Path(root).resolve()
    pyproject = root / _PYPROJECT
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectImprovementError(f"Could not read {pyproject}: {exc}") from exc
    except RecursionError as exc:
        raise ProjectImprovementError(f"{pyproject} exceeds the supported TOML nesting.") from exc
    tool = document.get("tool", {})
    cayu = tool.get("cayu", {}) if isinstance(tool, dict) else {}
    if not isinstance(cayu, dict) or "improvement" not in cayu:
        raise ProjectImprovementError(f"{pyproject}: [tool.cayu.improvement] is not configured.")
    return _declaration(root, pyproject, cayu["improvement"])


def discover_project_improvement(
    start: str | Path | None = None,
) -> ProjectImprovementDeclaration | None:
    """Find the nearest Cayu project from ``start`` upward and load its declaration.

    The search is the upward walk ``cayu eval run`` uses, stopping at the nearest
    ``pyproject.toml`` whose ``[tool.cayu]`` declares ``factory``, ``eval_target`` or
    ``improvement``. It returns ``None`` when there is no Cayu project or the nearest
    one declares no improvement table.
    """

    from cayu.cli.project import ProjectError, discover_cayu_project_configuration

    try:
        discovered = discover_cayu_project_configuration(
            discovery_keys=("factory", "eval_target", "improvement"),
            start=None if start is None else Path(start),
        )
    except ProjectError as exc:
        raise ProjectImprovementError(str(exc)) from exc
    if discovered is None or "improvement" not in discovered.config:
        return None
    return _declaration(
        discovered.root.resolve(), discovered.pyproject, discovered.config["improvement"]
    )


def validate_project_improvement(declaration: ProjectImprovementDeclaration) -> tuple[str, ...]:
    """Check the declaration against the checkout and return every problem found.

    Every corpus must be a file and load, every benchmark package must load,
    evaluator-only and body paths must exist, each eval target's module and enclosing
    packages must be found inside the project so they can be protected, body paths
    must not overlap protected paths, no body or protected directory may contain a
    symbolic link, and each correctness-floor suite must appear in a declared corpus
    or package.
    Python ``eval_targets`` are never imported, so when any are declared, floor
    suites missing from the static evals are assumed to come from those targets.
    """

    root = declaration.root
    config = declaration.config
    problems: list[str] = []
    suite_ids: set[str] = set()

    def existing(relative: str, label: str) -> Path | None:
        try:
            path = _contained(root, relative)
        except ProjectImprovementError as exc:
            problems.append(f"{label} {exc}")
            return None
        if not path.exists():
            problems.append(f"{label} {relative} does not exist.")
            return None
        return path

    for relative in config.corpora:
        if (path := existing(relative, "corpus")) is not None:
            if not path.is_file():
                problems.append(f"corpus {relative} is not a file.")
                continue
            try:
                suite_ids.update(suite.id for suite in load_eval_corpus(path).suites)
            except (OSError, ValueError) as exc:
                problems.append(f"corpus {relative} is invalid: {exc}")
    manifests: dict[str, str] = {}
    for relative in config.benchmark_packages:
        if (path := existing(relative, "benchmark package")) is not None:
            if _benchmark_package_directory(root, relative) == ".":
                problems.append(
                    f"benchmark package {relative} is at the project root; give it its own "
                    "directory so its inputs can be protected."
                )
            manifest = _benchmark_manifest_path(path).relative_to(root).as_posix()
            if (first := manifests.setdefault(_path_key(manifest), relative)) != relative:
                problems.append(f"benchmark packages {first} and {relative} are the same package.")
                continue
            if existing(manifest, "benchmark package") is None:
                continue
            try:
                suite_ids.add(load_benchmark_package(path).package.suite.suite.id)
            except (OSError, ValueError) as exc:
                problems.append(f"benchmark package {relative} is invalid: {exc}")
    for relative in config.evaluator_only_paths:
        existing(relative, "evaluator-only path")
    for target in config.eval_targets:
        files = _eval_target_files(root, target)
        if not files:
            problems.append(f"eval target {target} cannot be found, so it cannot be protected.")
        for path in files:
            if path.is_relative_to(root):
                existing(path.relative_to(root).as_posix(), f"eval target {target} module")
            else:
                problems.append(
                    f"eval target {target} imports {path}, which is outside the project, "
                    "so it cannot be protected."
                )
    existing(_PYPROJECT, "project file")
    protected = declaration.protected_paths
    body_paths = declaration.objectives.mutable_surfaces.body_paths
    for relative in body_paths:
        existing(relative, "body path")
        problems.extend(
            f"body path {relative} overlaps protected path {other}."
            for other in protected
            if _paths_overlap(relative, other)
        )
    # A link inside a body or protected directory would cross the boundary between
    # what a candidate may change and what only the evaluator may see.
    for label, relative in (
        *(("body path", path) for path in body_paths),
        *(("protected path", path) for path in protected),
    ):
        problems.extend(
            f"{label} {relative} contains a symbolic link at {link}."
            for link in _symbolic_links(root, relative)
        )
    if not config.eval_targets:
        problems.extend(
            f"correctness floor suite {entry.suite_id} is not in any declared corpus or "
            "benchmark package."
            for entry in declaration.objectives.correctness_floor
            if entry.suite_id not in suite_ids
        )
    return tuple(problems)


def project_improvement_document(
    declaration: ProjectImprovementDeclaration,
    problems: tuple[str, ...] = (),
) -> dict[str, Any]:
    """A JSON-ready description of the declaration for clients outside Python."""

    return {
        "schema_version": PROJECT_IMPROVEMENT_DOCUMENT_SCHEMA_VERSION,
        "root": str(declaration.root),
        "pyproject": str(declaration.pyproject),
        "improvement": declaration.config.model_dump(mode="json"),
        "objectives": declaration.objectives.model_dump(mode="json"),
        "objectives_fingerprint": declaration.objectives_fingerprint,
        "declaration_fingerprint": declaration.declaration_fingerprint,
        "protected_paths": list(declaration.protected_paths),
        "problems": list(problems),
    }


__all__ = [
    "PROJECT_IMPROVEMENT_DOCUMENT_SCHEMA_VERSION",
    "PROJECT_OBJECTIVES_MAX_BYTES",
    "PROJECT_OBJECTIVES_SCHEMA_VERSION",
    "SAMPLED_REVIEW_RATE_MAX_BPS",
    "ApprovalRulesV1",
    "CorrectnessFloorV1",
    "CostTargetV1",
    "MandatoryControlV1",
    "MutableSurfacesV1",
    "ObjectiveTargetsV1",
    "ProjectImprovementConfigV1",
    "ProjectImprovementDeclaration",
    "ProjectImprovementError",
    "ProjectObjectivesV1",
    "discover_project_improvement",
    "load_project_improvement",
    "load_project_objectives",
    "project_improvement_document",
    "project_improvement_fingerprint",
    "project_objectives_fingerprint",
    "project_objectives_from_toml",
    "validate_project_improvement",
]
