"""Pin public registration/contracts without exposing private receiver schemas."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from cayu.collaboration import exports

PUBLIC_EXPORT_NAMES = {
    "ExportLimits",
    "SessionExportAcceptance",
    "SessionExportAcceptanceReader",
    "SessionExportAccessContext",
    "SessionExportAction",
    "SessionExportAuthorization",
    "SessionExportCapacityExceeded",
    "SessionExportConflict",
    "SessionExportDenied",
    "SessionExportIntent",
    "SessionExportNamespace",
    "SessionExportPolicy",
    "SessionExportProjector",
    "SessionExportReceipt",
    "SessionExportRuntimeOrigin",
    "SessionExportReconciliation",
    "SessionExportRef",
    "SessionExportRegistration",
    "SessionExportRequest",
    "SessionExportSettlementReceipt",
    "SessionExportSettlementRequest",
    "SessionExportUnavailable",
}


@pytest.mark.parametrize("package_name", ["cayu", "cayu.collaboration"])
def test_session_export_manifests_stubs_and_runtime_agree(package_name) -> None:
    package = importlib.import_module(package_name)
    manifest = importlib.import_module(package_name + "._exports")
    declarations = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
    declared = {
        alias.asname or alias.name
        for node in declarations.body
        if isinstance(node, ast.ImportFrom) and node.module == exports.__name__
        for alias in node.names
    }
    exported = {
        name for name, (module, _) in manifest.EXPORTS.items() if module == exports.__name__
    }
    assert exported == declared == PUBLIC_EXPORT_NAMES
    assert set(package.__all__) >= PUBLIC_EXPORT_NAMES
    for name in PUBLIC_EXPORT_NAMES:
        assert getattr(package, name) is getattr(exports, name)
        assert manifest.EXPORTS[name] == (exports.__name__, name)
    assert (
        not {
            "ExportRoot",
            "ExportRecord",
            "ExportMutation",
            "SettlementRecord",
            "SessionExportCoordinator",
            "ExportDigest",
            "SourceIndex",
        }
        & exported
    )
    assert not any(
        module
        in {
            "cayu.collaboration._session_export_store",
            "cayu.collaboration._session_export_coordinator",
        }
        for module, _ in manifest.EXPORTS.values()
    )


def test_all_user_facing_export_contract_classes_are_manifested() -> None:
    definitions = ast.parse(Path(exports.__file__).read_text())
    classes = {node.name for node in definitions.body if isinstance(node, ast.ClassDef)}
    assert classes | {"SessionExportAction"} == PUBLIC_EXPORT_NAMES


@pytest.mark.parametrize("package_name", ["cayu", "cayu.collaboration"])
@pytest.mark.parametrize("family", ["mandates", "releases"])
def test_registered_authority_and_release_contracts_are_public(package_name, family):
    module = importlib.import_module("cayu.collaboration." + family)
    package = importlib.import_module(package_name)
    manifest = importlib.import_module(package_name + "._exports")
    source = ast.parse(Path(module.__file__).read_text())
    names = {node.name for node in source.body if isinstance(node, ast.ClassDef)}
    if family == "mandates":
        names |= {"MandateAction", "InputChannel"}
    declarations = ast.parse(Path(package.__file__).with_suffix(".pyi").read_text())
    declared = {
        alias.asname or alias.name
        for node in declarations.body
        if isinstance(node, ast.ImportFrom) and node.module == module.__name__
        for alias in node.names
    }
    assert declared == names
    assert {
        name for name, (owner, _) in manifest.EXPORTS.items() if owner == module.__name__
    } == names
    for name in names:
        assert name in package.__all__
        assert getattr(package, name) is getattr(module, name)


def test_session_export_example_imports_only_public_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    example = ast.parse((root / "examples/collaboration/session_export.py").read_text())
    for node in ast.walk(example):
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("cayu"):
            continue
        assert node.module in {"cayu", "cayu.collaboration"}
        package = importlib.import_module(node.module)
        for alias in node.names:
            assert alias.name in package.__all__
            assert getattr(package, alias.name) is not None
