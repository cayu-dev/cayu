from __future__ import annotations

import io
import json
import runpy
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest

from cayu._dashboard_artifacts import (
    API_TYPE_FILES,
    API_TYPES_ROOT,
    BUNDLE_MANIFEST_NAME,
    BUNDLE_PACKAGE_ROOT,
    EVAL_TEXT_LIMITS_FILE,
    GENERATED_SOURCE_FILES,
    LABEL_LIMITS_FILE,
    OPENAPI_FILE,
    RELEASE_METADATA_FILE,
    generated_api_contents,
    render_release_metadata,
    source_bundle_name,
)
from cayu.cli.dashboard import (
    DashboardSourceError,
    render_dashboard_source_manifest,
    validate_dashboard_source_bundle,
)

_ROOT = Path(__file__).parents[2]


def _bundle_with_generated_edit(path: str, content: bytes | None) -> tuple[bytes, dict]:
    root = files("cayu").joinpath(*BUNDLE_PACKAGE_ROOT.split("/"))
    bundle = next(item for item in root.iterdir() if item.name.endswith(".zip"))
    with zipfile.ZipFile(io.BytesIO(bundle.read_bytes())) as source:
        contents = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(contents.pop(BUNDLE_MANIFEST_NAME))
    if content is None:
        contents.pop(path)
    else:
        contents[path] = content
    contents[BUNDLE_MANIFEST_NAME] = render_dashboard_source_manifest(
        contents,
        cayu_version=manifest["cayu_version"],
        server_contract_version=manifest["server_contract_version"],
        compiled_dashboard_digest=manifest["compiled_dashboard_digest"],
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as target:
        for name, value in contents.items():
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            target.writestr(member, value)
    return output.getvalue(), manifest


@pytest.mark.parametrize("path", sorted(GENERATED_SOURCE_FILES))
def test_source_bundle_requires_every_generated_output_even_with_matching_manifest(path):
    bundle, manifest = _bundle_with_generated_edit(path, None)
    with pytest.raises(DashboardSourceError, match="bundle is incomplete"):
        validate_dashboard_source_bundle(
            bundle,
            expected_cayu_version=manifest["cayu_version"],
            expected_server_contract_version=manifest["server_contract_version"],
        )


@pytest.mark.parametrize(
    "path", ["src/lib/generated/stale.ts", "src/lib/generated/server-api/stale.ts"]
)
def test_source_bundle_rejects_undeclared_generated_outputs_with_matching_manifest(path):
    bundle, manifest = _bundle_with_generated_edit(path, b"export {}\n")
    with pytest.raises(DashboardSourceError, match="unexpected generated files"):
        validate_dashboard_source_bundle(
            bundle,
            expected_cayu_version=manifest["cayu_version"],
            expected_server_contract_version=manifest["server_contract_version"],
        )


def test_source_bundle_still_accepts_non_generated_source_extensions():
    bundle, manifest = _bundle_with_generated_edit("src/customer-panel.ts", b"export {}\n")
    result = validate_dashboard_source_bundle(
        bundle,
        expected_cayu_version=manifest["cayu_version"],
        expected_server_contract_version=manifest["server_contract_version"],
    )
    assert result.contents["src/customer-panel.ts"] == b"export {}\n"


@pytest.mark.parametrize("damage", ["missing", "extra", "stale-metadata"])
def test_release_validation_rejects_invalid_generated_source(damage, tmp_path):
    path, content = {
        "missing": (EVAL_TEXT_LIMITS_FILE, None),
        "extra": ("src/lib/generated/obsolete.ts", b"export {}\n"),
        "stale-metadata": (
            RELEASE_METADATA_FILE,
            render_release_metadata(cayu_version="0.0.0", server_contract_version="1"),
        ),
    }[damage]
    bundle, manifest = _bundle_with_generated_edit(path, content)
    validator = runpy.run_path(str(_ROOT / "scripts/check_release_artifacts.py"))
    prefix = validator["_WHEEL_DASHBOARD_SOURCE_PREFIX"]
    name = f"{prefix}/{source_bundle_name(manifest['cayu_version'])}"
    with pytest.raises(ValueError, match="invalid dashboard source bundle"):
        validator["_validate_dashboard_source"](
            {name: bundle},
            archive=tmp_path / "example.whl",
            package_version=manifest["cayu_version"],
            bundle_prefix=prefix,
            compiled_prefix="cayu/server/dashboard",
        )


def test_shared_release_renderer_preserves_version_values_and_api_digest_scope():
    metadata = render_release_metadata(cayu_version="2.3.4", server_contract_version="99")
    assert metadata == (
        b"// Generated from Cayu package metadata and cayu._server_contract_version.\n"
        b"// Regenerate with npm run generate:api in the intended Cayu environment.\n"
        b'export const DASHBOARD_SOURCE_CAYU_VERSION = "2.3.4"\n'
        b'export const SUPPORTED_SERVER_CONTRACT_VERSION = "99"\n'
    )
    contents = {
        OPENAPI_FILE: b"{}\n",
        **{path: b"export {}\n" for path in API_TYPE_FILES},
        EVAL_TEXT_LIMITS_FILE: b"limits",
        LABEL_LIMITS_FILE: b"label limits",
        RELEASE_METADATA_FILE: metadata,
    }
    assert generated_api_contents(contents) == {
        OPENAPI_FILE: b"{}\n",
        **{path: b"export {}\n" for path in API_TYPE_FILES},
    }


def test_api_check_reports_missing_release_metadata(tmp_path, monkeypatch):
    main = runpy.run_path(str(_ROOT / "dashboard/scripts/generate-api-types.py"))["main"]
    monkeypatch.setitem(main.__globals__, "RELEASE_METADATA", tmp_path / "absent.ts")
    with pytest.raises(SystemExit, match="release metadata is missing"):
        main(["--check"])


@pytest.mark.parametrize("check", [False, True], ids=["generate", "check"])
@pytest.mark.parametrize("damage", ["missing-api-output", "extra-api-output", "extra-generated"])
def test_api_command_rejects_output_inventory_drift(tmp_path, monkeypatch, check, damage):
    main = runpy.run_path(str(_ROOT / "dashboard/scripts/generate-api-types.py"))["main"]
    namespace = main.__globals__
    generated = tmp_path / API_TYPES_ROOT

    def generate(*, schema_path, output_dir):
        output_dir.mkdir(parents=True)
        for path in API_TYPE_FILES:
            (output_dir / Path(path).relative_to(API_TYPES_ROOT)).write_bytes(b"export {}\n")
        if damage == "missing-api-output":
            (output_dir / "types.gen.ts").unlink()
        elif damage == "extra-api-output":
            (output_dir / "unexpected.ts").write_bytes(b"export {}\n")

    for path in GENERATED_SOURCE_FILES:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"export {}\n")
    (tmp_path / OPENAPI_FILE).write_text("{}\n")
    (tmp_path / RELEASE_METADATA_FILE).write_bytes(namespace["_release_metadata"]())
    (tmp_path / EVAL_TEXT_LIMITS_FILE).write_bytes(namespace["_eval_text_limits"]())
    (tmp_path / LABEL_LIMITS_FILE).write_bytes(namespace["_label_limits"]())
    if damage == "extra-generated":
        (generated.parent / "unexpected.ts").write_text("export {}\n")
    for key, value in {
        "GENERATED_SOURCE_ROOT": generated.parent,
        "GENERATED_DIR": generated,
        "OPENAPI_BASELINE": tmp_path / OPENAPI_FILE,
        "RELEASE_METADATA": tmp_path / RELEASE_METADATA_FILE,
        "EVAL_TEXT_LIMITS": tmp_path / EVAL_TEXT_LIMITS_FILE,
        "LABEL_LIMITS": tmp_path / LABEL_LIMITS_FILE,
        "_assert_expected_cayu_import": lambda: None,
        "_openapi_json": lambda: "{}\n",
        "_run_generator": generate,
    }.items():
        monkeypatch.setitem(namespace, key, value)
    with pytest.raises(SystemExit, match="declared files|unexpected generated files"):
        main(["--check"] if check else [])
