from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest
from scripts import build_dashboard as builder
from scripts.build_dashboard_source_bundle import build_bundle, tree_contents
from scripts.check_release_artifacts import validate_dashboard_build

from cayu._dashboard_artifacts import (
    BUILD_INPUT_FILES,
    BUNDLE_PACKAGE_ROOT,
    COMPILED_PACKAGE_ROOT,
    SOURCE_ROOT,
)

_ROOT = Path(__file__).parents[2]


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    shutil.copyfile(_ROOT / ".gitattributes", root / ".gitattributes")
    shutil.copyfile(_ROOT / "pyproject.toml", root / "pyproject.toml")
    builder._write_files(root, builder._inputs(_ROOT))
    for package_path in (COMPILED_PACKAGE_ROOT, BUNDLE_PACKAGE_ROOT):
        source = files("cayu").joinpath(*package_path.split("/"))
        shutil.copytree(str(source), root / "src" / "cayu" / package_path)
    monkeypatch.setattr(builder, "ROOT", root)
    run = subprocess.run
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda command, **kwargs: None if command[0] == "docker" else run(command, **kwargs),
    )

    def compile(stage):
        shutil.copytree(
            root / "src" / "cayu" / COMPILED_PACKAGE_ROOT,
            stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT,
        )

    monkeypatch.setattr(builder, "_build_frontend", compile)
    return root


def test_canonical_build_uses_same_bundle_writer_and_repeats_without_changes(checkout):
    assert builder.main([]) == 0
    first = builder._outputs(checkout)
    assert builder.main([]) == 0
    assert builder._outputs(checkout) == first
    assert builder.main(["--check"]) == 0
    bundles = [content for name, content in first.items() if name.endswith(".zip")]
    assert bundles == [
        build_bundle(cayu_version=builder.project_version(checkout), repo_root=checkout)
    ]


def test_check_reports_changes_without_writing_and_regeneration_repairs_them(checkout, capsys):
    assert builder.main([]) == 0
    baseline = builder._outputs(checkout)
    assets = checkout / "src" / "cayu" / COMPILED_PACKAGE_ROOT
    bundle_root = checkout / "src" / "cayu" / BUNDLE_PACKAGE_ROOT
    bundle = next(bundle_root.glob("*.zip"))
    bundle.write_bytes(b"stale")
    (bundle_root / "cayu-dashboard-source-old.zip").write_bytes(b"old")
    source = checkout / SOURCE_ROOT / "src" / "new-panel.ts"
    source.write_bytes(b"export {}\n")
    # Compile from captured good assets, not the damaged checkout outputs.
    compiled = {
        name.split(f"{COMPILED_PACKAGE_ROOT}/", 1)[1]: content
        for name, content in baseline.items()
        if f"{COMPILED_PACKAGE_ROOT}/" in name
    }
    builder._build_frontend = lambda stage: builder._write_files(
        stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT, compiled
    )
    (assets / "index.html").unlink()
    (assets / "obsolete.js").write_bytes(b"old")
    before = builder._outputs(checkout)
    assert builder.main(["--check"]) == 1
    assert builder._outputs(checkout) == before
    report = capsys.readouterr().out
    assert "added   src/cayu/server/dashboard/index.html" in report
    assert "removed src/cayu/server/dashboard/obsolete.js" in report
    assert "removed src/cayu/data/dashboard_source/cayu-dashboard-source-old.zip" in report
    assert f"changed src/cayu/data/dashboard_source/{bundle.name}" in report
    assert builder.main([]) == 0
    assert not (assets / "obsolete.js").exists()
    assert builder.main(["--check"]) == 0
    assert source.read_bytes() == b"export {}\n"


def test_canonical_inputs_normalize_text_and_preserve_binary_and_local_builds(checkout):
    text = checkout / SOURCE_ROOT / "src" / "new-panel.ts"
    binary = checkout / SOURCE_ROOT / "public" / "binary.dat"
    binary.parent.mkdir(exist_ok=True)
    binary.write_bytes(b"\0\r\n\xff")
    for root in (checkout / SOURCE_ROOT / "node_modules", checkout / SOURCE_ROOT / "dist"):
        root.mkdir()
        (root / "local").write_bytes(b"keep")
    text.write_bytes(b"export {}\r\n")
    (checkout / "NOTICE").write_bytes(b"Notice\r\n")
    inputs = builder._inputs(checkout)
    assert inputs[f"{SOURCE_ROOT}/src/new-panel.ts"] == b"export {}\n"
    assert inputs[f"{SOURCE_ROOT}/public/binary.dat"] == binary.read_bytes()
    assert inputs["NOTICE"] == b"Notice\n"
    assert not any("node_modules/" in path or "dist/" in path for path in inputs)
    assert text.read_bytes() == b"export {}\r\n"
    assert (checkout / SOURCE_ROOT / "node_modules/local").read_bytes() == b"keep"


@pytest.mark.parametrize("attribute", ["-text", "binary"])
def test_canonical_build_preserves_explicit_binary_assets(checkout, monkeypatch, attribute):
    public = checkout / SOURCE_ROOT / "public"
    public.mkdir(exist_ok=True)
    (public / ".gitattributes").write_text(f'"signed payload.bin" {attribute}\n')
    payload = b"SIGNED-PAYLOAD\r\n"
    asset = public / "signed payload.bin"
    asset.write_bytes(payload)
    compile = builder._build_frontend

    def compile_with_asset(stage):
        compile(stage)
        staged_asset = stage / SOURCE_ROOT / "public" / asset.name
        assert staged_asset.read_bytes() == payload
        shutil.copyfile(staged_asset, stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT / asset.name)

    monkeypatch.setattr(builder, "_build_frontend", compile_with_asset)
    assert builder.main([]) == 0
    assert asset.read_bytes() == payload
    assert (checkout / "src" / "cayu" / COMPILED_PACKAGE_ROOT / asset.name).read_bytes() == payload
    bundle = next((checkout / "src" / "cayu" / BUNDLE_PACKAGE_ROOT).glob("*.zip"))
    with zipfile.ZipFile(io.BytesIO(bundle.read_bytes())) as archive:
        assert archive.read(f"public/{asset.name}") == payload
    assert builder.main(["--check"]) == 0


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [("docs", "docs/nested/index.txt"), ("docs/nested/index.txt", "docs")],
)
def test_regeneration_handles_asset_path_type_changes(checkout, monkeypatch, old_name, new_name):
    assets = checkout / "src" / "cayu" / COMPILED_PACKAGE_ROOT
    compiled = tree_contents(assets)
    builder._write_files(assets, {old_name: b"old"})
    # Directory-to-file replacement must also remove empty nested directories.
    if "/" in old_name:
        (assets / "docs" / "empty" / "nested").mkdir(parents=True)
    original = builder._outputs(checkout)
    compiled[new_name] = b"new"
    monkeypatch.setattr(
        builder,
        "_build_frontend",
        lambda stage: builder._write_files(
            stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT, compiled
        ),
    )
    assert builder.main(["--check"]) == 1
    assert builder._outputs(checkout) == original
    assert (assets / old_name).read_bytes() == b"old"
    assert builder.main([]) == 0
    assert tree_contents(assets) == compiled
    assert (assets / new_name).read_bytes() == b"new"
    assert builder.main(["--check"]) == 0


@pytest.mark.parametrize(
    ("old_directory", "new_directory"),
    [("guide", "Guide"), ("nested/guide", "nested/Guide"), ("guide/nested", "Guide/nested")],
)
def test_regeneration_preserves_asset_directory_capitalization(
    checkout, monkeypatch, old_directory, new_directory
):
    assets = checkout / "src" / "cayu" / COMPILED_PACKAGE_ROOT
    old_name = f"{old_directory}/index.txt"
    new_name = f"{new_directory}/index.txt"
    builder._write_files(assets, {old_name: b"old", "nested/keep.txt": b"retained sibling"})
    (assets / old_directory / "empty" / "nested").mkdir(parents=True)
    compiled = tree_contents(assets)
    del compiled[old_name]
    compiled[new_name] = b"new"
    monkeypatch.setattr(
        builder,
        "_build_frontend",
        lambda stage: builder._write_files(
            stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT, compiled
        ),
    )
    original = builder._outputs(checkout)
    paths = {path.relative_to(assets).as_posix() for path in assets.rglob("*")}
    assert builder.main(["--check"]) == 1
    assert builder._outputs(checkout) == original
    assert {path.relative_to(assets).as_posix() for path in assets.rglob("*")} == paths
    assert builder.main([]) == 0
    assert tree_contents(assets) == compiled
    directories = {
        path.relative_to(assets).as_posix() for path in assets.rglob("*") if path.is_dir()
    }
    assert new_directory in directories
    assert old_directory not in directories
    bundle = next((checkout / "src" / "cayu" / BUNDLE_PACKAGE_ROOT).glob("*.zip"))
    validate_dashboard_build(
        tree_contents(assets),
        bundle.read_bytes(),
        package_version=builder.project_version(checkout),
    )
    assert builder.main(["--check"]) == 0


def test_crlf_checkout_produces_the_same_bundle_without_rewriting_inputs(checkout):
    assert builder.main([]) == 0
    expected = builder._outputs(checkout)
    for name, content in builder._inputs(checkout).items():
        if b"\0" not in content:
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            (checkout / name).write_bytes(content.replace(b"\n", b"\r\n"))
    assert b"\r\n" in (checkout / "NOTICE").read_bytes()
    assert builder.main(["--check"]) == 0
    assert builder._outputs(checkout) == expected
    assert b"\r\n" in (checkout / "NOTICE").read_bytes()


@pytest.mark.parametrize(
    "failure", ["compiler", "validation", "concurrent-input", "concurrent-output"]
)
def test_build_failures_do_not_publish_outputs(checkout, monkeypatch, failure):
    original = builder._outputs(checkout)
    compile = builder._build_frontend

    def fail(stage):
        compile(stage)
        if failure == "compiler":
            raise subprocess.CalledProcessError(1, ["docker", "run"])
        if failure == "validation":
            (stage / "src" / "cayu" / COMPILED_PACKAGE_ROOT / "LICENSE").unlink()
        if failure == "concurrent-input":
            (checkout / SOURCE_ROOT / "README.md").write_bytes(b"new source")
        if failure == "concurrent-output":
            (checkout / "src" / "cayu" / COMPILED_PACKAGE_ROOT / "index.html").write_bytes(
                b"new output"
            )

    monkeypatch.setattr(builder, "_build_frontend", fail)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        builder.main([])
    if failure == "concurrent-output":
        original[f"src/cayu/{COMPILED_PACKAGE_ROOT}/index.html"] = b"new output"
    assert builder._outputs(checkout) == original


@pytest.mark.parametrize("location", ["source-file", "source-root", "output-file", "output-root"])
def test_builder_rejects_symlinks_without_modifying_the_target(checkout, location):
    source = checkout / SOURCE_ROOT
    compiled = checkout / "src" / "cayu" / COMPILED_PACKAGE_ROOT
    path = {
        "source-file": source / "README.md",
        "source-root": source,
        "output-file": compiled / "index.html",
        "output-root": compiled,
    }[location]
    target = checkout / "saved"
    path.rename(target)
    try:
        path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError):
        builder.main([])
    assert target.exists()


@pytest.mark.parametrize(
    "damage", ["missing-license", "stale-compiled", "extra-compiled", "bundle"]
)
def test_staged_release_checks_reject_corrupt_outputs(damage):
    package = files("cayu")
    compiled = tree_contents(Path(str(package.joinpath(*COMPILED_PACKAGE_ROOT.split("/")))))
    bundle_root = package.joinpath(*BUNDLE_PACKAGE_ROOT.split("/"))
    bundle = next(path for path in bundle_root.iterdir() if path.name.endswith(".zip")).read_bytes()
    if damage == "missing-license":
        compiled.pop("LICENSE")
    elif damage == "stale-compiled":
        compiled["index.html"] = b"stale"
    elif damage == "extra-compiled":
        compiled["unexpected.js"] = b"extra"
    else:
        bundle = b"broken"
    with pytest.raises(ValueError):
        validate_dashboard_build(compiled, bundle, package_version=builder.project_version(_ROOT))


@pytest.mark.parametrize("interruption", ["timeout", "failure", "interrupt"])
def test_container_failure_always_removes_only_its_own_container(
    tmp_path, monkeypatch, interruption
):
    calls = []
    error = {
        "timeout": subprocess.TimeoutExpired("docker", 1200),
        "failure": subprocess.CalledProcessError(1, "docker"),
        "interrupt": KeyboardInterrupt(),
    }[interruption]

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "run":
            raise error

    monkeypatch.setattr(builder.subprocess, "run", run)
    with pytest.raises(type(error)):
        builder._build_frontend(tmp_path)
    command, options = calls[0]
    name = command[command.index("--name") + 1]
    assert name.startswith("cayu-dashboard-")
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert any("@sha256:" in arg for arg in command)
    assert options["timeout"] == 1200
    assert calls[-1][0] == ["docker", "rm", "--force", name]
    assert "prune" not in repr(calls)


def test_cleanup_timeout_preserves_the_original_build_error(tmp_path, monkeypatch, capsys):
    failure = subprocess.CalledProcessError(7, ["docker", "run"])

    def run(command, **kwargs):
        if command[1] == "run":
            raise failure
        raise subprocess.TimeoutExpired(command, 15)

    monkeypatch.setattr(builder.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        builder._build_frontend(tmp_path)
    assert caught.value is failure
    assert "Could not remove temporary build container cayu-dashboard-" in capsys.readouterr().err


def test_all_hashed_dashboard_inputs_have_lf_checkout_attributes():
    names = [
        *BUILD_INPUT_FILES,
        *[
            f"{SOURCE_ROOT}/{name}"
            for name in tree_contents(_ROOT / SOURCE_ROOT, ignore_build_products=True)
        ],
    ]
    result = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "eol"],
        cwd=_ROOT,
        input="\0".join(names) + "\0",
        text=True,
        capture_output=True,
        check=True,
    )
    records = result.stdout.rstrip("\0").split("\0")
    assert len(records) == len(names) * 3
    assert all(records[index] == "lf" for index in range(2, len(records), 3))
