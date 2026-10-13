from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from cayu.evals import corpus

_ROOT = Path(__file__).parents[2]
_SCRIPT = _ROOT / "dashboard/scripts/generate-api-types.py"


def test_dashboard_text_limits_are_generated_from_the_domain(monkeypatch):
    namespace = runpy.run_path(str(_SCRIPT))
    assert namespace["EVAL_TEXT_LIMITS"].read_bytes() == namespace["_eval_text_limits"]()
    monkeypatch.setattr(corpus, "EVAL_CORPUS_MAX_NAME_CHARS", 37)
    monkeypatch.setattr(corpus, "EVAL_CORPUS_MAX_DESCRIPTION_CHARS", 193)
    rendered = namespace["_eval_text_limits"]().decode()
    assert "name: 37," in rendered
    assert "description: 193," in rendered
    assert f'unit: "{corpus.EVAL_CORPUS_TEXT_LENGTH_UNIT}"' in rendered


@pytest.mark.parametrize("damage", ["value", "unit", "missing"])
def test_dashboard_api_command_generates_and_checks_text_limits(tmp_path, monkeypatch, damage):
    main = runpy.run_path(str(_SCRIPT))["main"]
    namespace = main.__globals__
    generated = tmp_path / "generated/server-api"
    limits = tmp_path / "generated/eval-text-limits.ts"

    def generate(*, schema_path, output_dir):
        assert schema_path.read_text() == "{}\n"
        output_dir.mkdir(parents=True)
        (output_dir / "index.ts").write_text("export {}\n")

    replacements = {
        "GENERATED_DIR": generated,
        "OPENAPI_BASELINE": tmp_path / "openapi.json",
        "RELEASE_METADATA": tmp_path / "release-metadata.ts",
        "EVAL_TEXT_LIMITS": limits,
        "_openapi_json": lambda: "{}\n",
        "_run_generator": generate,
        "_assert_expected_cayu_import": lambda: None,
    }
    for key, value in replacements.items():
        monkeypatch.setitem(namespace, key, value)
    assert main([]) == 0
    assert main(["--check"]) == 0

    if damage == "missing":
        limits.unlink()
    else:
        original = limits.read_text()
        changed = (
            original.replace("name: 256", "name: 255")
            if damage == "value"
            else original.replace("unicode_scalar_values", "utf16_code_units")
        )
        assert changed != original
        limits.write_text(changed)
    with pytest.raises(SystemExit, match="evaluation text limits are stale"):
        main(["--check"])
    assert main([]) == 0
    assert main(["--check"]) == 0
