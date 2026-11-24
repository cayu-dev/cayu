"""SQLAlchemy declarative models, as `cayu new --with app-database` generates them, pass the
import check; anything else in a model body stays unproven."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_check import check_declared_scaffold_source

_BASE = """from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    MetaData,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "app"


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA)
"""

_ACCEPTED = {
    "generated_models": project_files("probe", with_capabilities=("app-database",))[
        "database/models.py"
    ],
    "related_models": _BASE
    + '''

class Organization(Base):
    """A business this application serves."""

    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Membership(Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("subject", "organization_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    subject: Mapped[str] = mapped_column(String(200), index=True)
    organization_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.organizations.id", ondelete="CASCADE")
    )
    department_id: Mapped[str | None] = mapped_column(
        ForeignKey("app.departments.id", ondelete="SET NULL")
    )
    revision: Mapped[int] = mapped_column(BigInteger, default=0)
    note: Mapped[str | None]

    def label(self) -> str:
        return f"{self.subject}@{self.organization_id}"

    def __repr__(self) -> str:
        return self.label()


class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["organization_id", "department_id"],
            [f"{SCHEMA}.departments.organization_id", f"{SCHEMA}.departments.id"],
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64))
    department_id: Mapped[str | None] = mapped_column(String(64))
''',
    "empty_base": (
        "from sqlalchemy.orm import DeclarativeBase\n\n\nclass Base(DeclarativeBase):\n    pass\n"
    ),
    "evaluated_annotations": _BASE.replace("from __future__ import annotations\n\n", "")
    + """

class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    body: Mapped[str | None] = mapped_column(String(2000), nullable=True)
""",
}

_SHARED_COLUMNS = '''from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, MetaData, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "app"
NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA, naming_convention=NAMING)
    type_annotation_map = {dict[str, Any]: JSONB, list[str]: ARRAY(Text)}


class Timestamps:
    """Columns every table shares."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Upload(Timestamps, Base):
    __tablename__ = "uploads"
    __table_args__ = (Index(None, "owner", "created_at"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    owner: Mapped[str] = mapped_column(String(200), index=True)
    tags: Mapped[list[str]]
    details: Mapped[dict[str, Any] | None]
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")


class UploadNote(Base):
    __tablename__ = "upload_notes"

    upload_id: Mapped[str] = mapped_column(
        ForeignKey(f"{SCHEMA}.uploads.id", ondelete="CASCADE"), primary_key=True
    )
    labels: Mapped[list[str]] = mapped_column(ARRAY(String(40)))
'''
_ACCEPTED["mixins_naming_and_type_map"] = _SHARED_COLUMNS
_ACCEPTED["dialect_module_types"] = (
    _BASE.replace(
        "from sqlalchemy.orm import",
        "from sqlalchemy.dialects import postgresql\nfrom sqlalchemy.orm import",
    )
    + """

class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    body: Mapped[dict] = mapped_column(postgresql.JSONB)
    tags: Mapped[list[str]] = mapped_column(postgresql.ARRAY(String(40)))
"""
)

_REJECTED = {
    # A column default the proof can't evaluate as data.
    "computed_default": (
        _BASE + "\n\ndef start() -> int:\n    return 1\n\n\nclass Row(Base):\n"
        '    __tablename__ = "rows"\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True, default=start())\n",
        "start(...)",
    ),
    # Any other call in a model body.
    "class_body_call": (
        _BASE + "\n\ndef setup() -> int:\n    return 1\n\n\nclass Row(Base):\n"
        '    __tablename__ = "rows"\n    marker = setup()\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n",
        "setup(...)",
    ),
    # Table arguments are literal constraints only.
    "table_args_call": (
        _BASE + "\n\ndef constraint() -> object:\n    return None\n\n\nclass Row(Base):\n"
        '    __tablename__ = "rows"\n    __table_args__ = (constraint(),)\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n",
        None,
    ),
    # Mapper configuration hooks and options stay unproven.
    "mapper_args": (
        _BASE + '\n\nclass Row(Base):\n    __tablename__ = "rows"\n'
        '    __mapper_args__ = {"eager_defaults": True}\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n",
        None,
    ),
    # A mixin could bring its own hooks into the mapped class.
    "mixin_base": (
        _BASE + "\n\nclass Audited:\n    pass\n\n\nclass Row(Audited, Base):\n"
        '    __tablename__ = "rows"\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n",
        None,
    ),
    # A mixin's body is held to a model's rules: no hook SQLAlchemy would run on a subclass.
    "mixin_hook": (
        _BASE + "\n\nclass Audited:\n"
        "    revision: Mapped[int] = mapped_column(default=0)\n\n"
        "    def __init_subclass__(cls, **kwargs):\n        return None\n\n\n"
        'class Row(Audited, Base):\n    __tablename__ = "rows"\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n",
        None,
    ),
    "mixin_call": (
        _BASE + "\n\ndef start() -> int:\n    return 1\n\n\nclass Audited:\n"
        "    revision: Mapped[int] = mapped_column(default=start())\n",
        "start(...)",
    ),
    # The type map names reviewed column types only.
    "type_map_call": (
        _BASE + "\n    type_annotation_map = {dict: make_type()}\n\n\n"
        "def make_type() -> object:\n    return None\n",
        None,
    ),
    # A naming convention changed in place at import no longer holds what it was written with.
    "augmented_naming_convention": (
        _BASE.replace(
            "    metadata = MetaData(schema=SCHEMA)",
            "    metadata = MetaData(schema=SCHEMA, naming_convention=NAMING)",
        ).replace('SCHEMA = "app"\n', 'SCHEMA = "app"\nNAMING = {"pk": "pk_%(table_name)s"}\n')
        + '\n\nNAMING |= {"ix": "ix_%(column_0_label)s"}\n',
        None,
    ),
    # SQLAlchemy calls these while it declares the class.
    "table_class_hook": (
        _BASE + '\n\nclass Row(Base):\n    __tablename__ = "rows"\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n\n"
        "    @classmethod\n    def __table_cls__(cls, *args, **kwargs):\n"
        "        return None\n",
        None,
    ),
    "mapper_class_hook": (
        _BASE + '\n\nclass Row(Base):\n    __tablename__ = "rows"\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n\n"
        "    def __mapper_cls__(self, *args, **kwargs):\n        return None\n",
        None,
    ),
    # A decorated method can be a descriptor SQLAlchemy runs while it reads the class.
    "descriptor_on_base": (
        "from sqlalchemy.orm import DeclarativeBase\n\n\nclass Base(DeclarativeBase):\n"
        "    @classmethod\n    @property\n    def metadata(cls):\n        return None\n",
        None,
    ),
    "property_on_model": (
        _BASE + '\n\nclass Row(Base):\n    __tablename__ = "rows"\n\n'
        "    id: Mapped[int] = mapped_column(primary_key=True)\n\n"
        "    @property\n    def label(self) -> str:\n        return str(self.id)\n",
        None,
    ),
    # A method by a name SQLAlchemy reads while it declares the class.
    "registry_method": (
        "from sqlalchemy.orm import DeclarativeBase\n\n\nclass Base(DeclarativeBase):\n"
        "    def registry(self):\n        return None\n",
        None,
    ),
    "base_declaration_hook": (
        _BASE + "\n    @classmethod\n    def __declare_first__(cls):\n        return None\n",
        None,
    ),
    # Metadata options are literals or literal constants.
    "metadata_from_environment": (
        _BASE.replace(
            "    metadata = MetaData(schema=SCHEMA)",
            '    metadata = MetaData(schema=os.environ["SCHEMA"])',
        ).replace("from datetime import datetime\n", "import os\nfrom datetime import datetime\n"),
        None,
    ),
}


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("sqlalchemy-models")
    assert main(["new", "probe", "--with", "app-database", "--dir", str(root)]) == 0
    return root / "probe"


def _findings(project: Path, source: str) -> list:
    module = project / "tools" / "candidate.py"
    module.write_text(source, encoding="utf-8")
    try:
        return [
            diagnostic
            for diagnostic in check_declared_scaffold_source(project)
            if diagnostic.path.startswith("tools/candidate.py")
        ]
    finally:
        module.unlink()


@pytest.mark.parametrize("name", sorted(_ACCEPTED))
def test_declarative_models_pass_the_import_check(project: Path, name: str) -> None:
    assert _findings(project, _ACCEPTED[name]) == []


@pytest.mark.parametrize("name", sorted(_REJECTED))
def test_anything_else_in_a_model_stays_unproven(project: Path, name: str) -> None:
    source, construct = _REJECTED[name]

    findings = _findings(project, source)

    assert [finding.code for finding in findings] == ["SCAFFOLD_IMPORT_SIDE_EFFECT"]
    if construct is not None:
        assert findings[0].parameters["construct"] == construct


def test_an_agent_that_uses_its_tables_passes_cayu_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["new", "probe", "--with", "app-database", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "probe"
    registration = project / "tools" / "registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            '"""\n\n', '"""\n\nfrom database.models import Note  # noqa: F401\n', 1
        ),
        encoding="utf-8",
    )
    assert "from database.models import Note" in registration.read_text(encoding="utf-8")
    capsys.readouterr()
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


_TYPED_IMPORT = """from __future__ import annotations

{flag}

if TYPE_CHECKING:
    from database.effectful import Effect


def describe(effect: Effect) -> str:
    return str(effect)
"""


@pytest.mark.parametrize(
    ("flag", "followed"),
    [
        ("from typing import TYPE_CHECKING", False),
        ("import typing_extensions as typing\nTYPE_CHECKING = typing.TYPE_CHECKING", True),
        ("from typing import TYPE_CHECKING\n\nTYPE_CHECKING = True", True),
    ],
)
def test_imports_under_type_checking_are_not_followed(
    project: Path, flag: str, followed: bool
) -> None:
    # Outside the application packages, so only an import brings it into the check.
    effectful = project / "database" / "effectful.py"
    effectful.write_text(
        'from pathlib import Path\n\nPath("marker").write_text("x")\n\n\nclass Effect:\n    pass\n',
        encoding="utf-8",
    )
    module = project / "tools" / "candidate.py"
    module.write_text(_TYPED_IMPORT.format(flag=flag), encoding="utf-8")
    try:
        paths = {
            diagnostic.path.partition(":")[0]
            for diagnostic in check_declared_scaffold_source(project)
            if diagnostic.code == "SCAFFOLD_IMPORT_SIDE_EFFECT"
        }
    finally:
        module.unlink()
        effectful.unlink()

    assert ("database/effectful.py" in paths) is followed


def test_typing_with_the_models_from_application_code_passes_cayu_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["new", "probe", "--with", "app-database", "--dir", str(tmp_path)]) == 0
    project = tmp_path / "probe"
    (project / "database" / "models.py").write_text(_SHARED_COLUMNS, encoding="utf-8")
    (project / "operations" / "uploads.py").write_text(
        '"""Read uploads."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import TYPE_CHECKING\n\n"
        "from database.models import Upload\n\n"
        "if TYPE_CHECKING:\n"
        "    from database.session import Session\n\n\n"
        "def owner(db: Session, upload_id: str) -> str | None:\n"
        "    upload = db.get(Upload, upload_id)\n"
        "    return None if upload is None else upload.owner\n",
        encoding="utf-8",
    )
    registration = project / "tools" / "registration.py"
    registration.write_text(
        registration.read_text(encoding="utf-8").replace(
            '"""\n\n', '"""\n\nfrom operations.uploads import owner  # noqa: F401\n', 1
        ),
        encoding="utf-8",
    )
    assert "from operations.uploads import owner" in registration.read_text(encoding="utf-8")
    capsys.readouterr()
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []
