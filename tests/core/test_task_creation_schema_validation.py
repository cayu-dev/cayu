"""Task receipt catalog validation across PostgreSQL catalog representations."""

import asyncio

import pytest

from cayu.storage import _postgres_schema_history, _postgres_task_schema


@pytest.mark.parametrize(
    "extra_constraint", [None, ("f", "FOREIGN KEY (task_id) REFERENCES cayu_tasks(id)")]
)
def test_task_receipt_validator_ignores_pg18_not_null_but_rejects_foreign_key(extra_constraint):
    class CatalogCursor:
        async def execute(self, query):
            self.query = query

        async def fetchall(self):
            if "information_schema.columns" in self.query:
                return [("task_id", "text", "NO", "C"), ("request_sha256", "text", "NO", "C")]
            rows = [
                ("c", "CHECK ((request_sha256 ~ '^[0-9a-f]{64}$'::text))"),
                ("p", "PRIMARY KEY (task_id)"),
            ]
            if "contype IN" not in self.query:
                rows.append(("n", "NOT NULL task_id"))
            if extra_constraint is not None:
                rows.append(extra_constraint)
            return sorted(rows)

        async def fetchone(self):
            if self.query.removeprefix("SELECT ") == "current_schema()":
                return ("public",)
            return (
                7,
                "O",
                None,
                "",
                0,
                _postgres_schema_history._MIGRATION_STEPS[127][1].split("$$")[1],
                False,
                "public",
            )

    async def run():
        operation = _postgres_task_schema._validate_task_creation_operations(
            CatalogCursor(),
            expected_guard_sql=_postgres_schema_history._MIGRATION_STEPS[127][1],
        )
        if extra_constraint is None:
            await operation
        else:
            with pytest.raises(RuntimeError, match="constraints"):
                await operation

    asyncio.run(run())


def test_creation_validator_reference_harness_supplies_its_own_revision_guard():
    from tests.core.test_postgres_schema_owners import _arguments

    assert _arguments(_postgres_task_schema._validate_task_creation_operations) == {
        "expected_guard_sql": _postgres_schema_history._MIGRATION_STEPS[127][1],
    }


@pytest.mark.parametrize(
    "schema",
    [
        {"properties": {"document": {"$ref": "https://example.invalid/schema"}}},
        {"$defs": {"document": {"$schema": "https://example.invalid/dialect"}}},
    ],
)
def test_actual_nested_schema_keywords_remain_restricted(schema):
    from cayu.tasks import TaskTypeSpec

    with pytest.raises(ValueError):
        TaskTypeSpec(name="document", input_schema=schema)
