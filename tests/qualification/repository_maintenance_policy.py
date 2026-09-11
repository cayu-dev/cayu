"""Application-specific parameter rules; Runtime retains dispatch ownership."""

from typing import Any

from cayu import ParameterRule
from tests.qualification.repository_maintenance_case import ALLOWED_CHANGE_PATHS


class MaintenancePatchScopeRule(ParameterRule):
    @property
    def parameter(self) -> str:
        return "operations"

    def check(self, arguments: dict[str, Any]) -> str | None:
        operations = arguments.get("operations")
        if type(operations) is not list or not 1 <= len(operations) <= 2:
            return "Maintenance patches require one or two scoped operations."
        for operation in operations:
            if type(operation) is not dict:
                return "Maintenance patch operation must be an object."
            kind = operation.get("type")
            if type(kind) is not str or kind not in {"create", "update", "delete", "move"}:
                return "Maintenance patch operation type is not admitted."
            paths = ("from_path", "to_path") if kind == "move" else ("path",)
            if set(operation) & {"path", "from_path", "to_path"} != set(paths):
                return "Maintenance patch path authority is incomplete."
            if any(
                type(operation.get(key)) is not str or operation[key] not in ALLOWED_CHANGE_PATHS
                for key in paths
            ):
                return "Maintenance patch escapes the admitted paths."
        return None


class MaintenanceWriteBoundRule(ParameterRule):
    @property
    def parameter(self) -> str:
        return "max_bytes"

    def check(self, arguments: dict[str, Any]) -> str | None:
        maximum = arguments.get("max_bytes")
        if type(maximum) is not int or not 1 <= maximum <= 16 * 1024:
            return "Maintenance writes require explicit max_bytes between 1 and 16384."
        return None
