"""Compose frozen tool declarations into the common environment admission policy."""

from cayu.environments.admission import ExecutionRequirements, ExecutionToolRequirement
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime.sessions import Session
from cayu.runtime.tool_exposure import tool_capability_ceiling_from_session_metadata


def effective_execution_requirements(
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
) -> ExecutionRequirements:
    """Require every tool the durable ceiling permits this invocation to dispatch."""

    base = registered_agent.execution_requirements
    if not any(tool.execution_requirements for tool in registered_agent.tools.values()):
        return ExecutionRequirements.model_validate(base.model_dump(mode="python", warnings=False))
    ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
    clauses = {(item.tool_name, item.requirement.name): item for item in base.tool_requirements}
    for name in ceiling.tool_names:
        tool = registered_agent.tools.get(name)
        if tool is None:
            raise ValueError("The admitted tool ceiling references an unavailable registration.")
        for requirement in tool.execution_requirements:
            clause = ExecutionToolRequirement(tool_name=name, requirement=requirement)
            key = (name, requirement.name)
            previous = clauses.setdefault(key, clause)
            if previous != clause:
                raise ValueError("A tool declaration conflicts with a caller-owned requirement.")
            if len(clauses) > 64:
                raise ValueError("The admitted tool set exceeds the execution requirement bound.")
    values = base.model_dump(mode="python", warnings=False)
    values["tool_requirements"] = tuple(clauses[key] for key in sorted(clauses))
    return ExecutionRequirements.model_validate(values)
