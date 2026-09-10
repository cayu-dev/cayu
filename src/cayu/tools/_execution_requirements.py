from __future__ import annotations

from cayu.core.tools import ToolExecutionRequirement, ToolSpec


def with_intrinsic_execution_requirements(
    spec: ToolSpec, intrinsic: tuple[ToolExecutionRequirement, ...]
) -> ToolSpec:
    """Preserve implementation dependencies when customizing a built-in spec."""
    if not isinstance(spec, ToolSpec):
        raise TypeError("Tool spec must be a ToolSpec.")
    spec = ToolSpec.model_validate(spec.model_dump(mode="python", warnings=False))
    requirements = {item.name: item for item in spec.execution_requirements}
    for requirement in intrinsic:
        previous = requirements.setdefault(requirement.name, requirement)
        if previous != requirement:
            raise ValueError("Intrinsic execution requirement conflicts with supplied tool spec.")
    # ToolSpec.model_copy validates the merged set, including aggregate limits.
    return spec.model_copy(
        update={
            "execution_requirements": tuple(requirements[name] for name in sorted(requirements))
        }
    )
