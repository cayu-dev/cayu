# Named checks

`RunCheckTool` gives an agent a finite application-owned check surface without
exposing shell text, process argv, working-directory selection, environment,
stdin, timeouts, output limits, images, networks, or runners as model input.
Use it when the application knows the exact deterministic checks that are safe
to offer for a trusted workspace.

```python
from cayu import (
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    NamedCheck,
    ProcessCommandPolicy,
    RunCheckTool,
)

unit = NamedCheck(
    name="test",
    description="Run the deterministic unit test suite.",
    command=ExecCommand.process("uv", "run", "pytest", "-q"),
    timeout_s=120,
    max_output_bytes=100_000,
    required_executables=("uv", "pytest"),
    execution_profile_identity=ExecutionProfileBehaviorIdentity(
        name="project-unit-check",
        behavior_version="1",
        implementation_version="2026-08-28",
    ),
)

run_check = RunCheckTool(
    checks=(unit,),
    command_policy=ProcessCommandPolicy(
        allowed_executables=("uv",),
        allowed_cwds=("/workspace",),
        max_timeout_s=120,
    ),
)
```

The exposed JSON schema contains one required `check` string enum and rejects
additional properties. A `NamedCheck` accepts only process-form `ExecCommand`
values, snapshots argv immutably, and validates all declaration and result
bounds during construction. Check order is canonicalized by name.

Three independent gates remain in force:

1. The agent's ordinary `ToolPolicy` authorizes `run_check` and its `check`
   argument. Use `RequiredAllowlistRule` to deny unknown names statically, or a
   `ToolPolicyDecision.REQUIRE_APPROVAL` policy for a durable pause/resume
   checkpoint.
2. `RunCheckTool` resolves the selected declaration and passes its exact
   process command through the mandatory `CommandPolicy`, canonical runner cwd,
   runner preflight, and execution boundary. A
   `CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL` is an inline refusal, not a
   durable approval checkpoint.
3. Environment admission must establish that the final runner/workspace can
   satisfy each declaration's `required_executables`. The declaration and
   manifest are requirements, not proof that a host `PATH`, image name, or
   runner contains those programs.

The structured result distinguishes `passed`, `failed`, `timed_out`,
`cancelled`, `runner_unavailable`, `policy_denied`, `approval_required`, typed
runner execution failure, and malformed runner evidence. A nonzero exit is a
completed failing check (`is_error=false`) so the model can inspect and repair
the workspace. Timeout and cancellation remain error results. Runner cleanup
evidence is retained separately as `workspace_mutation_settlement`.

Runner capture is capped by the check's `max_output_bytes`. The model projection
has an additional per-stream bound. When that smaller projection truncates and
an `ArtifactStore` is configured, Cayu stores the captured redacted stdout and
stderr as a session-scoped JSON artifact with digest metadata; the structured result
retains previews, independent runner/projection truncation flags, byte totals
when authoritative, the output digest, and artifact status.

The effective execution profile includes the canonical check declarations,
command digests, limits, executable requirements, declaration identities,
result-projection behavior, and the attached command policy. Reconstructing an
active durable session with behavior-changing check or policy configuration is
therefore rejected unless the caller explicitly adopts the new profile through
Cayu's normal execution-profile contract.

Named checks execute repository-controlled code. They do not make a local
runner or an ordinary Docker container safe for hostile repositories, grant
network or credentials, install dependencies, or authorize source publication.
