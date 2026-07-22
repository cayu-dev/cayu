# Cayu project diagnostics

`cayu check` renders these stable findings from the public application manifest.
Run the correction, then rerun `cayu inspect --json` and `cayu check --json`.

## app-no-agents

`APP_NO_AGENTS` means the factory returned an app with no registered agent.
Register an `AgentSpec` with `CayuApp.register_agent()`.

## agent-generated-tracer-bullet-unfinished

`AGENT_GENERATED_TRACER_BULLET_UNFINISHED` means a generator left the agent's
explicit authoring-state marker in place. The generated prompt, placeholder
tool behavior, runtime test, and scripted trajectory are a runnable wiring
proof, not evidence that the requested domain behavior is complete.

Replace the domain system prompt, tool schema and implementation, runtime test
inputs and assertions, and trajectory eval behavior and assertions. For a
scaffold updated by `cayu generate tool`, change `_AUTHORING_STATE` to `None`
inside the generated agent-config region. For `cayu generate slice`, remove
`authoring_state=AgentAuthoringState.UNFINISHED_GENERATED_TRACER_BULLET` and
the unused import from the generated agent module. Verify with
`cayu inspect --json && cayu check --fail-on warning --json`.

The marker is an author assertion, not source analysis: Cayu does not scan
Python or prose for words such as `sample`, `echo`, or `tracer bullet`, and an
absent marker does not prove domain correctness.

## agent-provider-not-found

`AGENT_PROVIDER_NOT_FOUND` means an agent's explicit provider is absent, or no
model-pattern/default provider can resolve the agent. Register the named
provider, correct `provider_name`, or define an unambiguous default route.

## agent-provider-ambiguous

`AGENT_PROVIDER_AMBIGUOUS` means more than one registered provider pattern
matches the agent model. Make patterns disjoint or set the agent's
`provider_name` explicitly.

## agent-workflow-tool-not-registered

`AGENT_WORKFLOW_TOOL_NOT_REGISTERED` means an agent's explicit
`workflow_tool_names` contract names a tool that is not registered for that
same agent. Use the exact registered name, update the machine-owned tool-name
source after a rename, or register the intended tool. Cayu checks this explicit
contract and does not parse arbitrary natural-language prompt text.

## external-tool-unguarded

`EXTERNAL_TOOL_UNGUARDED` means a tool declaring `ToolEffect.EXTERNAL` is under
a policy that can allow that specific tool without an enforcing boundary. The
diagnostic reports the effective per-tool coverage rather than trusting the
policy class name. Register an enforcing policy. Use
`AlwaysRequireApprovalToolPolicy(tools=[...])` when a human must authorize
execution, and include the external tool's actual name in its scope.

## external-tool-coverage-unknown

`EXTERNAL_TOOL_COVERAGE_UNKNOWN` means an external-effect tool uses a custom or
otherwise unrecognized policy whose behavior Cayu cannot verify statically.
This remains an error rather than an acknowledgment-based bypass: use a
statically describable enforcing policy until Cayu provides a trusted custom
coverage contract.

## tool-input-schema-unconstrained

`TOOL_INPUT_SCHEMA_UNCONSTRAINED` means a registered tool exposes `{}` as its
input schema. That is valid JSON Schema, but it accepts every JSON value and
does not teach the model which arguments to send. Declare the expected object
properties, required fields, and `additionalProperties` behavior in
`ToolSpec.input_schema`. If a tool derives its schema dynamically, override the
public `Tool.schema` property; Cayu treats that property as authoritative when
the tool is registered.

Tool implementations must also declare `run` with `async def`. Cayu validates
that contract during agent registration so a synchronous implementation fails
before a session starts.

Inspection and checks are structural. Clearing all diagnostics does not prove a
provider credential, remote service, sandbox, network path, or deployment is
live. The manifest reports `has_system_prompt` but never prompt text; its
fingerprint records prompt presence, not prompt contents. A prompt edit between
two non-empty values therefore needs a runtime test or eval for verification.
