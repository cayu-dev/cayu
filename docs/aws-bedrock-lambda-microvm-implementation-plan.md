# AWS Bedrock and Lambda MicroVM design record

Issues: [#212](https://github.com/vertexkg/cayu/issues/212), [#78](https://github.com/vertexkg/cayu/issues/78)  

## Outcome

The design consists of two independent, first-party AWS adapters:

1. `BedrockProvider`, which runs Anthropic Claude models through Amazon Bedrock's
   `ConverseStream` interface while preserving Cayu's provider-neutral transcript,
   streaming, tool, structured-output, usage, and typed-error contracts.
2. `LambdaMicroVMRunner`, which runs Cayu `ExecCommand` requests in an AWS Lambda
   MicroVM through a versioned command sidecar and composes with the existing
   `RunnerWorkspace` and environment-factory contracts.

The implementations share only an optional AWS SDK dependency. They do not introduce a
generic AWS framework or couple model inference to sandbox execution.

## Decisions

### Use the official Boto3 clients behind injected seams

- Add an `aws` optional extra with `boto3>=1.43.44,<2`. This version contains both
  `bedrock-runtime.count_tokens` and the new `lambda-microvms` client.
- Construct SDK clients lazily so `import cayu` continues to work without the AWS extra.
- Accept injected clients in both adapters. Unit tests use fakes and never require AWS
  credentials or network access.
- Use the official AWS credential chain (environment, profile, workload identity, instance
  role, and so on). Do not accept or copy raw access-key fields into provider/runner events or
  reconnect metadata.
- Reject ambiguous configuration such as passing an injected client together with
  `profile_name` or `endpoint_url`.

### Bedrock uses ConverseStream, not the Anthropic direct adapter

`BedrockProvider` is a distinct provider adapter with `name="bedrock"`. It will not infer a
provider from a model-name prefix, mutate `AnthropicProvider`, or silently fall back to the
direct Anthropic endpoint.

The first implementation targets Claude through Bedrock, but it accepts any explicit Bedrock
model ID or inference-profile ARN that supports `ConverseStream`. Model selection remains the
existing `ModelRequest.model` responsibility. Cayu will not guess or rewrite regional model
IDs.

Use Bedrock's normalized Converse shapes for system messages, messages, tools, tool results,
stream events, stop reasons, and token usage. Do not translate Cayu to Anthropic JSON and then
translate that JSON again to Bedrock.

### Reuse RunnerWorkspace for Lambda MicroVM files

Do not add `LambdaMicroVMWorkspace` in the first implementation. That would duplicate path
containment, bounded reads/listing, file-copy behavior, and tests already concentrated behind
`RunnerWorkspace`.

The first-party MicroVM sidecar image guarantees `python3` and `/workspace`, so the public
composition is:

```python
runner = await LambdaMicroVMRunner.create(...)
workspace = RunnerWorkspace(runner)
```

This still keeps command execution and file access inside the same MicroVM. A future native
workspace adapter is justified only if measurements show that command-backed file operations
are a material bottleneck or AWS adds a native filesystem interface.

### Use a versioned asynchronous command sidecar

A single synchronous `POST /exec` cannot satisfy Cayu's cancellation contract: cancelling the
HTTP request does not prove that the guest process stopped. The image template therefore ships
a small sidecar with this interface:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Readiness probe used after run/resume |
| `POST` | `/v1/commands` | Idempotently start a command with a caller-generated command ID |
| `GET` | `/v1/commands/{id}` | Poll state and obtain a bounded result |
| `DELETE` | `/v1/commands/{id}` | Stop the command's process group and confirm cleanup |
| `POST` | `/aws/lambda-microvms/runtime/v1/run` | Initialize per-MicroVM state |
| `POST` | `/aws/lambda-microvms/runtime/v1/resume` | Revalidate state after resume |
| `POST` | `/aws/lambda-microvms/runtime/v1/suspend` | Flush command/result state |
| `POST` | `/aws/lambda-microvms/runtime/v1/terminate` | Kill commands and flush diagnostics |

The start request carries the exact `ExecCommand` form, resolved cwd, explicit env, optional
stdin, timeout, and output limit. The sidecar:

- never inherits the image process environment into commands;
- preserves process argv without shell parsing and uses an explicit shell only for
  `ExecCommand.bash(...)`;
- validates cwd under `/workspace` again inside the guest;
- captures stdout/stderr incrementally into bounded buffers;
- returns output as base64 plus byte counts and truncation flags;
- kills a process group on timeout/cancellation, including descendants;
- keeps completed results briefly so a retried poll is idempotent.

The caller-generated command ID closes the cancellation race where the HTTP start request is
cancelled after the sidecar created the process but before the client received the response.

### Lifecycle and reconnect policy

Expose `LambdaMicroVMCloseAction = Literal["terminate", "suspend", "none"]`:

- `terminate`: final cleanup for completed/failed disposable sessions;
- `suspend`: preserve memory/disk for an interrupted session that will resume;
- `none`: detach locally when lifecycle is owned elsewhere.

`close()` is idempotent and applies the selected action once. Also expose explicit `suspend()`
and `resume()` methods for app-owned lifecycle handling.

The environment-factory recipe persists only non-secret reconnect metadata:

```json
{
  "microvm_id": "mvm-...",
  "endpoint": "...lambda-microvm...on.aws",
  "region": "us-west-2",
  "image_identifier": "arn:...",
  "image_version": "..."
}
```

It never persists the JWE endpoint token. Tokens are generated through
`create_microvm_auth_token`, cached only in memory until shortly before expiry, refreshed on
authorization failure once, and discarded on close. A resume factory attaches with
`from_existing(...)`; a fork creates a fresh MicroVM instead of inheriting the parent's
MicroVM.

The recipe uses a small lifecycle binding that delegates native bind behavior, suspends on an
`interrupted` finalize outcome, and terminates on `completed` or `failed` after workspace
finalization. This puts lifecycle cleanup at the existing binding-finalize seam and prevents a
factory-created MicroVM from leaking after a terminal session.

## Public interfaces

### BedrockProvider

Public shape:

```python
class BedrockProvider(ModelProvider):
    name = "bedrock"
    usage_dialect = UsageDialect.ANTHROPIC

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
        name: str = "bedrock",
        max_tokens: int = 4096,
        stream_idle_timeout_s: float = 120.0,
    ) -> None: ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...
    async def count_input_tokens(self, request: ModelRequest) -> InputTokenCountResult | None: ...
    async def aclose(self) -> None: ...
```

`ModelRequest.options["bedrock"]` may supply copied, validated Converse options such as
`inferenceConfig`, `additionalModelRequestFields`, `additionalModelResponseFieldPaths`,
`guardrailConfig`, `performanceConfig`, `requestMetadata`, and `serviceTier`. The adapter owns
and rejects overrides of `modelId`, `messages`, `system`, `toolConfig`, and stream mechanics.

Do not claim native structured output initially. Cayu's existing `strategy="tool"` flow must
work end to end through Bedrock and is the required structured-output acceptance path.
`supports_native_structured_output` remains false until a separate slice implements Bedrock's
schema subset, preflight rules, and model-feature compatibility without making false promises
for unsupported models.

### LambdaMicroVMRunner

Public shape:

```python
class LambdaMicroVMRunner(Runner):
    isolation = "lambda-microvm"
    default_cwd = "/workspace"

    @classmethod
    async def create(
        cls,
        image_identifier: str,
        *,
        region_name: str | None = None,
        image_version: str | None = None,
        execution_role_arn: str | None = None,
        ingress_network_connectors: list[str] | None = None,
        egress_network_connectors: list[str] | None = None,
        idle_policy: dict[str, Any] | None = None,
        maximum_duration_in_seconds: int | None = None,
        run_hook_payload: str | None = None,
        close_action: LambdaMicroVMCloseAction = "terminate",
        client: Any | None = None,
        endpoint_transport: LambdaMicroVMEndpointTransport | None = None,
        ...
    ) -> LambdaMicroVMRunner: ...

    @classmethod
    async def from_existing(
        cls,
        microvm_id: str,
        *,
        region_name: str | None = None,
        close_action: LambdaMicroVMCloseAction = "none",
        client: Any | None = None,
        endpoint_transport: LambdaMicroVMEndpointTransport | None = None,
        ...
    ) -> LambdaMicroVMRunner: ...

    async def exec(...) -> ExecResult: ...
    async def suspend(self) -> None: ...
    async def resume(self) -> None: ...
    async def close(self) -> None: ...
```

`create(...)` passes AWS-specific run options through only after copying and validating them,
then waits for the authenticated `/health` endpoint rather than trusting the eventually
consistent `get_microvm` state. If setup is cancelled or readiness fails, a newly created
MicroVM is terminated within a bounded cleanup window.

The default endpoint transport uses Cayu's shared async HTTP machinery. AWS control calls are
sync Boto3 calls and run off the event loop. The Bedrock stream bridge likewise consumes the
Boto3 event stream on a worker thread and forwards events through a bounded async queue;
cancellation closes the SDK event stream and joins the worker within a bounded timeout.

## Bedrock implementation (#212)

### Implementation

1. Add the `aws` extra and lockfile entries; keep all AWS imports lazy.
2. Add `src/cayu/providers/bedrock.py` with:
   - constructor/config validation and injected-client ownership;
   - Cayu message/tool/file projection to Converse input;
   - a non-blocking Boto3 `ConverseStream` bridge with idle timeout and cancellation cleanup;
   - stream parsing for text, reasoning/provider state where supported, fragmented tool JSON,
     stop reasons, usage, and in-stream failures;
   - `CountTokens` support using the same projected conversation input;
   - `BedrockError`, `BedrockAPIError`, `BedrockContextOverflowError`, and
     `BedrockProtocolError`.
3. Convert Bedrock usage to Cayu's canonical raw usage keys while also retaining the original
   Bedrock usage under a provider-specific payload field:
   - `inputTokens` -> `input_tokens`;
   - `outputTokens` -> `output_tokens`;
   - `totalTokens` -> `total_tokens`;
   - `cacheReadInputTokens` -> `cache_read_input_tokens`;
   - `cacheWriteInputTokens` -> `cache_creation_input_tokens`.
   Bedrock documents `inputTokens` as uncached input when caching is active, so the existing
   Anthropic usage dialect correctly folds cache reads/writes into total input.
4. Normalize Bedrock stop reasons without discarding the original reason. Extend the generic
   completion mapping only for documented Bedrock values (`stop_sequence`,
   `guardrail_intervened`, `content_filtered`, `model_context_window_exceeded`, and malformed
   output) that do not already map.
5. Map Boto3 `ClientError` and stream error events into typed provider failures. Preserve HTTP
   status, AWS error code/type, request ID, retryability, and retry delay when present. Treat
   throttling, model-not-ready, timeout, internal, service-unavailable, and documented stream
   failures according to their AWS semantics. Detect clear context-window validation failures
   as `BedrockContextOverflowError` so runtime compaction can run.
6. Export the adapter/errors from `cayu.providers` and the provider from top-level `cayu`.
7. Document explicit region/model/profile configuration, IAM permissions
   (`bedrock:CountTokens`, `bedrock:InvokeModel`,
   `bedrock:InvokeModelWithResponseStream`), tool structured output,
   token counting, pricing-catalog entries under provider `bedrock`, and cleanup.

### Tests

Add `tests/core/test_bedrock_provider.py` with injected fake clients/event streams covering:

- system/user/assistant text projection;
- tool specifications, multiple tool calls, fragmented tool JSON, and tool results;
- file attachment projection and rejection of unsupported content;
- explicit model IDs and inference-profile ARNs with no prefix guessing;
- provider option ownership and defensive copies;
- text streaming followed by exactly one completed event;
- finish-reason mappings and protocol failures for incomplete streams;
- usage/cache conversion plus a runtime integration assertion that usage and cost accounting
  see provider `bedrock` and the requested model;
- `CountTokens` request parity and `InputTokenCountResult` metadata;
- tool-based `StructuredOutputSpec` through a complete Cayu run;
- pre-stream and in-stream AWS errors, retry metadata, context overflow, and request IDs;
- event-loop responsiveness, stream cancellation, idle timeout, and owned-client cleanup;
- absence of credentials in events, errors, and request metadata.

### Live verification

`examples/bedrock_provider_live.py` is gated by an explicit live flag plus region/model. Boto3
resolves the standard credential chain only after the operator opts in. It runs text, tool,
tool-result, structured-output, and token-counting paths and emits structured nightly evidence.
Register it in
`scripts/nightly_verification.py` as an opt-in provider-contract check that reports `skipped`
when prerequisites are absent. It is not CI-enforced.

## Lambda MicroVM implementation (#78)

### Implementation

1. Add `src/cayu/runners/lambda_microvm.py` with:
   - lazy/injected `lambda-microvms` client construction;
   - `create`, `from_existing`, readiness, auth-token refresh, reconnect properties;
   - sidecar command start/poll/cancel translation;
   - output decoding/bounds and `ExecResult` construction;
   - timeout/cancellation policies using Cayu's existing cleanup artifact vocabulary;
   - bounded late-start cleanup using the caller-generated command ID;
   - suspend/resume/terminate/none lifecycle behavior and idempotent close.
2. Add a deployable template under `examples/lambda_microvm_sidecar/` containing the sidecar,
   Dockerfile, requirements, and image-build instructions. Keep the sidecar protocol versioned
   and test its command supervisor separately from HTTP routing.
3. Compose `RunnerWorkspace` in `examples/environments/lambda_microvm.py`. Include the
   environment factory and lifecycle binding that:
   - creates a MicroVM for a new session;
   - reattaches on resume from non-secret reconnect metadata;
   - allocates a fresh MicroVM for forks;
   - suspends interrupted sessions;
   - terminates completed/failed sessions after binding finalization.
4. Export `LambdaMicroVMRunner`, `LambdaMicroVMCloseAction`, and constants from
   `cayu.runners` and top-level `cayu`. Do not export the example factory as a core contract.
5. Document image creation, connector/IAM requirements, the JWE auth-token lifecycle,
   sidecar protocol, `RunnerWorkspace` composition, environment-factory reconnect behavior,
   and ECS/Fargate control-plane plus Lambda MicroVM sandbox-plane deployment.

### Tests

Add `tests/runners/test_lambda_microvm.py` and sidecar tests covering:

- create/from-existing calls and validation of image, region, connectors, and idle policy;
- authenticated readiness instead of trusting eventually consistent state;
- setup failure/cancellation terminating only newly created MicroVMs;
- process versus shell translation, cwd containment, explicit env, stdin, and no host-env leak;
- stdout/stderr byte bounds, replacement decoding, non-zero exits, and partial results;
- guest timeout -> `ExecResult(timed_out=True)` with confirmed process-group cleanup;
- cancellation before/after the start response, all cleanup policies, bounded cleanup, and
  `cayu.runner_cleanup.v1` artifacts;
- latching the runner closed when command state cannot be proven safe;
- endpoint token caching/refresh without serialization or log leakage;
- idempotent suspend/resume/terminate/close and `none` detach behavior;
- `RunnerWorkspace` read/write/list/delete and path/symlink guard behavior through the fake
  sidecar;
- factory reconnect, fork isolation, and terminal lifecycle actions;
- sidecar idempotency, process-group termination, result retention, and lifecycle hooks.

All default CI tests use fake control clients and an in-process sidecar transport. They must not
call AWS.

### Live verification

Add `examples/lambda_microvm_runner_live.py`, gated by an explicit live flag, AWS region, and
image identifier. It verifies:

- create/readiness and authenticated endpoint access;
- process and shell forms;
- host-env isolation and explicit env;
- bounded stdout/stderr;
- timeout and cancellation cleanup;
- `RunnerWorkspace` read/write/list/delete;
- suspend/resume with state preserved;
- termination in `finally`, even after an assertion failure.

Register it in `scripts/nightly_verification.py` as an opt-in `verified` AWS sandbox check with
structured evidence. It is not CI-enforced because it requires AWS credentials, a built image,
regional availability, and incurs cost.

## Validation commands

### CI-enforced / hermetic

```bash
uv lock --check
uv run ruff check src/ tests/ examples/ scripts/
uv run ruff format --check src/ tests/ examples/ scripts/
uv run ty check src/cayu examples
uv run pytest -q
```

Targeted during development:

```bash
uv run pytest tests/core/test_bedrock_provider.py -q
uv run pytest tests/runners/test_lambda_microvm.py -q
uv run pytest tests/environments/test_lambda_microvm_factory.py -q
```

### Manual / credentialed

```bash
uv run --extra aws python examples/bedrock_provider_live.py
uv run --extra aws python examples/lambda_microvm_runner_live.py
uv run python scripts/nightly_verification.py --check bedrock-provider-live
uv run python scripts/nightly_verification.py --check lambda-microvm-live
```

The live examples must fail non-zero after they start a real check and an assertion fails. A
missing prerequisite is reported as skipped by the nightly harness, not as a false successful
run.

## Explicitly out of scope

- Silent routing or fallback between direct Anthropic and Bedrock.
- A generic AWS client framework shared across unrelated Cayu modules.
- Bedrock Agents, Knowledge Bases, Mantle endpoints, or server-side tool execution.
- Native Bedrock structured-output claims before schema/model compatibility is implemented.
- A native Lambda filesystem adapter before `RunnerWorkspace` is shown to be insufficient.
- Building MicroVM images dynamically from Cayu core.
- Persisting AWS endpoint auth tokens or raw credentials.

## Primary references

- [Amazon Bedrock ConverseStream](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html)
- [Amazon Bedrock CountTokens](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html)
- [Amazon Bedrock prompt caching usage semantics](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- [AWS Lambda MicroVMs developer guide](https://docs.aws.amazon.com/lambda/latest/dg/lambda-microvms-guide.html)
- [Running and using Lambda MicroVMs](https://docs.aws.amazon.com/lambda/latest/dg/microvms-launching.html)
- [Boto3 LambdaMicroVMs client](https://docs.aws.amazon.com/boto3/latest/reference/services/lambda-microvms.html)
- Cayu contracts: `docs/runtime-contracts.md`, `docs/build-a-runner.md`, and
  `docs/environment-factories.md`
