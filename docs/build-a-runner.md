# Build a Runner

This is a how-to guide for implementing a custom `Runner` — the component that
executes commands inside a workspace or sandbox — so you can connect Cayu to any
execution platform (a cloud sandbox, a container service, a microVM, a remote
worker). It walks through a working `ModalRunner` ([`examples/modal_runner.py`](../examples/modal_runner.py))
built against [Modal](https://modal.com) Sandboxes, but the contract is the same
for any backend.

The built-in runners (`LocalRunner`, `DockerRunner`, `E2BRunner`,
`MicrosandboxRunner`, `SbxRunner`) all implement this same contract; read their
source in `src/cayu/runners/` alongside this guide.

## The Runner contract

A runner subclasses `cayu.runners.Runner` and implements one method, `exec`
(`src/cayu/runners/base.py`):

```python
class Runner(ABC):
    isolation: str = "unknown"

    @abstractmethod
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult: ...
```

- `command` is positional; everything else is keyword-only. Match the signature
  and defaults exactly — `ExecCommandTool` calls `exec(...)` with these keywords.
- Set `isolation` to a short label for your backend (`"modal"`, `"docker"`, …).
- Implement `resolve_cwd()` as an idempotent containment boundary. Relative
  requests resolve beneath `default_cwd`; an absolute input is accepted only
  when it is the runner's own canonical root/child path. In particular,
  `resolve_cwd(resolve_cwd(value))` must equal `resolve_cwd(value)`. A configured
  `CommandPolicy` authorizes that canonical result and `ExecCommandTool` passes
  the same value to `exec(...)`, including the canonical default when the model
  omitted `cwd`. Therefore attaching a command policy requires a runner whose
  resolver accepts its own canonical output on every invocation.
- **`Runner` is exported from `cayu.runners`, not the top-level `cayu`** (only the
  concrete runners are re-exported at the top level):

  ```python
  from cayu.runners import Runner
  from cayu import ExecCommand, ExecResult, RunnerCancelledError, DEFAULT_EXEC_OUTPUT_LIMIT_BYTES
  ```

### `ExecCommand` — what you receive

```python
class ExecCommand(BaseModel):
    kind: Literal["process", "shell"] = "process"
    argv: list[str] | None = None    # process form (the safe default)
    shell: str | None = None         # explicit shell script
```

`process` is argv (no shell parsing). `shell` is a script string where quoting and
expansion are intentional. Translate both to your platform:

- If your SDK takes argv natively (Modal, Microsandbox), pass `command.argv`
  through and run a `shell` script as `["bash", "-c", command.shell]`.
- If your SDK only takes a command *string* (E2B runs through Bash), quote argv
  with `shlex.join(command.argv)` and pass `command.shell` verbatim. See
  `src/cayu/runners/e2b.py`.

### `ExecResult` — what you must return

```python
class ExecResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: StrictInt = 0
    timed_out: StrictBool = False
    cancelled: StrictBool = False
    stdout_truncated: StrictBool = False
    stderr_truncated: StrictBool = False
    artifacts: list[dict] = Field(default_factory=list)
```

`exec` must **return a real `ExecResult`** (the tool layer does
`type(result) is ExecResult`). A non-zero exit code is **not** an error — put it
in `exit_code` and return normally. Reserve raising for typed runner conditions:
cancellation (below), or a `RunnerUnavailableError` subclass when provider
evidence confirms that the execution environment cannot accept more commands.

### Cancellation

On `asyncio.CancelledError`, stop the running command, then raise
`RunnerCancelledError` (a subclass of `asyncio.CancelledError` that carries
optional cleanup `artifacts`). Do not swallow it.

## Build a ModalRunner, step by step

Follow [`examples/modal_runner.py`](../examples/modal_runner.py).

1. **Lazy-import the SDK.** Modal is not a Cayu dependency, so import it lazily so
   the module imports even when `modal` is absent — and let tests inject a fake:

   ```python
   def _modal_module(module=None):
       if module is not None:
           return module
       try:
           return importlib.import_module("modal")
       except ModuleNotFoundError as exc:
           if exc.name != "modal":
               raise
           raise RuntimeError("ModalRunner requires the optional modal package. ...") from exc
   ```

   This mirrors `_e2b_module()` / `_microsandbox_module()` in the built-in runners.

2. **Subclass and translate.** Set `isolation = "modal"`, validate
   `type(command) is ExecCommand`, resolve relative or already-canonical `cwd`
   to an absolute in-root path, and build argv (`process` → `argv`; `shell` →
   `["bash", "-c", script]`). Reject every absolute path outside the runner root.

3. **Run it on the platform.** Modal's `exec` takes a plain `env` dict directly
   (it also accepts a native `timeout=` and a `secrets=` list) — passing the
   resolved `env` is the `ExecCommand` → SDK translation point:

   ```python
   process = await self._sandbox.exec.aio(*argv, workdir=working_dir, env=environment)
   ```

   (Modal's own guide demonstrates env via `secrets=[modal.Secret.from_dict(env)]`; the
   `env=` parameter in the API reference is equivalent and simpler, so we use it.)

4. **Capture output with truncation.** Set `stdout_truncated` / `stderr_truncated`
   and decode bytes with `errors="replace"` so non-UTF-8/binary output never
   crashes you (the example's `_truncate` accepts `bytes | str`). **The contract
   expects `output_limit_bytes` to bound capture *inside* the runner** so a command
   can't exhaust memory before the result is built. The example reads *all* output
   and truncates the final string — simple, but **not memory-safe for unbounded
   output**. A production runner streams into a bounded buffer (`_LimitedBytes` in
   `microsandbox.py`), which also lets partial output **survive a timeout** and —
   for OS-pipe backends — lets you read stdout/stderr **concurrently**
   (`asyncio.gather`) to avoid a full-pipe deadlock. (Modal's streams are
   network-buffered, so the example reads them sequentially — and because it reads
   *before* `wait()` returns, partial output survives a command timeout.)

5. **Enforce timeout and cancellation.** Prefer your platform's *native* command
   timeout so one slow command doesn't destroy shared state: the example passes
   Modal's `exec(..., timeout=timeout_s)` (bounded server-side). At the deadline Modal
   kills the command and `wait()` returns `-1` with the partial output still readable,
   which the example maps to `ExecResult(timed_out=True, exit_code=-9, ...)` (keeping the
   partial output) **without** tearing down the sandbox. On `asyncio.CancelledError`,
   **raise** `RunnerCancelledError`; *there*
   the example does terminate the sandbox, because Modal has no per-command kill
   (`ContainerProcess` has no `terminate`) — and it **bounds** that cleanup so a
   hung terminate can't make cancellation hang, attaching a `cayu.runner_cleanup.v1`
   diagnostic. A backend with a real per-command kill should stop just the command
   (the built-in runners do this via their cleanup-policy path).

6. **Return** a real `ExecResult` with the captured stdout/stderr/exit code.

## The secret-injection gotcha

**Runners never receive a vault or secret references — only a plain, already
resolved `env: dict[str, str]`.** Two rules follow:

- **Never inherit the host process environment.** The built-in sandbox runners
  call `copy_runner_env(env, inherit_env=False)`, so host secrets (your API keys,
  cloud credentials) never leak into the sandbox. Build the child env from the
  explicit `env` only, as the example's `_copy_env` does.
- **Secret resolution happens at the environment/vault boundary — it is the
  app's responsibility, not something the runtime does for command execution.**
  Cayu's secret machinery (`SecretRef`, `Vault`, `Environment.resolve_secret`)
  lives on the `Environment`. Note that the built-in `ExecCommandTool` forwards the
  **plain `env` from its tool arguments** straight to `runner.exec` — it does
  **not** itself resolve `SecretRef` into runner env. An app that wants secrets in a
  command's environment resolves them at the environment/vault boundary and passes
  the resolved values in. Either way, do **not** add a vault parameter to your
  runner — that breaks the `Runner` contract and the substitutability the tool layer
  relies on.

Also avoid baking long-lived secrets into the sandbox image or a persistent
sandbox env, where they outlive the command and may be exfiltrated by sandboxed
code. Modal's own guidance is explicit: never put secrets inside a sandbox. For
credential brokering that keeps secrets off the sandbox entirely, see:

- [Secret-injection research report](../research/secret-injection-report.html) — Cayu's deep dive on why plain env vars leak secrets and the proxy/credential-broker approaches
- [Modal Secrets](https://modal.com/docs/guide/secrets)
- [Infisical Agent Vault](https://infisical.com/blog/agent-vault-the-open-source-credential-proxy-and-vault-for-agents) — open-source credential proxy; recommended for platforms without built-in proxy support
- [IETF CB4A draft](https://www.ietf.org/archive/id/draft-hartman-credential-broker-4-agents-00.html) — emerging credential-brokering standard

## Testing pattern

Mock the platform SDK — never hit the real service in unit tests. Construct the
runner with a **fake sandbox** and drive `exec()`; the SDK module is only needed
by `create()`, which takes a `modal_module=` injection hook (mirroring
`sandbox_module=` on `MicrosandboxRunner`), so you never monkeypatch `importlib`:

```python
# FakeSandbox.exec records its call and returns stdout="abcdef".
sandbox = FakeSandbox(stdout="abcdef")
runner = ModalRunner(sandbox)
result = await runner.exec(
    ExecCommand.process("echo", "abcdef"), cwd="src", env={"VISIBLE": "1"},
    timeout_s=5, output_limit_bytes=3,
)
assert sandbox.exec_calls[0]["argv"] == ["echo", "abcdef"]
assert sandbox.exec_calls[0]["workdir"] == "/workspace/src"
assert sandbox.exec_calls[0]["env"] == {"VISIBLE": "1"}   # no host env leaked in
assert result.stdout == "abc" and result.stdout_truncated is True
```

Assert: command translation (argv/workdir/env recorded by the fake), output
truncation, the no-host-env-leak rule (a host env var you set is absent from the
forwarded env), timeout → `timed_out=True`, and cancellation → `RunnerCancelledError`.
The built-in tests are full worked examples: `tests/runners/test_microsandbox.py`,
`tests/runners/test_e2b.py`.

## Registration

A runner attaches to an `Environment` (which validates
`isinstance(runner, Runner)`), and the environment registers on the app:

```python
from cayu import CayuApp, Environment, EnvironmentSpec

runner = await ModalRunner.create(app=modal_app, image=modal.Image.debian_slim())
app = CayuApp()
app.register_environment(Environment(EnvironmentSpec(name="modal"), runner=runner), default=True)
```

Any agent in that environment whose tools include `ExecCommandTool` then runs its
commands through your runner.

## Checklist

- [ ] Subclass `cayu.runners.Runner`; set `isolation`.
- [ ] Match the exact `exec` signature/defaults; validate `type(command) is ExecCommand`.
- [ ] Translate `process` (argv) and `shell` (script) commands.
- [ ] Enforce `timeout_s` → return `ExecResult(timed_out=True, exit_code=-9)` (the
      built-in sandbox runners' timeout convention).
- [ ] On `asyncio.CancelledError`, terminate the command and raise `RunnerCancelledError`.
- [ ] Truncate stdout/stderr to `output_limit_bytes`; set the `*_truncated` flags.
- [ ] For OS-pipe backends, read stdout/stderr concurrently and buffer incrementally
      (preserves partial output on timeout; avoids pipe deadlock).
- [ ] Capture stderr separately; never raise on a non-zero exit — return it in `exit_code`.
- [ ] Build the child env from the explicit `env` only — no host-env inheritance.
- [ ] Resolve secrets at the environment/vault boundary, not in the runner.
- [ ] Resolve relative `cwd` beneath `default_cwd`; accept only contained
      canonical absolute values; reject outside/escaping paths.
- [ ] Keep `resolve_cwd()` idempotent so policy-authorized canonical paths execute
      unchanged.
- [ ] Lazy-import the optional SDK with an injectable module hook for tests.
- [ ] Return a real `ExecResult`.
