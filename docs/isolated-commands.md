# Isolated commands on files

`run_isolated` runs one fixed command on some files inside a runner and reads
the files it writes back as bytes. Use it for small fixed programs that read
untrusted input, such as converting an uploaded spreadsheet, where the program
should never see the network or the host.

```python
from cayu.runners import ExecCommand, IsolatedLimits, run_isolated

run = await run_isolated(
    runner,
    ExecCommand.process("python3", "-I", "convert.py"),
    inputs={"convert.py": script, "input.xlsx": upload},
    outputs=["output.xlsx", "result.json"],
    limits=IsolatedLimits(timeout_s=45, max_output_file_bytes=20 * 1024 * 1024),
)
if not run.succeeded:
    raise ValueError(run.result.stderr or "conversion failed")
converted = run.outputs["output.xlsx"]
```

Paths are relative to the runner's working directory and may not be absolute
or use `..`. Inputs are written before the command starts, parent folders
included. Declared outputs that are not inputs are deleted first, so a
reused runner never returns an earlier run's files. Outputs are read only after the command exits 0 without timing out;
`run.outputs` holds the ones it wrote and `run.missing` names the ones it did
not. `run.result` is the command's `ExecResult` with bounded stdout and stderr.
`run.succeeded` is true when the command exited 0 and wrote every output.
Output archives are read in memory, only for the declared regular files;
nothing from the runner is extracted on the host.

## Choosing the runner

The runner decides the isolation; `run_isolated` does not create or close it.
`open_isolated_runner` opens a runner with no network and always releases it:

```python
from cayu.runners import isolated_run_key, open_isolated_runner, run_isolated

key = isolated_run_key(command, inputs)
async with open_isolated_runner(key) as runner:
    run = await run_isolated(runner, command, inputs=inputs, outputs=["output.xlsx"])
```

It selects the backend the way `select_coding_backend` does: Cayu Cloud's
sandbox broker (`cayu-cloud`) when `CAYU_CLOUD_API_URL`,
`CAYU_CLOUD_APPLICATION_ID` and `CAYU_CLOUD_SANDBOX_TOKEN` are set, Docker
(`docker`) otherwise. `CAYU_ISOLATED_BACKEND`, or `required=`, names one
backend that must be usable, so a missing Cloud variable fails instead of
falling back to Docker.

- `docker` creates a container with `network="none"` and
  `DockerWorkloadRestrictions()`, and removes it on exit.
  `DockerIsolatedOptions(image=...)` picks the image, which needs `python3`.
- `cayu-cloud` creates a sandbox with `egress_mode="none"`, waits until it is
  ready, refuses it unless the broker reports no egress, and destroys it on
  exit, including when the wait or your code fails. The lease is renewed
  while the block runs, so long commands keep their sandbox; a process that
  stops without exiting leaves only the remaining lease, which Cloud ends.
  `CloudIsolatedOptions` sets the catalog image (default `coding-python`), the
  lease, the ready timeout, labels, and an existing `broker` client to use.

Pass options keyed by backend name: `options={"docker":
DockerIsolatedOptions(image="python:3.13-slim")}`. The key labels the runner
(the container name, the sandbox's `cayu-run-key` label) and seeds the
sandbox's idempotency key; every call still gets a fresh runner.

Another provider plugs in by name. An opener takes an `IsolatedRunnerContext`
(the key, the environment and that backend's options) and returns an async
context manager that yields a `Runner`:

```python
from cayu.runners import default_isolated_backends

registry = default_isolated_backends()
registry.register("my-sandbox", open_my_sandbox, available=my_sandbox_unavailable_reason)
async with open_isolated_runner(key, registry=registry, required="my-sandbox") as runner:
    ...
```

Files move as one uncompressed tar each way, by the best transport the runner
offers:

- staged copies, when the runner has `staged_copies`, `copy_in` and `copy_out`
  (Cayu Cloud's brokered sandboxes), so large files skip the command channel;
- binary streams, when `binary_stream_capability()` returns one (Docker and
  local runners);
- otherwise the runner's bounded text output, which suits small files.

The last two use the guarded workspace operations, so the guest needs
`python3`.

## Run keys

`run.key`, also returned by `isolated_run_key(command, inputs)`, is a SHA-256
digest of the command and every input's contents. Equal commands on equal
files give the same key in any input order. Use it as a sandbox idempotency key
or a [task step](triggering-runs.md#saved-step-results) name, so a rerun on the
same files reuses the saved result instead of running the command again.
