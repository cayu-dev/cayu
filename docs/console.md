# Cayu application console

`cayu console` opens IPython with a freshly booted Cayu application. It is the
interactive inspection and administration surface for a project, modeled after
`rails console`.

The [application-anatomy guide](../src/cayu/guides/application-anatomy.md)
defines the factory, process ownership, durable-state, and lifecycle boundaries
that the console follows.

## Install

IPython is an optional dependency:

```bash
pip install 'cayu[console]'
# or
uv add 'cayu[console]'
```

The base `cayu` package and commands such as `cayu --help` do not import IPython.

## Configure a project

Declare a factory in the project's `pyproject.toml`:

```toml
[tool.cayu]
factory = "app:build_app"
```

The target must resolve to a synchronous, zero-argument callable that returns a
new `CayuApp`:

```python
from cayu import CayuApp


def build_app() -> CayuApp:
    app = CayuApp()
    # Register providers, environments, and agents here.
    return app
```

Module-global application objects, async factories, factories with required
arguments, and non-`CayuApp` return values are rejected. The factory is invoked
exactly once for each console.

## Open the console

From the project root or any descendant directory:

```bash
cayu console
```

Cayu walks upward to the nearest `pyproject.toml` containing
`[tool.cayu].factory`. That directory becomes the working directory and the
first import path for the lifetime of the console.

To bypass discovery and use the current directory as the root:

```bash
cayu console package.module:build_app
```

The initial namespace contains exactly these project-facing names:

- `cayu`: the Cayu package
- `app`: the console-local `CayuApp`
- `sessions`: `app.session_store`
- `tasks`: `app.task_store`, or `None`
- `knowledge`: `app.knowledge_store`, or `None`

`app` is not a framework singleton. It is the object returned by your factory
for this one console process.

## Async behavior

IPython supports top-level `await`, and terminal IPython reuses one event loop
across async cells:

```python
sessions = await app.session_store.list_sessions()
```

The loop pauses while IPython waits at the prompt. Background tasks therefore
do not keep making progress between cells. The console is an inspection and
administration process, not a Cayu worker, watcher, or scheduler.

## Safety and lifecycle

The banner identifies the project, factory, registered component names, and
store class names without printing configuration values or credentials.

IPython records entered commands in its history. Read credentials from the
environment or a configured vault; do not paste secret values into console
commands.

The namespace is live and writable. Mutating store methods affect the same
configured persistence backends as the application. Cayu does not start a
FastAPI lifespan, run recovery, resume or recover sessions, launch background
workers, invoke models or tools, or run storage migrations beyond behavior
explicitly requested by project construction. It performs no cleanup beyond
what the application factory itself does.

File-backed Cayu SQLite stores use WAL and a busy timeout, but SQLite remains
single-writer. Long or contended writes from another process can still raise
`database is locked`; use the console accordingly when a SQLite worker is busy.
