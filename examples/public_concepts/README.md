# Public concepts in a working application

[app.py](app.py) imports application composition, agents, messages, events,
context policy, and session state from their named public modules. It runs the
real Cayu session engine with an in-memory store and a deterministic provider.
No model credentials or external services are required.

From the repository root after installing the development dependencies:

```sh
uv run python -m examples.public_concepts.app
```

The output includes `Hello from the public concept layout.`,
`session.completed`, and `Recorded 10 runtime events.`

The public imports are:

```python
from cayu.applications import CayuApp
from cayu.agents import AgentSpec
from cayu.messages import Message
from cayu.events import Event, EventType
from cayu.context import DefaultContextPolicy
from cayu.sessions import InMemorySessionStore, RunRequest
from cayu.providers import ModelStreamEvent
from cayu.evals import ScriptedModelProvider
```

`build_app()` registers the provider and the agent. Context policy is configured
on `register_agent()`. `run_example()` submits a `RunRequest` with a user message
and collects the runtime's events. The test asserts that the session completes
and produces model text through these public entry points.
