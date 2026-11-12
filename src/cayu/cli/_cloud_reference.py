"""How `--application` names a Cayu Cloud Agent, said once for every command that takes it."""

from __future__ import annotations

#: An Agent's id is its slug, the `application` in its cayu-cloud.toml, and Cloud gives it to
#: the Agent's own processes as CAYU_CLOUD_APPLICATION_ID, so an app can show a command that
#: works as printed. Its name (`name` in cayu-cloud.toml) works too while it is unique.
APPLICATION_HELP = (
    "The Agent: its id (the `application` slug in cayu-cloud.toml, which Cloud gives the "
    "Agent as CAYU_CLOUD_APPLICATION_ID) or its name."
)
