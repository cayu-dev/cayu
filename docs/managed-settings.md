# Managed settings

An Agent deployed to Cayu Cloud can have its settings managed from the portal,
the API and the `cayu cloud` CLI. The code gives every default; a value set in
Cloud overrides it until someone clears it.

Cayu owns the schema of an Agent's own settings: each part of the application
declares the settings it has, and the running Release publishes the assembled
schema to Cloud when it starts. Cloud renders the portal's form from that
schema, so a new setting needs no Cloud or portal change. Platform settings
(sleep, CPU and memory, schedules, sign-in) are Cloud's own and use the same
descriptor.

## Declaring settings

A part declares its settings where it is registered, so an application
assembled by hand gets them as well as a scaffolded one. Keys are relative to
the part; registration puts the part's namespace in front and sets the owner.

| Where | Key | Owner |
| --- | --- | --- |
| `register_agent(spec, settings=[...])` | `agent.<agent>.<key>` | `agent:<agent>` |
| `register_provider(provider, settings=[...])` | `provider.<provider>.<key>` | `provider:<provider>` |
| `register_environment_factory(spec, factory, settings=[...])` | `environment.<environment>.<key>` | `environment:<environment>` |
| `app.declare_settings([...])` | `app.<key>` | `app` |

```python
from cayu import AgentSpec, CayuApp
from cayu.managed_settings import (
    ManagedSetting,
    model_setting,
    provider_option_setting,
    thinking_effort_setting,
)

app.register_agent(
    AgentSpec(name="reviewer", model="claude-sonnet-4-6", provider_options={"temperature": 0.2}),
    settings=[
        model_setting(),
        thinking_effort_setting(),
        provider_option_setting("temperature", type="number", label="Temperature", maximum=1),
        ManagedSetting(key="max_files", type="int", label="Files per review", default=20, minimum=1),
    ],
)
```

### Lists and maps

`list_setting` declares a list of strings, and `map_setting` a map from strings
to strings. Each item (a map's key) must match `item_pattern` or be one of
`item_choices`; a map's values must match `value_pattern` or be one of
`value_choices`. `max_items` caps the count and `max_length` (4096 unless given)
the length of the whole value. Patterns are Python regular expressions that must
match the whole item.

```python
from cayu.managed_settings import list_setting, map_setting

REPOSITORY = r"[A-Za-z0-9-]+/[A-Za-z0-9._-]+"

app.declare_settings(
    [
        list_setting(
            "repositories",
            label="Repositories",
            item_pattern=REPOSITORY,
            max_items=100,
            applies="live",
        ),
        map_setting(
            "merge_modes",
            label="Merge modes",
            item_pattern=REPOSITORY,
            value_choices=("report", "approve", "auto"),
            default={"cayu-tech/cayu": "report"},
            applies="live",
        ),
    ]
)

app.managed_setting_value("app.repositories")  # ("cayu-tech/cayu", "cayu-tech/cayu-cloud")
app.managed_setting_value("app.merge_modes")  # {"cayu-tech/cayu": "approve"}
```

Both are `string` settings with a `format` (`list` or `map`), so a Cloud that
doesn't know `format` still takes them. The value is written and stored as text, items separated by commas or line breaks
and a map's entries as `KEY=VALUE`: `cayu-tech/cayu,cayu-tech/cayu-cloud` and
`cayu-tech/cayu=approve,cayu-tech/cayu-cloud=auto`. Spaces around items are
dropped, a repeated list item counts once, and a repeated map key is refused, so
items can't contain a comma or a line break and a map's keys can't contain `=`.
The application reads a tuple for a list and a dict for a map, empty when there is
no value. A value with an item that doesn't fit is refused with a reason that
names the item; `ManagedSetting.validate_value` gives the check and the stored
text, and `read_value` the tuple or dict.

### A value from the environment

`ManagedSetting(env="NAME")` names an environment variable whose value applies
when Cloud has none, so a setting declared once works on Cayu Cloud and on a
laptop or in a test. The text is read by the setting's type
(`ManagedSetting.parse_text`): a `bool` takes `true`/`false`, `1`/`0`, `yes`/`no`
or `on`/`off` in any case, an `int` or `number` a number, a list or map its
comma-separated text. Blank text counts as unset, except for `string`, `list`
and `map` settings, where it is the empty value. A variable whose text doesn't fit
is ignored with a warning, and the default applies.

```python
list_setting("repositories", label="Repositories", item_pattern=REPOSITORY,
             env="SHEPHERD_REPOSITORIES")
ManagedSetting(key="introduce", type="bool", label="Introduce first",
               default=True, env="SHEPHERD_INTRODUCE")
```

The order is Cloud's value, then the environment's, then the code's default. The
descriptor names the variable (`env`), but Cloud shows the code's default as the
Release's value even where the variable is set on the Agent, so on Cloud set the
value in Cloud rather than in the environment. The settings Cayu applies to an
agent itself (`model`, `thinking.effort`, `provider_options.*`) can't take `env`.

A `restart` setting keeps what its variable held when the setting was declared,
as it keeps Cloud's value from the build: changing the variable in a running
process doesn't change the value until the process starts again.

### Reading a value and its source

`app.read_managed_setting(key)` returns a `ManagedSettingReading`: the `value`
`managed_setting_value` would return, its `source` (`"cloud"`, `"environment"`
or `"default"`), the `env` variable it reads, and `refused`, why a value from
Cloud or the environment wasn't used when one was there.
`app.read_managed_settings()` returns one for every declared setting, sorted by
key, which is what a project's `status` command prints:

```python
for reading in app.read_managed_settings():
    if reading.key.startswith("app."):
        print(reading.key, reading.value, reading.source, reading.refused or "")
```

Presets and factories that register a part for you pass `settings` through
(`WebBridgeProfile.register_agent`, `register_coding_environment`) and never
declare settings of their own on the part's behalf.

A part's name becomes one segment of the key, so only parts whose name uses
lowercase letters, digits, `_` and `-` can declare settings. A declaration that
contains one of the application's secret values, repeats a key, or would push
the schema over its limits is refused when it is made.

### Settings Cayu applies for an agent

Some agent keys are applied by Cayu itself, with the agent's own value as the
default:

- `model` (`model_setting()`, deprecated): the model new sessions use. A string,
  or an enum when you pass `choices`. It applies at `next_session` (or
  `restart`). Declare a model slot instead (`app.require_model(SLOT,
  default=MODEL)` and `register_agent(..., model_slot=SLOT)`; see
  [ADR 0003](adr/0003-connections-runtime-contract.md#models)), which Cayu
  Cloud binds per Agent. Calling `model_setting()` warns with a
  `DeprecationWarning`, and `cayu check` reports `MODEL_SETTING_DEPRECATED`.
  Until the setting is removed, an agent registered with it uses a model slot
  named after it (`_` as `-`) whose default is the agent's own model: a model
  bound to that slot wins, otherwise the setting's value applies. An agent whose
  model the Gateway's model policy manages (`CayuApp(model_policy=...)`, also
  deprecated) leaves `model` out of its schema.
- `thinking.effort` (`thinking_effort_setting()`): an enum of thinking efforts.
- `provider_options.<name>` (`provider_option_setting(name, ...)`): one scalar
  provider option.

Thinking and provider options apply only at `restart`, because a session's
execution profile fingerprints them: changing them between sessions would stop a
resumed session from matching the profile it started with. Any other key is the
application's own, and it reads the value itself.

### Settings the framework declares

Every application also declares, with owner `config`, these `CayuConfig`
fields, so a hand-assembled application has a schema even without declarations
of its own:

| Key | Applies |
| --- | --- |
| `config.run.max_steps` | `next_session` |
| `config.run.limits.max_input_tokens`, `max_output_tokens`, `max_total_tokens`, `max_tool_calls`, `max_elapsed_seconds` | `next_session` |
| `config.run.retry_policy.max_attempts` | `next_session` |
| `config.run.thinking.effort` | `restart` |

Their defaults are the application's `CayuConfig`.

Each application budget limit (`CayuApp(budget_policy=...)`) declares its
amount, `budget.<name>.max_estimated_cost`, with owner `budget:<name>`. The
name is the limit's scope, its key and its window: `app`, `app-month`,
`agent-reviewer`, `agent-reviewer-86400s`. A limit whose name is not unique, or
that can't be part of a key, is left out. Budget amounts apply at `restart`.

### When a setting applies

- `restart`: read when the application is built. Cloud republishes the service
  when one changes.
- `next_session`: read when a session starts. Running sessions, and sessions
  that resume, keep the values they started with.
- `live`: read whenever the application asks for the value. Cayu applies none of
  its own settings live; use it only for settings that are safe to change while a
  session runs.

### Scaffolded projects

`cayu new` declares a `primary` model slot in `agents/registration.py` and
registers the starter agent on it, so a deployed Agent's model is bound in Cayu
Cloud and defaults to the model in `agents/agent.py`. The run limits and retries
come with every application. `configuration/settings.py` still reads
`CAYU_MODEL` as a fallback default until the model slots replace it; a model
bound in Cloud overrides it. Add a
project's own settings in `build_app` with `app.declare_settings([...])`, and
pass `settings=` when registering further agents.

## The Release's schema

`app.managed_settings_schema()` returns the assembled
`cayu.managed-settings-schema/v1` document. It is also the `settings` section of
`app.describe()`, which the manifest fingerprint covers. The settings are sorted
by key, and `digest` is the SHA-256 of their canonical JSON (keys sorted, no
spaces, UTF-8 without escaping):

```python
json.dumps(settings, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
```

A schema holds at most 500 settings and 256 KiB. Cayu refuses to assemble a
larger one, at the declaration that would exceed a limit, instead of truncating
it.

When the server starts in Cloud, the agent registry registration
(`PUT /v1/applications/{id}/agent-registry/releases/{release}`, see
[cloud-inventory.md](cloud-inventory.md)) carries the schema in its optional
`settings_schema` field, so Cloud has the schema of the Release that is actually
running, however the application was assembled. If Cloud refuses the schema with
422, the Runtime registers again without it and keeps reporting tools and use.
`cayu cloud deploy` still sends only the deployment manifest: building the
application on a laptop at deploy fails whenever its dependencies aren't
installed there.

## How the Agent applies Cloud's values

`ManagedSettings` (in `cayu.managed_settings`) reads the Agent's effective
values from Cloud and reports what the process applied. It is on only in Cloud:
`CayuApp` creates it from `CAYU_CLOUD_API_URL`, `CAYU_CLOUD_APPLICATION_ID`,
`CAYU_CLOUD_REGISTRY_TOKEN` and the Release id (`CAYU_RELEASE_ID` or
`release_identity`), the same credential the agent registry uses. Without them
the code's defaults apply and nothing polls. To supply your own channel, pass
`CayuApp(managed_settings=ManagedSettings(channel))`.

An app that shows people how to change a setting from the CLI can print
`cayu cloud settings set --application "$CAYU_CLOUD_APPLICATION_ID" KEY=VALUE`:
the Agent's id is the `application` slug in its `cayu-cloud.toml`, and
`--application` takes it as well as the Agent's name.

- When the application is built, it reads Cloud's values once (5-second
  timeout). `restart` settings come from that read for the life of the process:
  config fields and budget amounts in the `CayuApp` constructor, an agent's
  thinking and provider options when it registers. An agent's `restart` model
  is chosen, like a `next_session` one, when a session starts, from the value read
  at build.
- `create_server`, `mount_cayu` (so `cayu serve`) and `cayu worker` call
  `await app.start_managed_settings(role="web" | "worker")`, which reads again
  and then every 25 seconds. `app.aclose()` stops it. Custom hosts call the same
  method; one-shot processes such as schedules use the read made at build.
- `next_session` settings apply when a session starts. Config fields swap the
  application's effective `CayuConfig` (`app.config`), so
  `with_application_run_defaults` gives new runs the managed `max_steps`, limits
  and retries, and their provenance becomes `"managed"`. An agent's `model` is
  chosen in session preparation.
- The order is: an explicit `RunRequest` value, then the value from Cloud, then
  the code's default. An agent with a Gateway model-policy binding keeps the
  policy's model.
- Running sessions, and sessions that resume, keep the values they started
  with: a session records its model, and a resume inherits the run controls the
  session was admitted with, so its execution profile still matches. A new run
  resolves its retry policy once, when its defaults are applied (a durable
  subagent gets it when it is submitted, so a queued child keeps it). A queued durable
  subagent keeps the model it was prepared with, and a participant session the
  model and run controls it was created with, as long as the process that runs it
  applied them.
- `session.started` records `managed_settings`: the snapshot's `revision` and the
  values from Cloud the session started with, leaving out any it doesn't use (an
  explicit request field, or a model or run control it kept from earlier).
- A value this process can't use is refused, and the code's default applies: it
  doesn't fit its descriptor, `CayuConfig` rejects it, the agent's provider
  refuses the model (`preflight_model_target`), or a `secret_ref` names an
  environment variable that isn't set.
- The application reads its own settings with
  `app.managed_setting_value("app.search_key")`: Cloud's value when it is set and
  usable, else the setting's `env` variable when it is set and valid, else the
  default. A `restart` setting returns the value read at build.
  `app.read_managed_setting(key)` also says where the value came from.

When Cloud can't be reached, the last good snapshot stays in use. When it never
got one, the code's defaults apply. When Cloud has no schema for the Release yet
(409 `settings_schema_missing`), the defaults apply and the next poll tries
again.

After each read the controller reports to Cloud when what it applied changed,
and at least every 5 minutes: `applied` with the snapshot's revision, `stale`
once the snapshot is past `valid_until`, or `unreported` (revision `null`) when
it never got one. Each declared setting is `adopted`, `default` (no value in
Cloud), or `refused` with a reason. A `restart` setting whose value changed since
the process started is refused with "takes effect when the service restarts"
until Cloud's republish replaces the process. Reports never carry values.

## Contracts

These are the documents of the `v1` contract between the Runtime and Cloud. All
are JSON in UTF-8, times are RFC 3339 in UTC, and readers ignore fields they
don't know, so either side can add optional fields within `v1`.

### Setting descriptor

One setting. The Runtime emits these for Agent settings; Cloud uses the same
shape for platform settings, which may also carry `admin_only`, `locked` and
`locked_reason`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:cayu:managed-setting/v1",
  "title": "Managed setting descriptor",
  "type": "object",
  "required": ["key", "owner", "type", "default", "applies", "label"],
  "properties": {
    "key": {
      "type": "string",
      "maxLength": 128,
      "pattern": "^[a-z][a-z0-9_]*(\\.[a-z0-9_-]+)*$"
    },
    "owner": {"type": "string", "minLength": 1, "maxLength": 128},
    "type": {"enum": ["bool", "int", "number", "string", "enum", "secret_ref"]},
    "default": {"type": ["boolean", "integer", "number", "string", "null"]},
    "minimum": {"type": "number"},
    "maximum": {"type": "number"},
    "max_length": {"type": "integer", "minimum": 1, "maximum": 4096},
    "choices": {
      "type": "array",
      "minItems": 1,
      "maxItems": 100,
      "items": {
        "type": "object",
        "required": ["value", "label"],
        "properties": {"value": {"type": "string"}, "label": {"type": "string"}}
      }
    },
    "applies": {"enum": ["restart", "next_session", "live"]},
    "label": {"type": "string", "minLength": 1, "maxLength": 80},
    "description": {"type": "string", "maxLength": 500},
    "format": {"enum": ["list", "map"]},
    "item_pattern": {"type": "string", "minLength": 1, "maxLength": 500},
    "item_choices": {"$ref": "#/properties/choices"},
    "value_pattern": {"type": "string", "minLength": 1, "maxLength": 500},
    "value_choices": {"$ref": "#/properties/choices"},
    "max_items": {"type": "integer", "minimum": 1, "maximum": 500},
    "env": {"type": "string", "pattern": "^[A-Z_][A-Z0-9_]{0,127}$"},
    "admin_only": {"type": "boolean"},
    "locked": {"type": "boolean"},
    "locked_reason": {"type": "string"}
  },
  "allOf": [
    {
      "if": {"required": ["format"]},
      "then": {"properties": {"type": {"const": "string"}}},
      "else": {
        "not": {
          "anyOf": [
            {"required": ["item_pattern"]},
            {"required": ["item_choices"]},
            {"required": ["value_pattern"]},
            {"required": ["value_choices"]},
            {"required": ["max_items"]}
          ]
        }
      }
    },
    {
      "if": {"properties": {"type": {"const": "bool"}}},
      "then": {"properties": {"default": {"type": ["boolean", "null"]}}}
    },
    {
      "if": {"properties": {"type": {"const": "int"}}},
      "then": {"properties": {"default": {"type": ["integer", "null"]}}}
    },
    {
      "if": {"properties": {"type": {"const": "number"}}},
      "then": {"properties": {"default": {"type": ["number", "null"]}}}
    },
    {
      "if": {"properties": {"type": {"const": "string"}}},
      "then": {
        "required": ["max_length"],
        "properties": {"default": {"type": ["string", "null"]}}
      }
    },
    {
      "if": {"properties": {"type": {"const": "enum"}}},
      "then": {
        "required": ["choices"],
        "properties": {"default": {"type": ["string", "null"]}}
      }
    },
    {
      "if": {"properties": {"type": {"const": "secret_ref"}}},
      "then": {
        "properties": {
          "default": {"type": ["string", "null"], "pattern": "^[A-Z_][A-Z0-9_]{0,127}$"}
        }
      }
    }
  ]
}
```

An Agent setting's key has at least one dot, so each is namespaced by its part
(the Release schema below requires it); only Cloud's platform settings may be a
single segment, such as `sign_in`. Rules the schema can't express: `minimum` and `maximum` apply only to `int` and
`number`; a non-null default must be valid for the type (within bounds, at most
`max_length` characters, one of the `choices`); enum choice values are distinct;
a key is unique within its namespace. A `secret_ref` value is the name of an
environment variable on the Agent, never a secret value.

A list or map setting is a `string` setting with `format` (`list` or `map`); a
reader that doesn't know `format` treats it as plain text, so a Cloud or Release
without it still takes the setting and its value. Its value is the items joined
with commas, a map's entries as `KEY=VALUE`, and each item (a map's key) matches
`item_pattern` or is one of `item_choices`, each map value matches
`value_pattern` or is one of `value_choices`, at most `max_items` of them.
`item_pattern` and `item_choices` (and `value_pattern` and `value_choices`) are
not given together, and the `value_` fields only with `map`. Patterns are Python
regular expressions matched against the whole item. Cayu's
`ManagedSetting.from_descriptor(descriptor).validate_value(value)` is the check,
naming the items that don't fit. `env` names the environment variable whose
value the Agent uses when Cloud has none; Cloud only shows it.

### Release schema: `cayu.managed-settings-schema/v1`

Published by the Runtime in the registry registration's `settings_schema`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:cayu:managed-settings-schema/v1",
  "title": "A Release's managed settings schema",
  "type": "object",
  "required": ["contract", "digest", "settings"],
  "properties": {
    "contract": {"const": "cayu.managed-settings-schema/v1"},
    "digest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
    "settings": {
      "type": "array",
      "maxItems": 500,
      "items": {
        "$ref": "urn:cayu:managed-setting/v1",
        "properties": {
          "key": {"pattern": "^[a-z][a-z0-9_]*(\\.[a-z0-9_-]+)+$"}
        }
      }
    }
  }
}
```

`settings` is sorted by key with unique keys, the whole document is at most
256 KiB, and `digest` is computed as above. Cloud recomputes the digest and
refuses a mismatch, an over-limit or an invalid schema with 422, keeping the rest
of the registration. Older Releases leave the field out, and Cloud shows their
Agent settings as waiting for the Release to start.

### Effective values: `cayu.managed-settings-values/v1`

`GET /v1/applications/{id}/settings/effective?release={release_id}`, with
`Authorization: Bearer <registry token>`. Another Agent's token gets 404, and the
response is `Cache-Control: no-store`. If Cloud has no schema for the Release
yet, it answers 409 `{"code": "settings_schema_missing"}`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:cayu:managed-settings-values/v1",
  "title": "Effective Agent settings for one Release",
  "type": "object",
  "required": [
    "contract",
    "application_id",
    "release_id",
    "schema_digest",
    "revision",
    "valid_until",
    "values"
  ],
  "properties": {
    "contract": {"const": "cayu.managed-settings-values/v1"},
    "application_id": {"type": "string"},
    "release_id": {"type": "string"},
    "schema_digest": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
    "revision": {"type": "integer", "minimum": 0},
    "valid_until": {"type": "string", "format": "date-time"},
    "values": {
      "type": "object",
      "additionalProperties": {"type": ["boolean", "integer", "number", "string"]}
    },
    "conflicts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["key", "reason"],
        "properties": {
          "key": {"type": "string"},
          "reason": {
            "enum": ["unknown_key", "invalid_value", "type_changed", "secret_ref_missing"]
          }
        }
      }
    }
  }
}
```

`values` holds only the Agent settings set in Cloud that fit the Release's
schema; a key that isn't there takes the code default. Platform settings are
never in it. `revision` increases with every change to the Agent's settings, and
`valid_until` is about five minutes out.

### Applied report: `cayu.managed-settings-report/v1`

`POST /v1/applications/{id}/settings/reports`, with the same credential. Per
process, the highest `sequence` wins; Cloud answers 202 `{"sequence": 42}`,
ignores a lower sequence than the one it has, and treats a process silent for 10
minutes as gone.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "urn:cayu:managed-settings-report/v1",
  "title": "What one process applied",
  "type": "object",
  "required": [
    "contract",
    "release_id",
    "process",
    "sequence",
    "revision",
    "state",
    "settings",
    "reported_at"
  ],
  "properties": {
    "contract": {"const": "cayu.managed-settings-report/v1"},
    "release_id": {"type": "string"},
    "process": {
      "type": "object",
      "required": ["id", "role"],
      "properties": {
        "id": {"type": "string"},
        "role": {"enum": ["web", "worker"]}
      }
    },
    "sequence": {"type": "integer", "minimum": 0},
    "revision": {"type": ["integer", "null"], "minimum": 0},
    "state": {"enum": ["applied", "stale", "unreported"]},
    "settings": {
      "type": "array",
      "maxItems": 500,
      "items": {
        "type": "object",
        "required": ["key", "outcome"],
        "properties": {
          "key": {"type": "string"},
          "outcome": {"enum": ["adopted", "refused", "default"]},
          "reason": {"type": "string", "maxLength": 200}
        },
        "if": {"properties": {"outcome": {"const": "refused"}}},
        "else": {"not": {"required": ["reason"]}}
      }
    },
    "reported_at": {"type": "string", "format": "date-time"}
  },
  "if": {"properties": {"state": {"const": "unreported"}}},
  "then": {"properties": {"revision": {"type": "null"}}}
}
```

A report is at most 64 KiB. `process.id` is stable for the life of the process.
