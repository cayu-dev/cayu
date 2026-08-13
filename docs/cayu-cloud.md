# Cayu Cloud CLI

The `cayu cloud` command group deploys and operates complete Cayu Agent applications.
Cayu Cloud is currently invite-only. Login-backed commands are pinned to the production
service at `https://cloud.cayu.dev`; users never select an API URL. Run the remaining
commands from an Agent repository.

```console
cayu cloud login
cayu cloud whoami
cayu cloud deploy .
cayu cloud deployment timeline DEPLOYMENT_ID --application my-agent
cayu cloud deployment logs DEPLOYMENT_ID --application my-agent
cayu cloud service status --application my-agent
```

`deployment timeline` shows the release pipeline milestones and whether a failed
release can be retried or an active release can be cancelled. `deployment logs`
returns the Cloud-owned structured publication activity for that exact immutable
release. Both commands use authenticated, tenant-scoped customer API responses and
emit the same machine-readable JSON envelope as the rest of the Cloud CLI.

`cayu cloud deploy` creates the 8-63 character application slug declared in
`cayu-cloud.toml` when it does not exist, then updates it on later deploys. Slugs use
lowercase letters, numbers, and interior hyphens. `--application SLUG`
selects a different create-or-update slug; check it carefully because a valid typo
creates a separate application.

Deploy verifies that the local evidence directory is writable before authentication or
Cloud mutation. Deploy output and stored evidence replace every runtime `environment`
value with `[redacted]`, including values whose variable names do not look secret.

If Cloud deployment and rollout succeed but the final local evidence write fails, the
remote success remains authoritative: the command exits `0`, returns the successful
deployment result, and includes this machine-readable evidence status:

```json
{
  "evidence": {
    "category": "local_state_unavailable",
    "message": "Deployment succeeded, but local evidence could not be recorded.",
    "status": "unavailable"
  },
  "evidence_id": null,
  "operation": "deploy",
  "result": {}
}
```

The `result` object above is abbreviated; the real response retains the redacted
successful deployment result. An unusable evidence destination discovered during
preflight instead exits `2` with `local_state_unavailable`, before Cloud mutation.

Login uses WorkOS device authorization. Cayu opens the complete verification URL when
possible and prints the URL and one-time user code to standard error. Pass
`--no-browser` on SSH, in a container, or when a coding agent should ask a human to
complete authentication in another browser:

```console
cayu cloud login --no-browser
```

The access and rotating refresh tokens are kept in a private local auth file. Cayu
refreshes the short-lived access token before Cloud API calls. `cayu cloud logout`
deletes that local login.

Authentication selection is explicit: a successful `cayu cloud login` clears the
persisted private-context selection, while `cayu cloud context use PATH` selects that
context for later commands. Clearing the context falls back to the saved WorkOS login
and the fixed production endpoint. A saved login for any other endpoint is rejected;
run `cayu cloud login` again to sign in to production.

Private contexts and API keys remain the internal noninteractive path for CI,
operational handoffs, and automation:

```console
cayu cloud context use /private/path/cloud-context.json
CAYU_CLOUD_API_KEY_FILE=/private/path/key cayu cloud doctor
```

## Agent environment variables

Cloud-managed variables belong to the long-lived Agent, not to one immutable release.
They override same-named values from `cayu-cloud.toml` and are applied to the web
process, worker, and schedules when the Agent service rolls forward.

Plain values may be supplied as an assignment:

```console
cayu cloud env set MODE=demo --application my-agent
```

Secrets are write-only and must come from a file or standard input. The value is never
accepted as a command-line argument and is never returned by the API:

```console
cayu cloud env set VAPI_API_KEY --secret \
  --value-file /private/path/vapi-key \
  --application my-agent

printf '%s' "$VAPI_API_KEY" | \
  cayu cloud env set VAPI_API_KEY --secret --value-file - --application my-agent
```

List or remove configuration with:

```console
cayu cloud env list --application my-agent
cayu cloud env unset VAPI_API_KEY --application my-agent
```

Set and unset responses include the Agent service rollout status when a live service
is being updated.
