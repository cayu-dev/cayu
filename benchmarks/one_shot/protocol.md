# Benchmark protocol

Protocol version: **1**.

## Pin the run

Record the Cayu wheel SHA-256 and version, source commit, operating system,
Python version, coding-agent product/version, exact model/reasoning setting,
non-empty agent configuration and permissions, allowed tools, network policy,
elapsed-time ceiling, and cost ceiling. A no-limit setting must be recorded
explicitly; it is not an omitted field.

Use the report schema's typed limit objects. A bounded run records, for example,
`{"elapsed_time":{"mode":"limited","seconds":900},"cost":{"mode":"limited","amount":1,"currency":"USD"}}`.
An unbounded run records
`{"elapsed_time":{"mode":"no_limit"},"cost":{"mode":"no_limit"}}`.
Arbitrary limit keys or stringly typed `"no-limit"` values are invalid.

## Prepare a fresh trial

1. Create a clean virtual environment and install only the built Cayu wheel plus
   dependencies required by the generated project.
2. Run `cayu new application` in an empty directory.
3. Initialize a fresh Git repository and commit the untouched scaffold.
4. Place the selected case prompt outside the repository. Give the agent the
   product request as its only initial instruction.
5. Do not expose the Cayu checkout, other benchmark results, VibeCoder, or
   framework-specific advice through parent instructions, memory, skills, or
   environment files.

Global coding skills are allowed only when they are ordinary agent defaults and
contain no Cayu-specific knowledge. Record their names and versions.

## Clarification

The agent may ask one bounded batch of product-domain questions before editing.
Return only the predetermined answers in the case. Record every question and
answer. Questions about how Cayu works receive no custom answer; the agent must
use the generated/package-shipped public surfaces.

## First submission

The first submission is the repository state when the agent reports completion
or reaches its configured limit. Stop the agent before running the acceptance
suite. Commit or archive the exact state. Do not feed failures back to the agent
until the scored result has been recorded. Store a non-empty Git diff/patch or
safe zip/tar archive as `submission_artifact` inside the trial's
`submission_path`; an empty directory is not a submission.

## Verification layers

Capture output and exit status separately for:

1. `cayu inspect --json`;
2. `cayu check --json`;
3. generated/project tests;
4. trajectory evals;
5. a clean process using the built wheel;
6. optional live provider/environment checks.

Store each executed layer and each case requirement in a distinct file beneath
`SUBMISSION_PATH/evidence/`. Evidence paths must resolve within that directory;
one shared file cannot stand in for unrelated claims within or across trials.
Every trial uses its own submission directory.

Construction, inspection, scripted models, mocks, and local runners never count
as live verification. Mark unavailable optional checks `not_run`; do not turn
them into implied success.

## Grade and publish

Grade behavior through public outputs and repository contents. Search the diff
for private Cayu imports and app-local reimplementations of Cayu session,
approval, budget, artifact, recovery, and verification behavior. Treat any
security-critical violation as a failed trial even when other checks pass.

Publish all trial reports and first-submission artifacts. Classify every failure
with an entry in the report schema's non-empty `failures` array:

- `classification` records discoverability, module depth/caller lifecycle,
  invalid assembly accepted, cryptic diagnostic, unrealistic test boundary, or
  missing capability;
- `disposition` records whether the failure became a regression fixture, an
  authoring/diagnostic improvement, or a linked capability issue;
- `reference` is a typed, validated Cayu repository path, issue URL, or pull
  request URL naming the resulting fixture or tracked work. Regression fixtures
  require repository paths, linked capabilities require issue URLs, and
  authoring/diagnostic improvements require repository paths or pull request
  URLs.

The schema requires at least one record when the trial declares a failed first
submission and can preserve multiple independent causes from the same trial.
The scorer also requires classified dispositions when artifact, evidence,
isolation, or policy validation turns a claimed pass into a scored failure, and
it verifies that repository fixture paths exist in the Cayu checkout.
