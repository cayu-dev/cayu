# Cayu fresh-agent one-shot benchmark

Protocol and rubric version: **1**.

This benchmark measures whether a fresh coding agent can build a complete Cayu
vertical slice from public, package-shipped surfaces on its first submission.
It is a release gate, not a curated demo.

Use the built wheel under test. Each trial starts in a clean directory and may
see only:

- the case prompt;
- predetermined clarification answers, only after one bounded clarification batch;
- the repository produced by `cayu new`;
- `AGENTS.md`, `cayu guide`, public imports/introspection, and public CLI output.

The agent must not see the Cayu checkout, Cayu history/tests, VibeCoder notes,
prior trial output, benchmark acceptance failures, or a hand-written Cayu
prompt. Do not repair the first submission before recording it.

Run at least three fresh trials for every case. Pin and publish the agent/model,
reasoning configuration, tool permissions, wall-clock and cost ceilings, wheel
hash/version, prompt, clarification transcript, first-submission commit/diff,
command output, and evidence report. Publish failures as well as successes.
The versioned report schema requires an explicit reasoning setting, non-empty
agent configuration and permissions, and typed elapsed-time/cost limits. Both
bounded and `no_limit` modes are machine-validated.

The required cases are:

- [`cases/rfp.md`](cases/rfp.md)
- [`cases/research-document.md`](cases/research-document.md)
- [`cases/coding-repository.md`](cases/coding-repository.md)

Record trials using [`trial-report.schema.json`](trial-report.schema.json), then
score the combined report:

```bash
python scripts/score_one_shot_benchmark.py path/to/report.json
```

The report is evidence-indexed rather than a list of self-reported booleans.
Each verification layer records its command, exit status, and a non-empty
output file. Each case-specific requirement has its own evidence file, and the
scorer verifies the exact requirement set. Every trial has a non-empty
`submission_artifact` inside its `submission_path`: a Git diff/patch or a safe
zip/tar archive of the untouched first submission. Evidence files are distinct
per claim and live below that trial's `evidence/` directory; symlinks cannot
escape either scope, and submission/evidence identities cannot be reused across
trial IDs. Optional live checks use `not_run` plus a reason; they must not
contain invented command or exit data. The scorer validates report structure,
path scope, and evidence indexing; human grading under the protocol validates
whether submission and evidence contents actually prove the claimed behavior.
The scorer validates the versioned JSON Schema before grading. Every failed
trial records one or more structured classifications, dispositions, and typed
follow-up references so independent benchmark failures become reproducible
tests, authoring improvements, or linked capability work instead of disappearing
into aggregate scores.
Scorer output separates ordinary `trial_failures`, which affect the published
rates, from release-gate `violations` such as missing classifications, report
integrity failures, human hints, private interfaces, and security violations.

Passing requires at least 80% aggregate first-submission success, at least two
passes in every three trials for each archetype, zero framework-specific human
hints, and zero private-interface or security violations across all trials.

See [`protocol.md`](protocol.md) for isolation, execution, and grading rules and
[`rubric.md`](rubric.md) for the behavior/framework-leverage rubric.
