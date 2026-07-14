# Coding and repository agent

## Initial prompt

Build an agent that reviews a repository change, runs targeted checks, and prepares a patch for human review.

## Predetermined clarification answers

- The input identifies a local Git repository and target revision.
- Work in an isolated mutable workspace; never edit the source checkout directly.
- The agent may read files and run a narrow test/lint command policy.
- It may prepare a patch artifact but must not commit, push, or open a pull request.
- No task queue, server, persistent memory, or multi-agent topology is required.
- Acceptance uses a fixture repository and deterministic scripted model.

## Case-specific acceptance

The vertical slice uses an explicit repository workspace/runner boundary, narrow
command authority, durable patch artifact, deterministic runtime test, and eval.
It contains no unrestricted command execution and no delivery side effect.
