# Research and document analysis

## Initial prompt

Build an agent that analyzes a supplied market report and produces a cited risk brief.

## Predetermined clarification answers

- One local analyst supplies document artifact IDs and a research question.
- Preserve the source document as an artifact; the output brief is also a durable artifact.
- The model may extract claims and synthesize risks but must cite the supplied source.
- No workflow, task queue, human approval, server, memory, or multi-agent topology is required.
- Acceptance is credential-free; use deterministic document/tool behavior and a trajectory eval.

## Case-specific acceptance

The vertical slice models document input and artifact output, verifies citations
and final artifact creation, and omits unrelated workflow/task/approval/server/
memory/multi-agent machinery.
