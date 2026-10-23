import type { SessionOverviewResponse } from "@/lib/generated/server-api"
import type { SessionOverview } from "./types"

/** The canonical server owns every fact; numbers here are presentation values only. */
export function overviewForDisplay(overview: SessionOverviewResponse): SessionOverview {
  return {
    ...overview,
    cost: overview.cost ? { ...overview.cost, amount: Number(overview.cost.amount) } : null,
    model_steps: Number(overview.model_steps),
    tool_calls: Number(overview.tool_calls),
    tokens: {
      input: Number(overview.tokens.input),
      output: Number(overview.tokens.output),
      cached: Number(overview.tokens.cached),
    },
    uses: {
      ...overview.uses,
      subagents: Number(overview.uses.subagents),
      artifacts: Number(overview.uses.artifacts),
    },
    eval_trial: overview.eval,
  }
}
