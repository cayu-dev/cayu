import type { EvalTarget } from "./api.ts"
import type { ScenarioLaunchSettingsV2 } from "./generated/server-api/index.ts"

export type ScenarioSettingsDraft = {
  environmentName: string
  trials: string
  maxConcurrency: string
  timeoutSeconds: string
  maxSteps: string
  maxTotalTokens: string
  maxToolCalls: string
  maxElapsedSeconds: string
  maxEstimatedCost: string
  currency: string
  artifactReferences: Record<string, string>
}

export const DEFAULT_SCENARIO_SETTINGS: ScenarioSettingsDraft = Object.freeze({
  environmentName: "",
  trials: "1",
  maxConcurrency: "1",
  timeoutSeconds: "300",
  maxSteps: "",
  maxTotalTokens: "",
  maxToolCalls: "",
  maxElapsedSeconds: "",
  maxEstimatedCost: "",
  currency: "USD",
  artifactReferences: {},
})

export const MAX_SAFE_SCENARIO_RUNTIME_LIMIT = Number.MAX_SAFE_INTEGER

function optionalPositiveInteger(value: string, maximum: number): number | null | undefined {
  if (value.trim() === "") return undefined
  const parsed = Number(value)
  if (!Number.isSafeInteger(parsed) || parsed < 1 || parsed > maximum) return null
  return parsed
}

export function scenarioLaunchSettingsContract(
  settings: ScenarioSettingsDraft,
  target: EvalTarget | undefined,
): ScenarioLaunchSettingsV2 | null {
  if (target === undefined) return null

  const trials = Number(settings.trials)
  const maxConcurrency = Number(settings.maxConcurrency)
  const timeoutSeconds = Number(settings.timeoutSeconds)
  const maxSteps = optionalPositiveInteger(settings.maxSteps, 256)
  const maxTotalTokens = optionalPositiveInteger(
    settings.maxTotalTokens,
    MAX_SAFE_SCENARIO_RUNTIME_LIMIT,
  )
  const maxToolCalls = optionalPositiveInteger(
    settings.maxToolCalls,
    MAX_SAFE_SCENARIO_RUNTIME_LIMIT,
  )
  const maxElapsedSeconds = optionalPositiveInteger(
    settings.maxElapsedSeconds,
    MAX_SAFE_SCENARIO_RUNTIME_LIMIT,
  )
  if (
    !Number.isInteger(trials) ||
    trials < 1 ||
    trials > target.max_trials ||
    !Number.isInteger(maxConcurrency) ||
    maxConcurrency < 1 ||
    maxConcurrency > target.max_concurrency ||
    !Number.isInteger(timeoutSeconds) ||
    timeoutSeconds < 1 ||
    timeoutSeconds > target.max_timeout_seconds ||
    maxSteps === null ||
    maxTotalTokens === null ||
    maxToolCalls === null ||
    maxElapsedSeconds === null
  ) {
    return null
  }
  const hasLimits =
    maxTotalTokens !== undefined || maxToolCalls !== undefined || maxElapsedSeconds !== undefined
  const cost = settings.maxEstimatedCost.trim()
  let costBudget: ScenarioLaunchSettingsV2["cost_budget"]
  if (cost !== "") {
    const currency = settings.currency.trim().toUpperCase()
    if (
      !/^(?:0|[1-9]\d*)(?:\.\d+)?$/.test(cost) ||
      Number(cost) <= 0 ||
      target.cost_budget_available !== true ||
      !target.cost_budget_currencies.includes(currency)
    ) {
      return null
    }
    costBudget = { max_estimated_cost: cost, currency }
  }
  const artifactReferences = Object.fromEntries(
    Object.entries(settings.artifactReferences)
      .map(([key, value]) => [key, value.trim()])
      .filter(([, value]) => value !== ""),
  )
  return {
    environment_name: settings.environmentName.trim() || null,
    trials,
    max_concurrency: maxConcurrency,
    timeout_seconds: timeoutSeconds,
    max_steps: maxSteps ?? null,
    limits: hasLimits
      ? {
          scope: "run",
          max_total_tokens: maxTotalTokens,
          max_tool_calls: maxToolCalls,
          max_elapsed_seconds: maxElapsedSeconds,
        }
      : null,
    cost_budget: costBudget ?? null,
    artifact_references: artifactReferences,
  }
}
