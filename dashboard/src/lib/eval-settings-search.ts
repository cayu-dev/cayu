export const EVAL_SETTINGS_TOOLS = [
  ["profiles", "Execution profiles"],
  ["judges", "Judges and calibration"],
  ["scenarios", "Production scenarios"],
  ["memory", "Memory reports"],
  ["portability", "Import / export"],
] as const

export type EvalSettingsTool = (typeof EVAL_SETTINGS_TOOLS)[number][0]

export function validateEvalSettingsSearch(search: Record<string, unknown>): {
  tool: EvalSettingsTool
  target?: string
} {
  return {
    tool: EVAL_SETTINGS_TOOLS.find(([key]) => key === search.tool)?.[0] ?? "profiles",
    target:
      typeof search.target === "string" && search.target.length > 0 && search.target.length <= 512
        ? search.target
        : undefined,
  }
}
