export type RunAgentIdentity = {
  name: string
}

/**
 * Preserve an explicit selection while it remains registered. A single-agent
 * application is unambiguous and can be selected automatically; multi-agent
 * applications require the operator to choose.
 */
export function reconcileRunAgentSelection(
  currentAgentName: string,
  agents: readonly RunAgentIdentity[],
): string {
  if (currentAgentName !== "" && agents.some((agent) => agent.name === currentAgentName)) {
    return currentAgentName
  }
  return agents.length === 1 ? (agents[0]?.name ?? "") : ""
}
