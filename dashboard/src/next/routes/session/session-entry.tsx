import { useParams } from "@tanstack/react-router"
import { CapabilityRoute, useDashboardCapability } from "@/components/dashboard/server-contract"
import { SessionPage } from "./session-page"

export function SessionEntry() {
  const { sessionId } = useParams({ strict: false })
  const toolDisplay = useDashboardCapability({ kind: "surface", surface: "tool_display" })
  const workflow = useDashboardCapability({ kind: "surface", surface: "workflow" })
  const taskReplay = useDashboardCapability({ kind: "surface", surface: "task_creation" })
  const structured = useDashboardCapability({
    kind: "surface",
    surface: "structured_output_transcript",
  })
  const interruption = useDashboardCapability({
    kind: "mutation",
    mutation: "session_interruption",
  })
  const provider = useDashboardCapability({
    kind: "mutation",
    mutation: "provider_operation_resolution",
  })
  const artifacts = useDashboardCapability({ kind: "surface", surface: "artifacts" })
  const recovery = useDashboardCapability({
    kind: "mutation",
    mutation: "pending_action_resolution",
  })
  return (
    <CapabilityRoute title="Session" requirement={{ kind: "surface", surface: "session_overview" }}>
      <SessionPage
        key={sessionId}
        workflowEnabled={workflow.enabled}
        taskReplayEnabled={taskReplay.enabled}
        structuredOutputEnabled={structured.enabled}
        toolDisplayEnabled={toolDisplay.enabled}
        interruptionEnabled={interruption.enabled}
        providerResolutionEnabled={provider.enabled}
        recoveryEnabled={recovery.enabled}
        artifactsEnabled={artifacts.enabled}
      />
    </CapabilityRoute>
  )
}
