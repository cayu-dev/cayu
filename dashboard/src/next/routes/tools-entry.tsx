import { useDashboardCapability } from "@/components/dashboard/server-contract"
import { ToolsPage } from "./tools-page"
export function ToolsEntry() {
  const usage = useDashboardCapability({ kind: "surface", surface: "tool_usage" })
  return <ToolsPage usageEnabled={usage.enabled} />
}
