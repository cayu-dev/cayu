import { useDashboardCapability } from "@/components/dashboard/server-contract"
import { SpecialistsPage } from "./specialists-page"
export function SpecialistsEntry() {
  const recent = useDashboardCapability({ kind: "surface", surface: "session_overview_queries" })
  const tasks = useDashboardCapability({ kind: "surface", surface: "tasks" })
  return <SpecialistsPage recentEnabled={recent.enabled} tasksEnabled={tasks.enabled} />
}
