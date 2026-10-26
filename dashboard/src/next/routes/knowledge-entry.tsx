import { useDashboardCapability } from "@/components/dashboard/server-contract"
import { dashboardCapabilityUnavailableText } from "@/lib/dashboard-capabilities"
import { KnowledgePage } from "./knowledge-page"

export function KnowledgeEntry() {
  const kept = useDashboardCapability({ kind: "surface", surface: "kept_knowledge" })
  const review = useDashboardCapability({ kind: "surface", surface: "inbox" })
  return (
    <KnowledgePage
      keptUnavailable={dashboardCapabilityUnavailableText(kept)}
      reviewUnavailable={dashboardCapabilityUnavailableText(review)}
    />
  )
}
