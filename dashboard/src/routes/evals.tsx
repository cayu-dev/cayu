import { Page, PageHeader, StateMessage } from "@/components/dashboard/layout"

export function EvalsPage() {
  return (
    <Page>
      <PageHeader
        title="Evals"
        description="Manage portable regression corpora and durable fresh evaluation runs."
      />
      <StateMessage className="rounded-lg border border-border bg-muted/30 py-12">
        The Evals catalog is loading.
      </StateMessage>
    </Page>
  )
}
