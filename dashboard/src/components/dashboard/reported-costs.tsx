import type { ReportedCostPage } from "../../lib/generated/server-api"
import { reportedCostText } from "../../lib/reported-cost"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../ui/table"
import { DataCard, StateMessage } from "./layout"

export function ReportedCosts({ page }: { page: ReportedCostPage | null | undefined }) {
  const records = page?.records ?? []
  return (
    <DataCard
      title="Provider-reported cost"
      description="Completion-time observations, separate from PriceBook estimates. Not current wallet balances or a summed bill. Generation lookup may report a later amount; this view does not poll the provider."
    >
      {page == null ? (
        <StateMessage>This store does not expose reported-cost observations.</StateMessage>
      ) : (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Session</TableHead>
                <TableHead>Request</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Reported cost</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {records.map((row) => (
                <TableRow key={`${row.session_id}:${row.event_id}`}>
                  <TableCell className="break-all">{row.session_id}</TableCell>
                  <TableCell className="break-all">{row.request_id ?? "Unavailable"}</TableCell>
                  <TableCell>{row.model ?? "Unknown"}</TableCell>
                  <TableCell>{reportedCostText(row)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {!records.length && (
            <StateMessage>No reported-cost observations in this scope.</StateMessage>
          )}
          {page.truncated && (
            <StateMessage>
              Only the latest 100 observations are shown. Narrow the time range or session filters.
            </StateMessage>
          )}
        </>
      )}
    </DataCard>
  )
}
