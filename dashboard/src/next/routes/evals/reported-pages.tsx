/** External evidence has its own inspection flow and no native execution controls. */
import { useMutation } from "@tanstack/react-query"
import { Download } from "lucide-react"
import { PayloadViewer } from "@/components/dashboard/layout"
import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { downloadEvalReport } from "@/lib/api"
import type {
  EvalReportedRunDetail,
  EvalReportedRunView,
  EvalReportedSuiteDetail,
} from "@/lib/generated/server-api"
import type { EvalCaseFilter } from "../../api/eval-views"
import {
  evalCaseFilter,
  evidenceText,
  runOrigin,
  suiteSelector,
  suiteViewRoute,
} from "../../app/eval-view-model"
import { NavLink, PageBody, TopBar } from "../common"
import {
  DateText,
  FIELD_CLASS,
  LoadState,
  Pages,
  PassingBar,
  RunsTable,
  Section,
  useEvalAccess,
} from "./shared"
import { downloadBlob } from "./technical-details"

type PageState = {
  cursor?: string
  setCursor: (value: string | undefined) => void
  busy: boolean
  error: Error | null
  retry: () => unknown
}

function ReportOrigin({ run }: { run: EvalReportedRunView }) {
  const origin = run.origin
  const facts = [
    ["Publisher", run.publisher_id],
    ["Program", run.suite.program_key],
    ["Repository", origin.repository],
    ["Branch", origin.branch],
    ["Tested commit", origin.commit],
    ["PR head commit", origin.pr_head_commit],
    ["Pull request", origin.pull_request == null ? null : `#${origin.pull_request}`],
    ["CI run", origin.run_id],
    ["Attempt", origin.run_attempt],
    ["Job", origin.job],
    ["Declared suite revision", run.declared_suite_revision ?? "Unknown"],
  ]
  return (
    <Section title="Origin">
      <p className="mb-3 text-xs text-muted-foreground">
        The publisher is authenticated. Repository and build details are supplied by the reporting
        job.
      </p>
      <dl className="grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
        {facts
          .filter(([, value]) => value != null)
          .map(([label, value]) => (
            <div key={label} className="contents">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="min-w-0 break-words [overflow-wrap:anywhere]">{value}</dd>
            </div>
          ))}
        <dt className="text-muted-foreground">Received</dt>
        <dd>
          <DateText value={run.created_at} />
        </dd>
        {origin.run_url && (
          <>
            <dt className="text-muted-foreground">Run URL</dt>
            <dd>
              <a
                href={origin.run_url}
                target="_blank"
                rel="noopener noreferrer"
                className="break-words underline [overflow-wrap:anywhere]"
              >
                Open CI run
              </a>
            </dd>
          </>
        )}
      </dl>
    </Section>
  )
}

function ReportNotice() {
  return (
    <p className="mt-4 rounded-xl border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
      Reported evidence is read-only. Comparisons and baseline approval are unavailable for these
      runs.
    </p>
  )
}

export function ReportedSuitePage({
  detail,
  cursor,
  setCursor,
  busy,
  error,
  retry,
}: PageState & { detail: EvalReportedSuiteDetail }) {
  const suite = detail.suite
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>
        <NavLink to="/evals" className="mr-2 text-muted-foreground">
          Evals /
        </NavLink>
        <span className="truncate">{suite.name}</span>
      </TopBar>
      <PageBody width="max-w-5xl">
        <h1 className="text-xl font-semibold break-words">{suite.name}</h1>
        <p className="mt-1 text-sm text-muted-foreground break-words">
          {suite.specialist} · Reported series · {suite.identity.publisher_id} ·{" "}
          {suite.identity.program_key}
        </p>
        <ReportNotice />
        <LoadState error={error} pending={false} retry={retry} />
        <Section title="Runs">
          <RunsTable runs={detail.runs.items} />
          <Pages
            label="Runs"
            cursor={cursor}
            busy={busy}
            hasMore={detail.runs.has_more ?? false}
            onFirst={() => setCursor(undefined)}
            onNext={() => setCursor(detail.runs.next_cursor ?? undefined)}
          />
        </Section>
        <p className="mt-4 text-sm text-muted-foreground">
          Open a run to inspect its retained cases, declared revision, and build metadata.
        </p>
      </PageBody>
    </div>
  )
}

export function ReportedRunPage({
  detail,
  cursor,
  setCursor,
  busy,
  error,
  retry,
  filter,
  changeSearch,
  selectedId,
}: PageState & {
  detail: EvalReportedRunDetail
  filter?: EvalCaseFilter
  selectedId?: string
  changeSearch: (patch: { filter?: EvalCaseFilter; case?: string }) => void
}) {
  const run = detail.run
  const { canReadDetails } = useEvalAccess()
  const selected = detail.results.items.find((item) => item.id === selectedId)
  const download = useMutation({
    mutationFn: async () => {
      if (!canReadDetails) throw new Error("Report downloads require developer access.")
      const file = await downloadEvalReport(run.id)
      downloadBlob(file.blob, file.filename)
    },
  })
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          canReadDetails && (
            <Button
              size="sm"
              variant="outline"
              disabled={download.isPending}
              onClick={() => download.mutate()}
            >
              <Download />
              {download.isPending ? "Downloading…" : "Download report"}
            </Button>
          )
        }
      >
        <NavLink to="/evals" className="mr-2 text-muted-foreground">
          Evals /
        </NavLink>
        <span className="truncate">Reported run</span>
      </TopBar>
      <PageBody width="max-w-5xl">
        <h1 className="text-xl font-semibold break-words">{run.suite.suite_id}</h1>
        <p className="mt-1 text-sm text-muted-foreground break-words">
          {run.specialist} · {runOrigin(run)} · {run.status}
        </p>
        <ReportNotice />
        <LoadState
          error={error ?? download.error}
          pending={false}
          retry={() => {
            download.reset()
            retry()
          }}
        />
        <div className="mt-4 flex flex-wrap gap-4 text-sm">
          <PassingBar cases={run.cases} />
          <span>Score: {run.score ?? "Unavailable"}</span>
        </div>
        {run.association && (
          <p className="mt-3 text-sm">
            <NavLink {...suiteViewRoute(suiteSelector(run.association))} className="underline">
              Exact associated corpus suite
            </NavLink>
          </p>
        )}
        <ReportOrigin run={run} />
        <Section title={`Cases (${run.cases.total})`}>
          <label className="mb-3 flex items-center gap-2 text-sm">
            Show
            <select
              aria-label="Case filter"
              value={filter ?? ""}
              className={FIELD_CLASS}
              onChange={(e) =>
                changeSearch({ filter: evalCaseFilter(e.target.value), case: undefined })
              }
            >
              <option value="">All cases</option>
              <option value="failing">Failing</option>
              <option value="flaky">Flaky</option>
              {(filter === "regressed" || filter === "improved") && (
                <option value={filter}>Comparison unavailable</option>
              )}
            </select>
          </label>
          <div className="overflow-hidden rounded-xl border border-border">
            {detail.results.items.map((item) => (
              <button
                type="button"
                key={item.id}
                onClick={() => changeSearch({ case: item.id })}
                className="flex w-full flex-wrap items-center justify-between gap-3 border-b border-border px-3 py-3 text-left text-sm last:border-0 hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
              >
                <span className="min-w-0 flex-1 break-words [overflow-wrap:anywhere]">
                  {item.id}
                </span>
                <span className="text-muted-foreground">
                  {item.status} · {item.trial_count} trials
                </span>
              </button>
            ))}
          </div>
          {!detail.results.items.length && (
            <p className="py-3 text-sm text-muted-foreground">
              No retained cases match this filter.
            </p>
          )}
          <Pages
            label="Cases"
            cursor={cursor}
            busy={busy}
            hasMore={detail.results.has_more ?? false}
            onFirst={() => setCursor(undefined)}
            onNext={() => setCursor(detail.results.next_cursor ?? undefined)}
          />
        </Section>
        {selected && (
          <Sheet
            open
            onOpenChange={(open) => {
              if (!open) changeSearch({ case: undefined })
            }}
          >
            <SheetContent className="overflow-y-auto data-[side=right]:w-[min(96vw,48rem)] data-[side=right]:sm:max-w-none">
              <SheetHeader className="pr-12">
                <SheetTitle className="break-words">{selected.id}</SheetTitle>
                <SheetDescription>
                  {selected.status} · Score: {selected.score ?? "Unavailable"} ·{" "}
                  {selected.trial_count} trials · Reported evidence · No comparison
                </SheetDescription>
              </SheetHeader>
              <div className="space-y-5 p-4">
                {selected.evidence_omitted && (
                  <p className="text-sm text-muted-foreground">
                    This case is too large to display here. Its status, score and trial count are
                    preserved.{" "}
                    {canReadDetails
                      ? "Close this panel and download the report for complete evidence."
                      : "Ask a developer to download the report for complete evidence."}
                  </p>
                )}
                {selected.evidence?.trials.map((trial) => (
                  <section key={trial.trial_number} className="space-y-2">
                    <h3 className="text-sm font-medium">
                      Trial {trial.trial_number} · {trial.status}
                    </h3>
                    <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-muted p-3 text-xs [overflow-wrap:anywhere]">
                      {evidenceText("output" in trial ? trial.output.text : trial.final_output)}
                    </pre>
                    <pre className="whitespace-pre-wrap break-words text-xs [overflow-wrap:anywhere]">
                      {trial.assertions
                        .map(
                          (check) =>
                            `${"assertion_id" in check ? check.assertion_id : check.name}: ${"code" in check ? check.code : check.outcome}${check.message ? ` · ${check.message}` : ""}`,
                        )
                        .join("\n")}
                    </pre>
                    <details className="text-xs">
                      <summary className="cursor-pointer">Retained trial evidence</summary>
                      <PayloadViewer value={trial} />
                    </details>
                  </section>
                ))}
              </div>
            </SheetContent>
          </Sheet>
        )}
      </PageBody>
    </div>
  )
}
