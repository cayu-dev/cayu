import { useQuery } from "@tanstack/react-query"
import { useSearch } from "@tanstack/react-router"
import { Settings2 } from "lucide-react"
import { useEffect } from "react"
import { EvalSuiteAuthoringAction } from "@/components/dashboard/eval-suite-authoring"
import { fetchEvalTargets } from "@/lib/api"
import { EVAL_TARGET_QUERY_KEY } from "@/lib/evals-dashboard"
import { fetchSuiteViews, resolveLegacyEvalSuite } from "../../api/eval-views"
import { runOrigin, runViewPath, suiteSelector, suiteViewPath } from "../../app/eval-view-model"
import { NavLink, PageBody, ROW_CLASS, ROW_LINK_CLASS, TopBar, useGo, useRowClick } from "../common"
import {
  DateText,
  FIELD_CLASS,
  LoadState,
  Pages,
  PassingBar,
  useEvalAccess,
  useEvalCursor,
  Verdict,
} from "./shared"

export function EvalsIndexPage() {
  const search = useSearch({ strict: false }) as {
    target?: string
    corpus?: string
    suite?: string
  }
  const go = useGo()
  const rowClick = useRowClick()
  const { canWrite, readiness } = useEvalAccess()
  const [cursor, setCursor] = useEvalCursor(JSON.stringify([search.target ?? null]))
  const suites = useQuery({
    queryKey: ["evals", "views", search.target, cursor],
    queryFn: ({ signal }) => fetchSuiteViews({ target_key: search.target, cursor }, signal),
  })
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
    enabled: canWrite,
  })
  const target = search.target ?? targets.data?.default_target_key
  const legacy = useQuery({
    queryKey: ["evals", "legacy-suite", search.corpus, search.suite, search.target],
    queryFn: ({ signal }) =>
      resolveLegacyEvalSuite(search.corpus!, search.suite!, search.target, signal),
    enabled: !!search.corpus && !!search.suite,
  })
  useEffect(() => {
    if (legacy.data) void go(suiteViewPath(legacy.data), undefined, { replace: true })
  }, [legacy.data, go])
  if (search.corpus && search.suite)
    return (
      <>
        <TopBar>Evals</TopBar>
        <PageBody>
          <LoadState error={legacy.error} pending={legacy.isPending} retry={legacy.refetch} />
        </PageBody>
      </>
    )
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar
        actions={
          <>
            <NavLink
              to="/evals/settings"
              aria-label="Eval settings"
              className="rounded-md p-2 hover:bg-muted"
            >
              <Settings2 className="size-4" />
            </NavLink>
            {canWrite && target && (
              <EvalSuiteAuthoringAction
                targetKey={target}
                disabled={readiness.catalog_write.state !== "ready"}
              />
            )}
          </>
        }
      >
        Evals
      </TopBar>
      <PageBody width="max-w-6xl">
        <p className="mt-1 text-sm text-muted-foreground">
          Suites, and how their latest runs compare with an approved baseline.
        </p>
        {canWrite && Object.values(readiness).some((operation) => operation.state !== "ready") && (
          <p className="mt-3 text-sm text-muted-foreground">
            Some eval features aren't available here.{" "}
            <NavLink to="/evals/settings" className="underline">
              See settings
            </NavLink>
          </p>
        )}
        {canWrite && targets.data && targets.data.items.length > 1 && (
          <label className="mt-4 flex items-center gap-2 text-sm">
            Specialist
            <select
              className={FIELD_CLASS}
              value={search.target ?? ""}
              onChange={(e) => {
                setCursor(undefined)
                void go("/evals", { target: e.target.value || undefined })
              }}
            >
              <option value="">All specialists</option>
              {targets.data.items.map((t) => (
                <option key={t.target_key} value={t.target_key}>
                  {t.target_key}
                </option>
              ))}
            </select>
          </label>
        )}
        <LoadState
          error={suites.error}
          pending={suites.isPending}
          retry={() => (cursor ? setCursor(undefined) : void suites.refetch())}
        />
        {canWrite && <LoadState error={targets.error} pending={false} retry={targets.refetch} />}
        {suites.data && (
          <>
            {suites.data.items.length === 0 ? (
              <div className="py-14 text-center">
                <h2 className="font-medium">No eval suites yet</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Create a suite or capture a finished session with Add to evals.
                </p>
              </div>
            ) : (
              <div className="mt-4 overflow-x-auto rounded-xl border border-border">
                <table className="w-full min-w-[760px] text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs text-muted-foreground">
                      {["Suite", "Cases", "Last run", "Against baseline", "Baseline"].map((h) => (
                        <th key={h} className="px-3 py-2 font-medium">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {suites.data.items.map((s) => (
                      <tr key={s.id} className={ROW_CLASS} onClick={rowClick(suiteViewPath(s.id))}>
                        <td className="max-w-80 px-3 py-2">
                          <NavLink
                            to={suiteViewPath(s.id)}
                            className={`${ROW_LINK_CLASS} block truncate`}
                          >
                            {s.name}
                          </NavLink>
                          <p className="mt-0.5 truncate text-xs text-muted-foreground">
                            {s.specialist} · {s.source === "authored" ? "Written here" : "Corpus"} ·{" "}
                            {s.trials.trial_count} trials
                          </p>
                        </td>
                        <td className="px-3 py-2 tabular-nums">{s.case_count}</td>
                        <td className="px-3 py-2">
                          {s.last_run ? (
                            <>
                              <NavLink
                                to={runViewPath(s.last_run.id)}
                                search={
                                  s.last_run.placement
                                    ? { placement: suiteSelector(s.last_run.placement.destination) }
                                    : undefined
                                }
                              >
                                <PassingBar cases={s.last_run.cases} />
                              </NavLink>
                              <p className="mt-1 text-xs text-muted-foreground">
                                <DateText value={s.last_run.started_at ?? s.last_run.created_at} />{" "}
                                ·{" "}
                                {s.last_run.placement
                                  ? "Scored session · one captured case"
                                  : runOrigin(s.last_run)}
                              </p>
                            </>
                          ) : (
                            <span className="text-muted-foreground">Not run yet</span>
                          )}
                        </td>
                        <td className="px-3 py-2">{s.last_run && <Verdict run={s.last_run} />}</td>
                        <td className="max-w-56 px-3 py-2 text-xs text-muted-foreground">
                          {s.baseline ? (
                            <>
                              {s.baseline.updated_by}
                              <br />
                              <DateText value={s.baseline.updated_at} />
                            </>
                          ) : s.baseline_scope === "execution_group" ? (
                            "Per execution group"
                          ) : (
                            "Not approved"
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <Pages
              label="Suites"
              cursor={cursor}
              busy={suites.isFetching}
              hasMore={suites.data.has_more ?? false}
              onFirst={() => setCursor(undefined)}
              onNext={() => setCursor(suites.data.next_cursor ?? undefined)}
            />
          </>
        )}
      </PageBody>
    </div>
  )
}
