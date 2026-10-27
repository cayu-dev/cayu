/**
 * What fresh trials run with, the judge, the limits, what this deployment supports, and the
 * advanced tools that used to crowd the Evals page header.
 */
import { Check, ChevronRight, X } from "lucide-react"
import type { ReactNode } from "react"
import { PrototypeLink as NavLink } from "../app/prototype-navigation"
import { useEvalSettings } from "../app/queries"
import { ErrorState, PageBody, TopBar } from "./common"

function Section({
  title,
  note,
  children,
}: {
  title: string
  note?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="mt-8">
      <h2 className="text-sm font-medium">{title}</h2>
      {note && <p className="mt-0.5 text-xs text-muted-foreground">{note}</p>}
      <div className="mt-2">{children}</div>
    </section>
  )
}

const ADVANCED: { name: string; what: string; how: string }[] = [
  {
    name: "Import or export a corpus",
    what: "Move suites between agents or into CI as a portable JSON corpus.",
    how: "cayu eval export · cayu eval merge",
  },
  {
    name: "Judge calibration",
    what: "Score fixed evidence repeatedly with the judge and against human scores, without running the agent.",
    how: "POST /api/evals/judge-calibrations",
  },
  {
    name: "Memory reports",
    what: "Compare memory variants across paired runs, with gates and a recommendation.",
    how: "cayu eval memory-report",
  },
  {
    name: "Production scenarios",
    what: "Replay a multi-step session, including its approvals and follow-up inputs, as one case.",
    how: "Add to evals on a multi-step session",
  },
]

export function EvalSettingsPage() {
  const settings = useEvalSettings()
  const s = settings.data
  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <TopBar>
        <span className="flex min-w-0 items-center gap-1">
          <NavLink to="/evals" className="shrink-0 text-muted-foreground hover:text-foreground">
            Evals
          </NavLink>
          <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
          <span className="truncate">Settings</span>
        </span>
      </TopBar>
      <PageBody width="max-w-3xl">
        {settings.error && <ErrorState error={settings.error} />}
        {s && (
          <>
            <Section
              title="What trials run with"
              note="Each trial starts a fresh session with the specialist as it is configured now."
            >
              <div className="overflow-hidden rounded-xl border border-border">
                {s.candidate.map((c) => (
                  <div
                    key={c.specialist}
                    className="flex flex-wrap items-center gap-x-3 border-b border-border px-3 py-2 text-sm last:border-0"
                  >
                    <span className="font-medium">{c.specialist}</span>
                    <span className="text-muted-foreground">
                      {c.model}
                      {c.provider ? ` via ${c.provider}` : ""}
                      {c.environment ? ` · ${c.environment}` : ""}
                    </span>
                  </div>
                ))}
              </div>
            </Section>

            <Section
              title="Judge"
              note={
                <>
                  Set in <code className="font-mono">[tool.cayu.evals.default_judge]</code>.
                </>
              }
            >
              {s.judge ? (
                <p className="text-sm">
                  {s.judge.model} via {s.judge.provider}
                  <span className="text-muted-foreground">
                    {s.judge.same_model_allowed
                      ? " · may judge its own model"
                      : " · never judges the model it's grading"}
                  </span>
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No judge configured, so judge checks can't run.
                </p>
              )}
            </Section>

            <Section title="Limits">
              <p className="text-sm">
                Up to {s.limits.max_trials} trials per case, {s.limits.max_concurrency} at a time,
                each stopping after {Math.round(s.limits.timeout_seconds / 60)} minutes or{" "}
                {s.limits.max_steps} model steps.
              </p>
            </Section>

            <Section title="Available in this deployment">
              <ul className="space-y-1.5 text-sm">
                {s.features.map((f) => (
                  <li key={f.name} className="flex gap-2">
                    {f.ready ? (
                      <Check
                        className="mt-0.5 size-4 shrink-0 text-success"
                        aria-label="Available"
                      />
                    ) : (
                      <X
                        className="mt-0.5 size-4 shrink-0 text-warning"
                        aria-label="Not available"
                      />
                    )}
                    <span>
                      {f.name}
                      {f.note && (
                        <span className="block text-xs text-muted-foreground">{f.note}</span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </Section>

            <Section
              title="Advanced"
              note="Still in today's Evals tools; the prototype doesn't rework these yet."
            >
              <div className="overflow-hidden rounded-xl border border-border">
                {ADVANCED.map((a) => (
                  <div
                    key={a.name}
                    className="border-b border-border px-3 py-2.5 text-sm last:border-0"
                  >
                    <p className="font-medium">{a.name}</p>
                    <p className="mt-0.5 text-muted-foreground">{a.what}</p>
                    <p className="mt-1 font-mono text-xs text-muted-foreground">{a.how}</p>
                  </div>
                ))}
              </div>
            </Section>
          </>
        )}
      </PageBody>
    </div>
  )
}
