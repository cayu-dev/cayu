import { Dialog } from "@base-ui/react/dialog"
import { useQuery } from "@tanstack/react-query"
import { useBlocker, useParams, useRouter, useSearch } from "@tanstack/react-router"
import { useEffect, useRef, useState } from "react"
import { EvalSuiteEditor } from "@/components/dashboard/eval-suite-editor"
import { Button } from "@/components/ui/button"
import { fetchEvalTargets } from "@/lib/api"
import { EVAL_TARGET_QUERY_KEY } from "@/lib/evals-dashboard"
import { fetchSuiteView } from "../../api/eval-views"
import {
  isNativeSuiteDetail,
  isReportedSuiteDetail,
  nativeRunSelector,
  runViewPath,
  suiteSelector,
  suiteViewPath,
} from "../../app/eval-view-model"
import { NavLink, PageBody, TopBar, useGo } from "../common"
import { FIELD_CLASS, LoadState, useEvalAccess } from "./shared"

type DraftOwner = { target: string; revision?: string; scope: string }

export function EvalSuiteEditorPage() {
  const { canWrite } = useEvalAccess()
  return canWrite ? (
    <EditorPage />
  ) : (
    <>
      <TopBar>Suite editor</TopBar>
      <PageBody>
        <p role="status">Suite authoring requires full access and an available eval catalog.</p>
      </PageBody>
    </>
  )
}

function EditorPage() {
  const { suiteId } = useParams({ strict: false }) as { suiteId?: string }
  const search = useSearch({ strict: false }) as { target?: string; case?: string; run?: boolean }
  const { readiness } = useEvalAccess()
  const go = useGo()
  const router = useRouter()
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])
  const targets = useQuery({
    queryKey: EVAL_TARGET_QUERY_KEY,
    queryFn: ({ signal }) => fetchEvalTargets(signal),
  })
  const suite = useQuery({
    queryKey: ["evals", "editor-suite", suiteId],
    queryFn: ({ signal }) => fetchSuiteView(suiteId!, {}, signal),
    enabled: !!suiteId,
    retry: false,
  })
  const native = isNativeSuiteDetail(suite.data) ? suite.data : undefined
  const reported = isReportedSuiteDetail(suite.data)
  const identity = native?.suite.identity
  const target = suiteId
    ? identity?.target_key
    : (search.target ?? targets.data?.default_target_key)
  const knownTarget = targets.data?.items.some((item) => item.target_key === target)
  const authored = !suiteId || identity?.source === "authored"
  const revision = suiteId && authored ? identity?.revision : undefined
  const scope =
    target && knownTarget && authored && (!suiteId || revision)
      ? JSON.stringify([target, revision ?? null])
      : undefined
  const [owners, setOwners] = useState<DraftOwner[]>([])
  const [dirty, setDirty] = useState<Record<string, boolean>>({})
  const [scopeLocked, setScopeLocked] = useState(false)
  const [navigationLocked, setNavigationLocked] = useState(false)
  const [savedViews, setSavedViews] = useState<Record<string, string>>({})
  if (scope && target && !owners.some((owner) => owner.scope === scope)) {
    setOwners([...owners, { target, revision, scope }])
  }
  const hasUnsaved = Object.values(dirty).some(Boolean)
  const blocker = useBlocker({
    shouldBlockFn: ({ current, next }) =>
      navigationLocked || hasUnsaved || (scopeLocked && current.pathname === next.pathname),
    enableBeforeUnload: hasUnsaved,
    withResolver: true,
  })
  const switchingTarget =
    blocker.status === "blocked" && blocker.next.pathname === blocker.current.pathname
  const origin = router.state.location.href
  const unavailable = readiness.catalog_write.state !== "ready"
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <TopBar>
        <NavLink to="/evals" className="mr-2 text-muted-foreground">
          Evals /
        </NavLink>
        {suiteId ? "Edit suite" : "New suite"}
      </TopBar>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2 text-sm">
        {suiteId ? (
          <NavLink to={suiteViewPath(suiteId)} className="underline">
            Back to suite
          </NavLink>
        ) : (
          <label className="flex min-w-0 items-center gap-2">
            Specialist
            <select
              className={`${FIELD_CLASS} max-w-full`}
              value={target ?? ""}
              disabled={!targets.data || scopeLocked}
              aria-label="Specialist"
              onChange={(event) => void go("/evals/suites/new", { target: event.target.value })}
            >
              {!knownTarget && (
                <option value={target ?? ""}>{target ?? "Choose specialist"}</option>
              )}
              {targets.data?.items.map((item) => (
                <option key={item.target_key} value={item.target_key}>
                  {item.target_key}
                </option>
              ))}
            </select>
          </label>
        )}
        <NavLink to="/evals/settings" className="underline">
          Eval settings
        </NavLink>
      </div>
      {scope && savedViews[scope] && (
        <div className="px-4 py-2 text-sm">
          <NavLink to={suiteViewPath(savedViews[scope])} className="underline">
            View saved suite
          </NavLink>
        </div>
      )}
      <div className="px-4">
        <LoadState error={targets.error} pending={targets.isPending} retry={targets.refetch} />
        {suiteId && (
          <LoadState error={suite.error} pending={suite.isPending} retry={suite.refetch} />
        )}
        {targets.data && target && !knownTarget && (
          <p role="alert">This specialist is not available.</p>
        )}
        {suite.data && !authored && (
          <p role="status">
            {reported
              ? "Reported series are read-only. Run the source program and publish a new report."
              : "Corpus suites are read-only here. Import a revised corpus from Eval settings."}
          </p>
        )}
        {unavailable && (
          <p role="status">
            The eval catalog is not currently writable. Your draft is retained here.
          </p>
        )}
      </div>
      {owners.map((owner) => (
        <EvalSuiteEditor
          key={owner.scope}
          targetKey={owner.target}
          initialRevision={owner.revision}
          initialCaseId={search.case}
          initialRunReview={search.run === true}
          active={owner.scope === scope}
          disabled={unavailable}
          onSaved={(saved) =>
            setSavedViews((current) => ({
              ...current,
              [owner.scope]: suiteSelector({
                source: "authored",
                target_key: saved.target_key,
                revision: saved.revision,
                suite_id: null,
              }),
            }))
          }
          onScopeLockedChange={setScopeLocked}
          onNavigationLockedChange={setNavigationLocked}
          onDirtyChange={(value) =>
            setDirty((current) =>
              current[owner.scope] === value ? current : { ...current, [owner.scope]: value },
            )
          }
          onLaunched={async (ids) => {
            if (alive.current && router.state.location.href === origin && ids[0])
              await go(runViewPath(nativeRunSelector(ids[0])))
          }}
        />
      ))}
      <Dialog.Root
        open={blocker.status === "blocked"}
        onOpenChange={(open) => {
          if (!open) blocker.reset?.()
        }}
      >
        <Dialog.Portal>
          <Dialog.Backdrop className="fixed inset-0 z-50 bg-black/40" />
          <Dialog.Popup className="fixed top-1/2 left-1/2 z-50 w-[min(28rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 space-y-4 rounded-xl border border-border bg-background p-5 shadow-xl">
            <Dialog.Title className="font-semibold">Leave this editor?</Dialog.Title>
            <Dialog.Description className="text-sm text-muted-foreground">
              {navigationLocked
                ? "Finish or retry the unconfirmed calibration before leaving. Leaving would lose its exact paid-run recovery."
                : switchingTarget && scopeLocked
                  ? "Finish the current operation, or save or discard scenario edits before switching specialists."
                  : "You have unsaved changes or an unconfirmed save. Leaving this page discards its drafts and exact save retry. Switching specialists on this page keeps their drafts until you leave."}
            </Dialog.Description>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => blocker.reset?.()}>
                Keep editing
              </Button>
              <Button
                disabled={navigationLocked || (switchingTarget && scopeLocked)}
                onClick={() => blocker.proceed?.()}
              >
                Leave editor
              </Button>
            </div>
          </Dialog.Popup>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  )
}
