import { NavLink, sessionRoute, taskRoute, useGo, useRowClick } from "../../src/next/routes/common"

export function NavigationContract() {
  const go = useGo()
  // @ts-expect-error The production registry has no such route.
  const unknownRoute = <NavLink to="/missing-dashboard-route" />
  // @ts-expect-error Inbox search does not support arbitrary keys.
  const unknownSearch = <NavLink to="/inbox" search={{ made_up: "value" }} />
  // @ts-expect-error Search values must keep their registered types.
  const wrongSearchType = <NavLink to="/inbox" search={{ kind: 123 }} />
  // @ts-expect-error Dynamic routes require their registered parameters.
  const missingParameters = <NavLink to="/sessions/$sessionId" />
  // @ts-expect-error Navigation is checked against the production registry.
  void go({ to: "/missing-dashboard-route" })
  // @ts-expect-error Navigation search remains route-specific.
  void go({ to: "/inbox", search: { made_up: "value" } })
  // @ts-expect-error Dynamic identity must be a string.
  void go({ to: "/sessions/$sessionId", params: { sessionId: 123 } })
  // @ts-expect-error Tasks search accepts only its registered keys.
  void go({ to: "/tasks", search: { made_up: "value" } })
  // @ts-expect-error Eval case filters are the registered closed set.
  void go({ to: "/evals/runs/$runId", params: { runId: "run" }, search: { filter: "unknown" } })
  const rowClick = useRowClick()
  // @ts-expect-error Row navigation uses the same registered contract.
  rowClick({ to: "/missing-dashboard-route" })
  // @ts-expect-error A task identity cannot stand in for the Session parameter.
  const wrongParameter = <NavLink to="/sessions/$sessionId" params={{ taskId: "wrong" }} />
  void go({ ...sessionRoute("raw / % ? café"), search: { view: "trace" }, replace: true })
  rowClick(taskRoute("raw task"))
  const validSession = <NavLink {...sessionRoute("raw / % ? café")} search={{ event: "event" }} />
  const validDefaultSettings = <NavLink to="/evals/settings" />
  return (
    <>
      {unknownRoute}
      {unknownSearch}
      {wrongSearchType}
      {missingParameters}
      {wrongParameter}
      {validSession}
      {validDefaultSettings}
    </>
  )
}
