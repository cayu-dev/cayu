/** The fake server behind today's unchanged pages: the contract, then each area's routes. */

import type { MockApi } from "../api"
import { installArtifactLinks } from "./artifact-links"
import { CONTRACT } from "./contract"
import { coreRoutes } from "./core"
import { installLegacyServer, type LegacyRoute } from "./server"

const contractRoutes: LegacyRoute[] = [{ method: "GET", path: "/contract", handle: () => CONTRACT }]

export function installLegacyApi(api: MockApi) {
  const context = { records: api.records, now: api.now }
  installLegacyServer([...contractRoutes, ...coreRoutes], context)
  const disposeLinks = installArtifactLinks(context)
  import.meta.hot?.dispose(disposeLinks)
}
