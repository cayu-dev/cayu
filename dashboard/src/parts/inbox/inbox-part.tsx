/**
 * The Inbox: everything waiting on a person, filtered by kind and text, decided in place.
 * This is the preset: `InboxFilters` over an `InboxList`, each usable on its own. The
 * Developer dashboard's Inbox page is this part under its own top bar; an app shows it in
 * its own shell, and adds its own content around the items (`description`, `footer`).
 */

import { type ReactNode, useState } from "react"
import type { InboxSearch } from "../host"
import { InboxFilters } from "./inbox-filters"
import { InboxList } from "./inbox-list"

export function InboxPart({
  search,
  onSearch,
  description,
  advancedFilters = true,
  footer,
}: {
  search: InboxSearch
  /** Show other filters; the host navigates, or keeps them in state. */
  onSearch: (next: InboxSearch) => void
  /** What the page says above the filters. */
  description?: ReactNode
  /** The session, specialist and page-size filters. An app for one Agent can leave them out. */
  advancedFilters?: boolean
  /** The host's own content below the items. */
  footer?: ReactNode
}) {
  const [pageSize, setPageSize] = useState(50)
  return (
    <>
      {description}
      <InboxFilters
        search={search}
        onSearch={onSearch}
        advancedFilters={advancedFilters}
        pageSize={pageSize}
        onPageSize={setPageSize}
      />
      <InboxList search={search} pageSize={pageSize} />
      {footer}
    </>
  )
}
