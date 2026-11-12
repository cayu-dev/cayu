import { Link } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { AppPageSection } from "@kit/layouts/app";
import { DataTable } from "@kit/patterns";
import { Button, StatusBadge } from "@kit/ui";
import { formatDateTime, formatRelative } from "@kit/lib";
import { queryTableProps } from "@kit/data";
import { useAreaApi } from "../app/services";
import { b2bApi, titleOf } from "./api";
import { conversationPath, conversationsQuery, SharingBadge } from "./conversations";
import { useMembership } from "./organization";

/** THE AGENT'S OWN. Its sections on the generated Overview: recent conversations, and where the person belongs. */

const RECENT = 5;

export function RecentConversations() {
  const api = useAreaApi(b2bApi);
  const { membership } = useMembership();
  const organization = membership?.organization ?? null;
  const recent = useQuery({ ...conversationsQuery(api, organization, RECENT), enabled: organization !== null });
  return (
    <AppPageSection
      title="Recent conversations"
      description={organization ? `The latest in ${organization} that you can see.` : "The latest you can see."}
      actions={
        <span className="flex gap-2">
          <Button variant="outline" size="sm" render={<Link to="/conversations" />}>
            All conversations
          </Button>
          <Button size="sm" render={<Link to="/conversations/new" />}>
            New conversation
          </Button>
        </span>
      }
    >
      <DataTable
        {...queryTableProps(recent, "Couldn't load the conversations.")}
        label="Recent conversations"
        showHeading={false}
        showCount={false}
        rows={recent.data?.sessions ?? []}
        rowId={(row) => row.id}
        itemLabel={{ singular: "conversation", plural: "conversations" }}
        emptyTitle="No conversations yet"
        emptyDescription="Start one, or ask someone to share theirs."
        columns={[
          {
            key: "title",
            label: "Conversation",
            cell: (row) => (
              <Link to={conversationPath(row.id)} className="font-medium underline-offset-4 hover:underline">
                {titleOf(row)}
              </Link>
            ),
          },
          { key: "sharing", label: "Shared with", cell: (row) => <SharingBadge labels={row.labels} /> },
          { key: "owner", label: "Owner", cell: (row) => row.labels.owner ?? "—" },
          {
            key: "activity",
            label: "Last activity",
            cell: (row) => (
              <time dateTime={row.last_activity_at ?? undefined} title={formatDateTime(row.last_activity_at)}>
                {formatRelative(row.last_activity_at)}
              </time>
            ),
          },
        ]}
      />
    </AppPageSection>
  );
}

export function YourOrganizations() {
  const { me, memberships } = useMembership();
  return (
    <AppPageSection
      title="Your organizations"
      description="Where you belong decides what you see: your own conversations, your department's and your organization's."
      actions={
        <Button variant="outline" size="sm" render={<Link to="/members" />}>
          Members
        </Button>
      }
    >
      <DataTable
        {...queryTableProps(me, "Couldn't load your memberships.")}
        label="Your organizations"
        showHeading={false}
        showCount={false}
        rows={memberships}
        rowId={(row) => row.organization}
        itemLabel={{ singular: "organization", plural: "organizations" }}
        emptyTitle="No organization yet"
        emptyDescription="No organization has added you. Ask one of its admins."
        columns={[
          { key: "organization", label: "Organization", cell: (row) => <span className="font-medium">{row.organization}</span> },
          { key: "department", label: "Department", cell: (row) => row.department ?? "—" },
          { key: "role", label: "Your role", cell: (row) => <StatusBadge tone={row.role === "admin" ? "success" : "neutral"}>{row.role}</StatusBadge> },
        ]}
      />
    </AppPageSection>
  );
}
