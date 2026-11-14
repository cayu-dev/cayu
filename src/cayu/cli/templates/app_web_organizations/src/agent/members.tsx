import * as React from "react";
import { queryOptions, useQuery, useQueryClient } from "@tanstack/react-query";
import { AppPageHeader, AppPageSection, AppPageTitle } from "@kit/layouts/app";
import { DataTable } from "@kit/patterns";
import { Button, Card, CardContent, Field, FieldLabel, Input, NativeSelect, NativeSelectOption, StatusBadge, toast } from "@kit/ui";
import { queryTableProps, whoamiKey } from "@kit/data";
import { UserGroupIcon } from "@kit/icons";
import type { AppArea } from "../app/area";
import { useAreaApi } from "../app/services";
import { b2bApi, type B2bApi, messageOf, ROLES, type Role } from "./api";
import { ErrorNote } from "./conversations";
import { membershipBlocker, meQuery, OrganizationSelect, useMembership } from "./organization";

/**
 * THE AGENT'S OWN. The active organization's members. Everyone in it reads the list; its
 * admins add, move and remove members. The server checks the admin role on every change and
 * keeps at least one admin.
 */
export const membersArea: AppArea = {
  id: "members",
  label: "Members",
  icon: UserGroupIcon,
  path: "/members",
  routes: () => [{ path: "members", Component: MembersPage }],
};

export const membersQuery = (api: B2bApi, organization: string) =>
  queryOptions({ queryKey: ["b2b", "members", organization], queryFn: ({ signal }) => api.members(organization, signal) });

function MembersPage() {
  const { me, membership } = useMembership();
  const blocked = membershipBlocker(me, membership);
  if (blocked) return blocked;
  const admin = membership?.role === "admin";
  return (
    <div className="flex flex-col gap-page-section" data-origin="own">
      <AppPageHeader
        description={
          membership
            ? admin
              ? `The people in ${membership.organization}. Removing someone takes effect at their next step, in background work too.`
              : `The people in ${membership.organization}. Only its admins add, move and remove members.`
            : null
        }
        actions={<OrganizationSelect />}
      >
        <AppPageTitle>Members</AppPageTitle>
      </AppPageHeader>
      {membership ? <MemberList key={membership.organization} organization={membership.organization} admin={admin} /> : null}
    </div>
  );
}

function MemberList({ organization, admin }: { organization: string; admin: boolean }) {
  const api = useAreaApi(b2bApi);
  const queryClient = useQueryClient();
  const members = useQuery(membersQuery(api, organization));
  // The member whose removal is waiting for a second click.
  const [confirming, setConfirming] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  // Whether the change was made.
  const change = async (action: () => Promise<unknown>, done: string): Promise<boolean> => {
    try {
      await action();
      setError(null);
      toast.add({ title: done, type: "success" });
      await queryClient.invalidateQueries({ queryKey: membersQuery(api, organization).queryKey });
      // An admin may have changed their own role.
      await queryClient.invalidateQueries({ queryKey: meQuery(api).queryKey });
      await queryClient.invalidateQueries({ queryKey: whoamiKey });
      return true;
    } catch (caught) {
      setError(messageOf(caught));
      return false;
    }
  };

  return (
    <>
      <DataTable
        {...queryTableProps(members, "Couldn't load the members.")}
        label="Members"
        showHeading={false}
        rows={members.data ?? []}
        rowId={(row) => row.subject}
        searchText={(row) => `${row.subject} ${row.department ?? ""}`}
        itemLabel={{ singular: "member", plural: "members" }}
        emptyTitle="No members"
        emptyDescription="Nobody is in this organization."
        columns={[
          { key: "subject", label: "Person", sortValue: (row) => row.subject, cell: (row) => <span className="font-medium">{row.subject}</span> },
          { key: "department", label: "Department", sortValue: (row) => row.department ?? "", cell: (row) => row.department ?? "—" },
          {
            key: "role",
            label: "Role",
            sortValue: (row) => row.role,
            cell: (row) => <StatusBadge tone={row.role === "admin" ? "success" : "neutral"}>{row.role}</StatusBadge>,
          },
          ...(admin
            ? [
                {
                  key: "remove",
                  label: "",
                  cell: (row: { subject: string }) =>
                    confirming === row.subject ? (
                      <span className="flex justify-end gap-2">
                        <Button variant="ghost" size="sm" onClick={() => setConfirming(null)}>
                          Cancel
                        </Button>
                        {/* The default variant: the two kits name their danger variants differently. */}
                        <Button
                          size="sm"
                          aria-label={`Confirm removing ${row.subject}`}
                          onClick={() => change(() => api.removeMember(organization, row.subject), `Removed ${row.subject}`).then((made) => made && setConfirming(null))}
                        >
                          Confirm remove
                        </Button>
                      </span>
                    ) : (
                      <span className="flex justify-end">
                        <Button variant="outline" size="sm" aria-label={`Remove ${row.subject}`} onClick={() => setConfirming(row.subject)}>
                          Remove
                        </Button>
                      </span>
                    ),
                },
              ]
            : []),
        ]}
      />
      {admin ? <AddMember organization={organization} onChange={change} /> : null}
      {error ? <ErrorNote message={error} /> : null}
    </>
  );
}

function AddMember({ organization, onChange }: { organization: string; onChange: (action: () => Promise<unknown>, done: string) => Promise<boolean> }) {
  const api = useAreaApi(b2bApi);
  const [subject, setSubject] = React.useState("");
  const [department, setDepartment] = React.useState("");
  const [role, setRole] = React.useState<Role>("member");
  const submit = async () => {
    const person = subject.trim();
    if (!(await onChange(() => api.putMember(organization, person, department.trim() || null, role), `Saved ${person}`))) return;
    setSubject("");
    setDepartment("");
    setRole("member");
  };
  return (
    <AppPageSection title="Add or change a member" description="Adding someone already in the organization moves them to this department and role.">
      <Card>
        <CardContent className="flex flex-wrap items-end gap-3">
          <Field className="w-56">
            <FieldLabel>Person's ID</FieldLabel>
            <Input value={subject} placeholder="Their sign-in ID" onChange={(event) => setSubject(event.target.value)} />
          </Field>
          <Field className="w-64">
            <FieldLabel>Department (optional)</FieldLabel>
            <Input value={department} placeholder={`e.g. ${organization}/finance`} onChange={(event) => setDepartment(event.target.value)} />
          </Field>
          <Field className="w-36">
            <FieldLabel>Role</FieldLabel>
            <NativeSelect value={role} onChange={(event) => setRole(event.target.value as Role)}>
              {ROLES.map((option) => (
                <NativeSelectOption key={option} value={option}>
                  {option}
                </NativeSelectOption>
              ))}
            </NativeSelect>
          </Field>
          <Button disabled={!subject.trim()} onClick={submit}>
            Add or update
          </Button>
        </CardContent>
      </Card>
    </AppPageSection>
  );
}
