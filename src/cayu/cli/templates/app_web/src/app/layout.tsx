import { SidebarBrand } from "@kit/layouts/app";
import { DropdownMenuLabel, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuSeparator } from "@kit/ui";
import { AppNavShell, type AppNavGroup } from "@kit/layouts/app/router";
import { InboxIcon, WorkflowSquare01Icon } from "@kit/icons";
import { useTheme } from "@kit/hooks";
import { areaMatch, type AgentIdentity, type AppArea, type ShellParts } from "./area";
import { session, useWhoami } from "./services";
import { AppSignInAgainPage } from "./status-pages";

/**
 * GENERATED. The navigation: the Agent's areas, then links to Cayu's Developer dashboard and
 * its Inbox, where what waits for a person (approvals, questions) is decided.
 */
export function navigation(agent: AgentIdentity, areas: readonly AppArea[]): AppNavGroup[] {
  return [
    { label: agent.group, items: areas.map((area) => ({ label: area.label, icon: area.icon, to: area.path, match: areaMatch(area) })) },
    // Sessions, transcripts, traces, tasks and the inbox stay on Cayu's own dashboard; the app links there.
    {
      label: "Runtime",
      items: [
        { label: "Inbox", icon: InboxIcon, href: "/cayu/inbox" },
        { label: "Developer dashboard", icon: WorkflowSquare01Icon, href: "/cayu/" },
      ],
    },
  ];
}

/** GENERATED. The shell every page renders in. */
export function AppLayout({ agent, areas, shell = {} }: { agent: AgentIdentity; areas: readonly AppArea[]; shell?: ShellParts }) {
  const { me } = useWhoami();
  const { SignIn = AppSignInAgainPage, ProfileItems } = shell;
  if (session.useExpired()) return <SignIn />;
  const name = me ? (me.person.name ?? me.person.email ?? me.person.subject) : "Signed in";
  return (
    <AppNavShell
      name={agent.name}
      home="/"
      brand={<SidebarBrand name={agent.name} mark render={<span />} />}
      navigation={navigation(agent, areas)}
      profile={{
        name,
        children: (
          <>
            <ProfileMenu role={me?.role ?? null} />
            {ProfileItems ? <ProfileItems /> : null}
          </>
        ),
      }}
    />
  );
}

function ProfileMenu({ role }: { role: string | null }) {
  const { theme, setTheme } = useTheme();
  return (
    <>
      {role ? (
        <>
          <DropdownMenuLabel>Signed in as {role}</DropdownMenuLabel>
          <DropdownMenuSeparator />
        </>
      ) : null}
      <DropdownMenuRadioGroup
        value={theme}
        onValueChange={(value) => {
          if (value === "light" || value === "dark" || value === "system") setTheme(value);
        }}
      >
        <DropdownMenuLabel>Appearance</DropdownMenuLabel>
        <DropdownMenuRadioItem value="light" closeOnClick>
          Light
        </DropdownMenuRadioItem>
        <DropdownMenuRadioItem value="dark" closeOnClick>
          Dark
        </DropdownMenuRadioItem>
        <DropdownMenuRadioItem value="system" closeOnClick>
          System
        </DropdownMenuRadioItem>
      </DropdownMenuRadioGroup>
    </>
  );
}
