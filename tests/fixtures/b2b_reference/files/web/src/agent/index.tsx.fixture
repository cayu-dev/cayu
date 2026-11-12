import type { AgentIdentity, AppArea, ShellParts } from "../app/area";
import { overviewArea } from "../generated/overview";
import { settingsArea } from "../generated/settings";
import { conversationsArea } from "./conversations";
import { membersArea } from "./members";
import { RecentConversations, YourOrganizations } from "./overview";
import { ProfileItems, SignIn } from "./sign-in";

/**
 * THE AGENT'S OWN. What the generated shell shows for this Agent: its name, its areas in
 * navigation order (the generated Overview and Settings, and its own Conversations and
 * Members), and its own sign-in page for `cayu serve --dev`. The shell adds the links to
 * Cayu's Inbox and Developer dashboard.
 *
 * No area lists task runs: this Agent queues no Cayu tasks of its own yet. When it does, an
 * area built on the TaskRuns block (src/blocks/task-runs.tsx) reads `GET /app/task-runs`.
 */
export const identity: AgentIdentity = {
  // The page's title, which `cayu new` set to the project's name in index.html.
  name: document.title,
  group: "Assistant",
  description: "An assistant several organizations share",
};

export const areas: AppArea[] = [
  overviewArea({
    description: "Your organizations, the conversations you can see, and what waits for a person.",
    sections: [RecentConversations, YourOrganizations],
  }),
  conversationsArea,
  membersArea,
  settingsArea,
];

export const shell: ShellParts = { SignIn, ProfileItems };
