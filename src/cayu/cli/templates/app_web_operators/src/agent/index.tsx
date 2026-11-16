import type { AgentIdentity, AppArea, ShellParts } from "../app/area";
import { overviewArea } from "../generated/overview";
import { settingsArea } from "../generated/settings";
import { taskRunsArea } from "./task-runs";

/**
 * THE AGENT'S OWN. What the generated shell shows for this Agent: its name, its areas in
 * navigation order (the generated Overview and Settings, and its own Task runs), and what it
 * adds to the shell. An area of its own is an `AppArea` in this folder, its navigation entry
 * and its routes; listing it here adds it. The shell adds the links to Cayu's Inbox and
 * Developer dashboard, and, with no sign-in page in `shell`, asks a person whose sign-in
 * ended to reload and sign in again: its people are the Agent's operators, who sign in as
 * they do to the Developer dashboard.
 */
export const identity: AgentIdentity = {
  // The page's title, which `cayu new` set to the project's name in index.html.
  name: document.title,
  group: "Agent",
  description: "What the Agent does, and what waits for a person",
};

export const areas: AppArea[] = [overviewArea({ description: "The Agent's metrics, and what waits for a person." }), taskRunsArea, settingsArea];

export const shell: ShellParts = {};
