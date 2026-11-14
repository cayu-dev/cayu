import { Settings01Icon } from "@kit/icons";
import type { AppArea } from "../../app/area";
import { SettingsPage } from "./page";

/** GENERATED. Settings: the Agent's managed settings, where each value comes from, and the ones that need a redeploy. */
export const settingsArea: AppArea = {
  id: "settings",
  label: "Settings",
  icon: Settings01Icon,
  path: "/settings",
  routes: () => [{ path: "settings", Component: SettingsPage }],
};
