import { z } from "zod";
import type { Requester } from "@kit/data";

/**
 * GENERATED. The Settings area, `GET /app/settings` (integrations/http/generated/settings.py),
 * built from the Agent's managed settings. The live
 * settings are Cayu managed settings: they're changed in Cayu Cloud (`cayu cloud
 * settings set`, or the portal), not here; the web app has no way to change them as
 * the person yet.
 */

const value = z.union([z.boolean(), z.number(), z.string()]).nullable();

export const liveSetting = z.object({
  name: z.string(),
  // The managed setting's key, and the environment variable that also sets it (Cayu's own
  // settings have none).
  key: z.string(),
  env_var: z.string().nullable(),
  group: z.string(),
  label: z.string(),
  help: z.string(),
  // When a change takes effect.
  applies: z.string(),
  // bool, int, number, choice (one of `choices`), or text.
  kind: z.enum(["bool", "int", "number", "choice", "text"]),
  choices: z.array(z.string()),
  minimum: z.number().nullable(),
  maximum: z.number().nullable(),
  blankOk: z.boolean(),
  // Null when the setting has no value: an unset limit, or a value refused (`error` says why).
  value,
  // Where the value comes from: managed (set in Cayu Cloud), environment or default.
  source: z.string(),
  environment: z.string().nullable(),
  default: value,
  note: z.string(),
  error: z.string().nullable(),
  // The `cayu cloud settings` commands that set it to its value, and clear it.
  change: z.string(),
  reset: z.string(),
});
export type LiveSettingRow = z.infer<typeof liveSetting>;

export const fixedSetting = z.object({
  name: z.string(),
  env_var: z.string().nullable(),
  label: z.string(),
  help: z.string(),
  value,
  source: z.string().nullable(),
  // "redeploy", or "not a setting yet".
  needs: z.string(),
  why: z.string(),
  warning: z.string().nullable(),
});
export type FixedSettingRow = z.infer<typeof fixedSetting>;

export const settingsView = z.object({
  settings: z.array(liveSetting),
  read_only: z.array(fixedSetting),
  // Whether the Agent reads Cayu Cloud's managed settings (false outside Cloud).
  managed: z.boolean(),
  // Whether Cayu Cloud's values are in use: state applied, stale or unreported (never read),
  // and what the last read did; both null outside Cloud.
  cloud: z.object({ state: z.string().nullable(), status: z.string().nullable() }),
  // Where the change history is: Cayu Cloud keeps it; the web app can't read it yet.
  history: z.string(),
});
export type SettingsView = z.infer<typeof settingsView>;

export function settingsApi(request: Requester) {
  return { get: (signal?: AbortSignal) => request("settings", { signal, schema: settingsView }) };
}

export type SettingsApi = ReturnType<typeof settingsApi>;
