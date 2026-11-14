import { queryOptions, useQuery } from "@tanstack/react-query";
import {
  AppPageDescription,
  AppPageHeader,
  AppPageSection,
  AppPageTitle,
  LiveSettingControl,
  LiveSettingRow,
  LiveSettingsForm,
  SettingsRow,
  SettingsSection,
  useLiveSettings,
} from "@kit/layouts/app";
import { AsyncContent } from "@kit/patterns";
import { Alert, AlertDescription, AlertTitle, CodeBlock, Skeleton, StatusBadge } from "@kit/ui";
import { queryContentProps } from "@kit/data";
import { settingsApi, type FixedSettingRow, type LiveSettingRow as Row, type SettingsApi, type SettingsView } from "../../api/settings";
import { useAreaApi } from "../../app/services";

export const settingsQuery = (api: SettingsApi) => queryOptions({ queryKey: ["settings"], queryFn: ({ signal }) => api.get(signal) });

type Value = boolean | number | string | null;

/** A value as the Agent's CLI writes it: on or off, a number, or the text. */
export const show = (value: Value) => (value === null ? "—" : typeof value === "boolean" ? (value ? "on" : "off") : value === "" ? "(empty)" : String(value));

/**
 * GENERATED. The Agent's settings: the live ones with their values and where each comes from, and
 * the ones that need a redeploy. Live settings are Cayu managed settings, changed in
 * Cayu Cloud, which checks the person's role and keeps who changed what; the web app
 * shows them and the command that changes each, and can't change them itself yet.
 */
export function SettingsPage() {
  const api = useAreaApi(settingsApi);
  const settings = useQuery(settingsQuery(api));
  return (
    <div className="flex flex-col gap-page-section" data-origin="generated">
      <AppPageHeader>
        <div className="flex flex-col gap-1">
          <AppPageTitle>Settings</AppPageTitle>
          <AppPageDescription>
            Each setting is set in Cayu Cloud, which checks your role and keeps who changed what. The Agent reads a change within 25 seconds, and each
            setting says when it applies.
          </AppPageDescription>
        </div>
      </AppPageHeader>
      <AsyncContent {...queryContentProps(settings, "Couldn't load the settings.")} skeleton={<Skeleton className="h-96 w-full" />}>
        {settings.data ? <SettingsContent view={settings.data} /> : null}
      </AsyncContent>
    </div>
  );
}

function SettingsContent({ view }: { view: SettingsView }) {
  return (
    <div className="flex flex-col gap-page-section">
      {view.managed ? null : (
        <Alert variant="info">
          <AlertTitle>Not on Cayu Cloud</AlertTitle>
          <AlertDescription>This Agent reads no managed settings, so each value is the release's own.</AlertDescription>
        </Alert>
      )}
      <CloudNotice cloud={view.cloud} />
      <LiveSettings rows={view.settings} />
      <AppPageSection
        title="Change a setting"
        description="Run these with the cayu CLI, signed in to Cayu Cloud as an admin or developer, or change the setting in the Cayu Cloud portal. Each sets the value shown; edit it first. Unset goes back to the release's value."
      >
        <CodeBlock label="Set" language="sh" wrap>
          {view.settings.map((row) => row.change).join("\n")}
        </CodeBlock>
        <CodeBlock label="Unset" language="sh" wrap>
          {view.settings.map((row) => row.reset).join("\n")}
        </CodeBlock>
        <p className="text-body-sm text-text-secondary">
          Who changed what: <code className="wrap-anywhere">{view.history}</code>.
        </p>
      </AppPageSection>
      {view.read_only.length ? <AppPageSection title="Need a redeploy" description="Read once when a process starts, so a change applies on the next deploy.">
        <SettingsSection>
          {view.read_only.map((row) => (
            <FixedRow key={row.name} row={row} />
          ))}
        </SettingsSection>
      </AppPageSection> : null}
    </div>
  );
}

/** Says when Cayu Cloud's values aren't in use: never read (the values shown are without them), or not read lately. */
function CloudNotice({ cloud }: { cloud: SettingsView["cloud"] }) {
  if (cloud?.state === "unreported") {
    return (
      <Alert variant="warning">
        <AlertTitle>Cayu Cloud's settings couldn't be read</AlertTitle>
        <AlertDescription>
          A value set in Cayu Cloud doesn't apply yet; each value shown is the one without it.
          {cloud.status === "schema_missing" ? " Cloud has no settings schema for this release yet." : ""}
        </AlertDescription>
      </Alert>
    );
  }
  if (cloud?.state === "stale") {
    return (
      <Alert variant="warning">
        <AlertTitle>Cayu Cloud hasn't answered lately</AlertTitle>
        <AlertDescription>For over five minutes; the values shown are the last ones it gave.</AlertDescription>
      </Alert>
    );
  }
  return null;
}

/** Where the value in effect comes from: Cayu Cloud, the environment variable, or the default. */
const SOURCES: Record<string, string> = { managed: "Cayu Cloud", environment: "Environment", default: "Default" };

/** The live settings, read-only, in the kit's live-settings rows. */
function LiveSettings({ rows }: { rows: Row[] }) {
  const groups = new Map<string, Row[]>();
  for (const row of rows) groups.set(row.group, [...(groups.get(row.group) ?? []), row]);
  const draft = useLiveSettings<Value>({ settings: rows.map((row) => ({ name: row.name, value: row.value ?? row.default, defaultValue: row.default })) });
  return (
    <LiveSettingsForm
      draft={draft}
      readOnly
      label="Live settings"
      // Read-only: there's no save. Changes are made in Cayu Cloud.
      onSave={async () => ({ status: "error", message: "Settings are changed in Cayu Cloud.", retryable: false })}
    >
      {[...groups].map(([group, settings]) => (
        <SettingsSection key={group} title={group} headingLevel={2}>
          {settings.map((row) => (
            <LiveSettingRow
              key={row.name}
              label={row.label}
              help={[
                row.help,
                `Applies: ${row.applies}`,
                // Where the value comes from, and what applies without it.
                row.note,
                // A list doesn't fit its field: say it whole.
                row.kind === "text" && row.value !== null ? `In effect: ${show(row.value)}.` : null,
              ]
                .filter(Boolean)
                .join(" ")}
              htmlFor={row.value === null ? undefined : `setting-${row.name}`}
              changed={false}
              resetPending={false}
              defaultText={show(row.default)}
              onReset={() => {}}
              disabled
              error={row.error ?? undefined}
            >
              {/* Narrow on phones, so the label and help keep the room; a list is said whole in the help. */}
              <span className="flex w-32 flex-col items-end gap-1 sm:w-64 [&_input]:w-full">
                {row.value === null ? (
                  row.error ? (
                    // Refused: the Agent has no value for it, and its readers stop until it's fixed.
                    <span className="text-footnote text-danger-foreground">No value</span>
                  ) : (
                    // An unset limit: nothing limits it.
                    <span className="text-footnote text-text-secondary">Not set</span>
                  )
                ) : (
                  <LiveSettingControl
                    id={`setting-${row.name}`}
                    name={row.name}
                    label={row.label}
                    kind={row.kind}
                    choices={row.choices}
                    value={draft.value(row.name)}
                    minimum={row.minimum}
                    disabled
                    onChange={() => {}}
                  />
                )}
                <StatusBadge tone={row.error ? "danger" : row.source === "managed" ? "success" : "neutral"}>
                  {row.error ? "Refused" : (SOURCES[row.source] ?? row.source)}
                </StatusBadge>
              </span>
            </LiveSettingRow>
          ))}
        </SettingsSection>
      ))}
    </LiveSettingsForm>
  );
}

function FixedRow({ row }: { row: FixedSettingRow }) {
  return (
    <SettingsRow
      label={row.label}
      description={[row.warning ? `Warning: ${row.warning}` : null, row.help, row.why, row.env_var ? `Set by ${row.env_var}.` : null].filter(Boolean).join(" ")}
      control={
        <span className="flex flex-col items-end gap-1 text-end">
          <span className="font-mono text-footnote wrap-anywhere">{show(row.value)}</span>
          {row.warning ? <StatusBadge tone="warning">Check it</StatusBadge> : null}
        </span>
      }
    />
  );
}
