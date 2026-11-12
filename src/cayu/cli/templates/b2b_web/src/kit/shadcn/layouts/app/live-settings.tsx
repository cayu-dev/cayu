import * as React from "react";
import { FormProvider, useForm, useWatch, type Resolver } from "react-hook-form";
import { cn } from "../../lib/utils";
import { plural } from "../../lib/format";
import { Button } from "../../components/ui/button";
import { Checkbox } from "../../components/ui/checkbox";
import { Input } from "../../components/ui/input";
import { Label } from "../../components/ui/label";
import { NativeSelect, NativeSelectOption } from "../../components/ui/native-select";
import { Spinner } from "../../components/ui/spinner";
import { Switch } from "../../components/ui/switch";
import { useAction, type ActionResult } from "../../patterns/action";
import { SettingsRow } from "./settings";

/** A setting an operator changes while the service runs, with the default it goes back to. */
export type LiveSetting<V = unknown> = { name: string; value: V; defaultValue: V };

/** What Save sends: each changed setting's new value, or `null` to go back to its default. */
export type LiveSettingChanges<V = unknown> = Record<string, V | null>;

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/**
 * The draft behind a live-settings page, on React Hook Form: the values being
 * edited, Reset to the default, and the changes to send. Fields not being edited
 * take new values when the settings refresh; fields being edited keep the draft.
 */
export function useLiveSettings<V>({ settings, resolver }: { settings: readonly LiveSetting<V>[]; resolver?: Resolver<Record<string, V>> }) {
  const saved = React.useMemo(() => Object.fromEntries(settings.map((setting) => [setting.name, setting.value])) as Record<string, V>, [settings]);
  const form = useForm<Record<string, V>>({ values: saved, resetOptions: { keepDirtyValues: true }, resolver });
  const [resets, setResets] = React.useState<ReadonlySet<string>>(() => new Set());
  const watched = useWatch({ control: form.control }) as Record<string, V>;
  const values = { ...saved, ...watched };

  const changesOf = (current: Record<string, V>): LiveSettingChanges<V> => {
    const changes: LiveSettingChanges<V> = {};
    for (const setting of settings) {
      const value = setting.name in current ? current[setting.name] : setting.value;
      if (resets.has(setting.name) && same(value, setting.defaultValue)) changes[setting.name] = null;
      else if (!same(value, setting.value)) changes[setting.name] = value;
    }
    return changes;
  };
  const changes = changesOf(values);
  type FieldName = Parameters<typeof form.setValue>[0];

  return {
    form,
    value: (name: string) => values[name] as V,
    /** Whether Save puts this setting back to its default. */
    resetPending: (name: string) => name in changes && changes[name] === null,
    edit: (name: string, value: V) => {
      form.clearErrors(name as FieldName);
      form.setValue(name as FieldName, value as never, { shouldDirty: true });
    },
    reset: (name: string) => {
      const setting = settings.find((candidate) => candidate.name === name);
      if (!setting) return;
      setResets((current) => new Set(current).add(name));
      form.clearErrors(name as FieldName);
      form.setValue(name as FieldName, setting.defaultValue as never, { shouldDirty: true });
    },
    discard: () => {
      setResets(new Set());
      form.reset(saved, { keepDirtyValues: false });
    },
    changes,
    changeCount: Object.keys(changes).length,
    changesToSave: (current: Record<string, V>) => changesOf(current),
    /** After a save: the saved values become the draft's starting point. */
    settle: () => {
      setResets(new Set());
      form.reset(form.getValues(), { keepDirtyValues: false });
    },
  };
}

export type LiveSettingsDraft<V = unknown> = ReturnType<typeof useLiveSettings<V>>;

/**
 * The form around a live-settings page's rows, with a save bar pinned to the bottom:
 * what's unsaved, Discard and Save. `readOnly` disables nothing by itself (the rows
 * do) and drops the bar. Children may be a function of `{ editable }`.
 */
export function LiveSettingsForm<V>({
  draft,
  onSave,
  label = "Settings",
  readOnly = false,
  className,
  children,
}: {
  draft: LiveSettingsDraft<V>;
  onSave: (changes: LiveSettingChanges<V>) => Promise<ActionResult>;
  label?: string;
  readOnly?: boolean;
  className?: string;
  children: React.ReactNode | ((state: { editable: boolean }) => React.ReactNode);
}) {
  const { form } = draft;
  const action = useAction();
  const count = draft.changeCount;
  const failed = action.result?.status === "error" ? action.result : null;
  const status = failed ? failed.message : count ? plural(count, "unsaved change") : action.result?.status === "success" && action.result.message ? action.result.message : "No unsaved changes";
  const submit = form.handleSubmit(async (values) => {
    const result = await action.run(() => onSave(draft.changesToSave(values)));
    if (result?.status === "success") draft.settle();
    if (result?.status === "error" && result.fieldErrors) {
      for (const [name, message] of Object.entries(result.fieldErrors)) form.setError(name as never, { message });
    }
  });
  return (
    <FormProvider {...form}>
      <form data-slot="live-settings-form" aria-label={label} noValidate className={cn("contents", className)} onSubmit={(event) => void submit(event)}>
        {typeof children === "function" ? children({ editable: !readOnly && !action.busy && !action.blocked }) : children}
        {readOnly ? null : (
          <div data-slot="live-settings-bar" className="sticky bottom-0 z-10 flex flex-wrap items-center justify-end gap-2 border-t bg-background py-3">
            <span role="status" className={cn("me-auto text-sm", failed ? "text-destructive" : "text-muted-foreground")}>
              {status}
            </span>
            {action.blocked ? (
              <Button variant="outline" onClick={action.acknowledge}>
                I checked the outcome
              </Button>
            ) : null}
            <Button variant="ghost" disabled={count === 0 || action.busy} onClick={draft.discard}>
              Discard
            </Button>
            <Button type="submit" disabled={count === 0 || action.busy || action.blocked}>
              {action.busy ? <Spinner data-icon="inline-start" role={undefined} aria-label={undefined} /> : null}
              {action.busy ? "Saving…" : "Save changes"}
            </Button>
          </div>
        )}
      </form>
    </FormProvider>
  );
}

const DEFAULT_NOTES = {
  resetPending: (value: string) => `Goes back to the default (${value}) when saved.`,
  changed: (value: string) => `Changed here; the default is ${value}.`,
};

/** One live setting as a SettingsRow: label, help, a note when it differs from its default, Reset, the control. */
export function LiveSettingRow({
  label,
  help,
  htmlFor,
  changed,
  resetPending,
  defaultText,
  onReset,
  disabled = false,
  error,
  notes = DEFAULT_NOTES,
  children,
}: {
  label: string;
  help?: string;
  htmlFor?: string;
  /** The saved value differs from the default. */
  changed: boolean;
  resetPending: boolean;
  /** The default, as text for the note. */
  defaultText: string;
  onReset: () => void;
  disabled?: boolean;
  error?: string;
  notes?: { resetPending: (defaultText: string) => string; changed: (defaultText: string) => string };
  children: React.ReactNode;
}) {
  const note = resetPending ? notes.resetPending(defaultText) : changed ? notes.changed(defaultText) : "";
  return (
    <SettingsRow
      label={label}
      description={
        help || note || error ? (
          <>
            {[help, note].filter(Boolean).join(" ")}
            {error ? <span className="block text-destructive">{error}</span> : null}
          </>
        ) : undefined
      }
      htmlFor={htmlFor}
      control={
        <>
          {changed && !resetPending && !disabled ? (
            <Button variant="ghost" size="sm" onClick={onReset}>
              Reset
            </Button>
          ) : null}
          {children}
        </>
      }
    />
  );
}

/** `bool` a switch, `int`/`number` a number field, `choice` a select, `multi` a checkbox per choice, `text` a text field. */
export type LiveSettingKind = "bool" | "int" | "number" | "choice" | "multi" | "text";

export function LiveSettingControl({
  id,
  name,
  label,
  kind,
  value,
  onChange,
  choices = [],
  choiceLabel = (choice) => choice,
  minimum,
  blankLabel,
  placeholder,
  disabled = false,
}: {
  id: string;
  name: string;
  /** Names the checkbox group of a `multi` setting. */
  label: string;
  kind: LiveSettingKind;
  value: unknown;
  onChange: (value: boolean | number | string | string[]) => void;
  choices?: readonly string[];
  choiceLabel?: (choice: string) => string;
  minimum?: number | null;
  blankLabel?: string;
  placeholder?: string;
  disabled?: boolean;
}) {
  if (kind === "bool") return <Switch id={id} name={name} checked={Boolean(value)} disabled={disabled} onCheckedChange={(checked) => onChange(checked)} />;
  if (kind === "multi") {
    const selected = Array.isArray(value) ? (value as string[]) : [];
    return (
      <div role="group" aria-label={label} className="flex flex-wrap gap-4">
        {choices.map((choice) => (
          <div key={choice} className="flex items-center gap-2">
            <Checkbox
              id={`${id}-${choice}`}
              name={name}
              checked={selected.includes(choice)}
              disabled={disabled}
              onCheckedChange={(checked) =>
                onChange(checked ? choices.filter((other) => other === choice || selected.includes(other)) : selected.filter((other) => other !== choice))
              }
            />
            <Label htmlFor={`${id}-${choice}`}>{choiceLabel(choice)}</Label>
          </div>
        ))}
      </div>
    );
  }
  if (kind === "choice") {
    return (
      <NativeSelect id={id} name={name} className="w-44" value={String(value)} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {blankLabel !== undefined ? <NativeSelectOption value="">{blankLabel}</NativeSelectOption> : null}
        {choices.map((choice) => (
          <NativeSelectOption key={choice} value={choice}>
            {choiceLabel(choice)}
          </NativeSelectOption>
        ))}
      </NativeSelect>
    );
  }
  if (kind === "text") {
    return <Input id={id} name={name} className="w-56" value={String(value ?? "")} disabled={disabled} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />;
  }
  return (
    <Input
      id={id}
      name={name}
      type="number"
      className="w-28"
      inputMode={kind === "int" ? "numeric" : "decimal"}
      min={minimum ?? undefined}
      step={kind === "int" ? 1 : "any"}
      value={String(value ?? "")}
      disabled={disabled}
      onChange={(event) => {
        // A half-typed number stays text until it parses.
        const number = event.target.valueAsNumber;
        onChange(Number.isNaN(number) ? event.target.value : number);
      }}
    />
  );
}
