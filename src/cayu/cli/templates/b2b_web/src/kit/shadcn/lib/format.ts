/**
 * Formatting for app screens, in the reader's locale. A missing value renders as an
 * em dash, so an empty cell reads as "nothing yet" rather than as an error.
 */
const MISSING = "—";

type Instant = string | number | Date | null | undefined;

const dateTimeFormat = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
const relativeFormat = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

function toTime(value: Instant): number | null {
  if (value === null || value === undefined || value === "") return null;
  const time = value instanceof Date ? value.getTime() : new Date(value).getTime();
  return Number.isNaN(time) ? null : time;
}

/** "Nov 11, 2026, 8:20 PM". */
export function formatDateTime(value: Instant): string {
  const time = toTime(value);
  return time === null ? MISSING : dateTimeFormat.format(time);
}

/** "3 minutes ago", "in 2 hours", "yesterday"; under a minute is "just now". */
export function formatRelative(value: Instant, now: number = Date.now()): string {
  const time = toTime(value);
  if (time === null) return MISSING;
  const seconds = Math.round((time - now) / 1000);
  const steps: [Intl.RelativeTimeFormatUnit, number][] = [
    ["day", 86_400],
    ["hour", 3_600],
    ["minute", 60],
  ];
  for (const [unit, size] of steps) {
    if (Math.abs(seconds) >= size) return relativeFormat.format(Math.round(seconds / size), unit);
  }
  return "just now";
}

/** A duration in seconds: "42s", "3m 5s", "2h 10m". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return MISSING;
  const whole = Math.max(0, Math.round(seconds));
  if (whole < 60) return `${whole}s`;
  if (whole < 3_600) return `${Math.floor(whole / 60)}m ${whole % 60}s`;
  return `${Math.floor(whole / 3_600)}h ${Math.floor((whole % 3_600) / 60)}m`;
}

/** "1 rerun", "3 reruns"; give `many` for an irregular plural ("3 passes"). */
export function plural(count: number, word: string, many?: string): string {
  return `${count} ${count === 1 ? word : (many ?? `${word}s`)}`;
}
