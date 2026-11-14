import * as React from "react";

/**
 * What an action reports. `retryable: false` means nobody can tell whether it was
 * applied, so the control stays blocked until the person says they checked.
 */
export type ActionResult =
  | { status: "success"; message?: string }
  | { status: "error"; message: string; retryable: boolean; fieldErrors?: Record<string, string> };

export type ActionState = {
  busy: boolean;
  /** An outcome nobody could confirm: no new attempt until `acknowledge()`. */
  blocked: boolean;
  result: ActionResult | null;
  run: (action: () => Promise<ActionResult>) => Promise<ActionResult | null>;
  acknowledge: () => void;
};

/** One async action at a time; a thrown error counts as an unknown outcome. */
export function useAction(): ActionState {
  const [busy, setBusy] = React.useState(false);
  const [result, setResult] = React.useState<ActionResult | null>(null);
  const pending = React.useRef(false);
  const blocked = result?.status === "error" && !result.retryable;
  const run = React.useCallback(
    async (action: () => Promise<ActionResult>) => {
      if (pending.current || blocked) return null;
      pending.current = true;
      setBusy(true);
      setResult(null);
      let outcome: ActionResult;
      try {
        outcome = await action();
      } catch {
        outcome = { status: "error", retryable: false, message: "The outcome couldn't be confirmed. Check the current state before trying again." };
      }
      pending.current = false;
      setBusy(false);
      setResult(outcome);
      return outcome;
    },
    [blocked],
  );
  const acknowledge = React.useCallback(() => setResult(null), []);
  return { busy, blocked, result, run, acknowledge };
}
