import * as React from "react";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../components/ui/alert-dialog";
import { Spinner } from "../components/ui/spinner";
import { useAction, type ActionResult } from "./action";

export type AgentDecisionAction = {
  id: string;
  label: string;
  /** Why it can't be chosen now; shown under the button, which stays disabled. */
  disabledReason?: string;
  /** Ask before acting. */
  confirmation?: { title: string; description: string; destructive?: boolean };
};

export type AgentDecisionProps = {
  requestId: string;
  title: string;
  description?: string;
  /** The evidence: what the person decides on. */
  children?: React.ReactNode;
  status?: "pending" | "approved" | "denied" | "skipped" | "superseded";
  /** Replaces the settled status line: who decided it, and where. */
  outcome?: React.ReactNode;
  actions: AgentDecisionAction[];
  onAction: (actionId: string) => Promise<ActionResult>;
  /** The title's heading level. Default 2. */
  headingLevel?: 2 | 3 | 4;
};

/**
 * A decision an Agent asks a person for, as a stock Card: the title, the evidence,
 * and one button per action (the first is primary). An action with `confirmation`
 * asks first in an AlertDialog. A result nobody could confirm keeps the buttons
 * blocked until the person says they checked the outcome.
 */
export function AgentDecision(props: AgentDecisionProps) {
  // A new request starts over: no state carries from the last one.
  return <Decision key={props.requestId} {...props} />;
}

const SETTLED: Record<Exclude<NonNullable<AgentDecisionProps["status"]>, "pending">, string> = {
  approved: "Approved",
  denied: "Denied",
  skipped: "Skipped",
  superseded: "This request was replaced by a newer one.",
};

function Decision({ title, description, children, status = "pending", outcome, actions, onAction, headingLevel = 2 }: AgentDecisionProps) {
  const action = useAction();
  const [active, setActive] = React.useState<string>();
  const [confirming, setConfirming] = React.useState<AgentDecisionAction>();
  const titleRef = React.useRef<HTMLDivElement>(null);
  const done = status !== "pending" || action.result?.status === "success";

  async function run(choice: AgentDecisionAction) {
    if (done || choice.disabledReason) return;
    setActive(choice.id);
    const result = await action.run(() => onAction(choice.id));
    setConfirming(undefined);
    if (result?.status === "success") titleRef.current?.focus();
  }

  return (
    <Card data-slot="agent-decision">
      <CardHeader>
        <CardTitle ref={titleRef} tabIndex={-1} role="heading" aria-level={headingLevel} className="outline-none">
          {title}
        </CardTitle>
        {description ? <CardDescription>{description}</CardDescription> : null}
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {children}
        {done ? (
          <p role="status" className="text-sm text-muted-foreground">
            {status !== "pending" ? (outcome ?? SETTLED[status]) : action.result?.status === "success" ? (action.result.message ?? "Done") : null}
          </p>
        ) : (
          <>
            <div className="flex flex-wrap items-start gap-2">
              {actions.map((choice, index) => (
                <div key={choice.id} className="flex flex-col items-start gap-1">
                  <Button
                    variant={index === 0 ? "default" : "outline"}
                    disabled={action.busy || action.blocked || !!choice.disabledReason}
                    aria-busy={action.busy && active === choice.id}
                    onClick={() => (choice.confirmation ? setConfirming(choice) : void run(choice))}
                  >
                    {action.busy && active === choice.id ? <Spinner data-icon="inline-start" role={undefined} aria-label={undefined} /> : null}
                    {choice.label}
                  </Button>
                  {choice.disabledReason ? <p className="text-xs text-muted-foreground">{choice.disabledReason}</p> : null}
                </div>
              ))}
            </div>
            {action.blocked ? (
              <Button variant="outline" className="self-start" onClick={action.acknowledge}>
                I checked the outcome
              </Button>
            ) : null}
            <p role="status" className="text-sm text-muted-foreground empty:sr-only">
              {action.busy ? "Submitting…" : action.result?.status === "error" ? action.result.message : ""}
            </p>
          </>
        )}
      </CardContent>
      <AlertDialog
        open={confirming !== undefined}
        onOpenChange={(open) => {
          // A pending action holds the dialog open.
          if (!open && !action.busy) setConfirming(undefined);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{confirming?.confirmation?.title}</AlertDialogTitle>
            <AlertDialogDescription>{confirming?.confirmation?.description}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={action.busy}>Cancel</AlertDialogCancel>
            <Button
              variant={confirming?.confirmation?.destructive ? "destructive" : "default"}
              disabled={action.busy}
              onClick={() => confirming && void run(confirming)}
            >
              {action.busy ? <Spinner data-icon="inline-start" role={undefined} aria-label={undefined} /> : null}
              {confirming?.label}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}
