import * as React from "react";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "../../lib/utils";

/**
 * A group of settings rows under a title. The title is an h3 by default; set
 * `headingLevel` to fit the page's outline (2 directly under the page's h1).
 */
export function SettingsSection({
  className,
  title,
  headingLevel = 3,
  children,
  ...props
}: Omit<React.ComponentProps<"section">, "title"> & { title?: React.ReactNode; headingLevel?: 2 | 3 | 4 | 5 | 6 }) {
  const Heading = `h${headingLevel}` as const;
  return (
    <section data-slot="settings-section" className={cn("flex flex-col", className)} {...props}>
      {title ? <Heading className="border-b pt-4 pb-3 text-base font-semibold">{title}</Heading> : null}
      <div className="flex flex-col divide-y">{children}</div>
    </section>
  );
}

type SettingsRowBase = {
  label: React.ReactNode;
  description?: React.ReactNode;
  control?: React.ReactNode;
  /** Ties the label to a control by id. */
  htmlFor?: string;
};

export type SettingsRowProps =
  | (SettingsRowBase & Omit<React.ComponentProps<"div">, "onClick"> & { onClick?: undefined })
  | (SettingsRowBase & Omit<React.ComponentProps<"button">, "onClick"> & { onClick: React.MouseEventHandler<HTMLButtonElement> });

/**
 * One setting: its label and description on the left, the control on the right.
 * With `onClick` the row is one button (the control sits beside it, never inside).
 */
export function SettingsRow({ className, label, description, control, htmlFor, onClick, ...props }: SettingsRowProps) {
  const controlId = React.useId();
  const text = (Tag: "label" | "div" | "span", tagProps: { htmlFor?: string } = {}) => (
    <Tag {...tagProps} className="flex min-w-0 flex-1 flex-col gap-1 py-4">
      <span className="text-sm font-medium">{label}</span>
      {description ? <span className="text-sm text-muted-foreground">{description}</span> : null}
    </Tag>
  );
  const controlSlot = control ? (
    <div id={onClick ? controlId : undefined} className="relative z-10 flex shrink-0 items-center gap-2 text-sm">
      {control}
    </div>
  ) : null;
  if (onClick) {
    const buttonProps = props as React.ComponentProps<"button">;
    return (
      <div data-slot="settings-row" className={cn("relative -mx-2 flex min-h-14 items-center gap-4 rounded-md px-2 hover:bg-muted/50 has-[>button:focus-visible]:ring-3 has-[>button:focus-visible]:ring-ring/50", className)}>
        <button
          type="button"
          onClick={onClick}
          aria-describedby={buttonProps["aria-describedby"] ?? (control ? controlId : undefined)}
          className="flex min-w-0 flex-1 text-left outline-none after:absolute after:inset-0 after:rounded-md"
          {...buttonProps}
        >
          {text("span")}
        </button>
        {controlSlot}
        <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} className="size-4 shrink-0 text-muted-foreground" />
      </div>
    );
  }
  return (
    <div data-slot="settings-row" className={cn("flex min-h-14 items-center gap-4", className)} {...(props as React.ComponentProps<"div">)}>
      {htmlFor ? text("label", { htmlFor }) : text("div")}
      {controlSlot}
    </div>
  );
}
