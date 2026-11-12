import * as React from "react";
import { cn } from "../lib/utils";
import {
  DropdownMenuGroup as StockDropdownMenuGroup,
  DropdownMenuLabel as StockDropdownMenuLabel,
  DropdownMenuRadioGroup as StockDropdownMenuRadioGroup,
} from "./ui/dropdown-menu";

export {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuPortal,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";

/*
 * Base UI's group label must sit inside a group. These wrappers let a
 * DropdownMenuLabel also head a menu on its own ("Signed in as admin"), where it
 * renders as plain text with the stock label's look.
 */
const InGroup = React.createContext(false);

export function DropdownMenuGroup(props: React.ComponentProps<typeof StockDropdownMenuGroup>) {
  return (
    <InGroup.Provider value>
      <StockDropdownMenuGroup {...props} />
    </InGroup.Provider>
  );
}

export function DropdownMenuRadioGroup(props: React.ComponentProps<typeof StockDropdownMenuRadioGroup>) {
  return (
    <InGroup.Provider value>
      <StockDropdownMenuRadioGroup {...props} />
    </InGroup.Provider>
  );
}

export function DropdownMenuLabel(props: React.ComponentProps<typeof StockDropdownMenuLabel>) {
  const inGroup = React.useContext(InGroup);
  if (inGroup) return <StockDropdownMenuLabel {...props} />;
  const { className, inset, render: _render, style, ...rest } = props;
  return (
    <div
      data-slot="dropdown-menu-label"
      data-inset={inset}
      className={cn("px-1.5 py-1 text-xs font-medium text-muted-foreground data-inset:pl-7", typeof className === "string" ? className : undefined)}
      style={typeof style === "object" ? style : undefined}
      {...(rest as React.ComponentProps<"div">)}
    />
  );
}
