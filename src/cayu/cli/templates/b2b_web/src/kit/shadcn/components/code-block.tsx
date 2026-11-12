import * as React from "react";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "../lib/utils";
import { Button } from "./ui/button";

async function writeClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // The async clipboard can be refused (an insecure page, a sandboxed frame); fall back to a selection copy.
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    const copied = document.execCommand("copy");
    area.remove();
    return copied;
  }
}

/**
 * Read-only preformatted text with a Copy button: a CLI snippet, a config file.
 * Lines scroll sideways; `wrap` wraps them. The header shows `label`, else `language`.
 */
export function CodeBlock({
  children,
  language,
  label,
  wrap = false,
  onCopied,
  className,
  ...props
}: Omit<React.ComponentProps<"div">, "children"> & {
  /** The text, exactly as it should be copied. */
  children: string;
  language?: string;
  label?: string;
  wrap?: boolean;
  onCopied?: () => void;
}) {
  const [status, setStatus] = React.useState<"" | "Copied" | "Couldn't copy">("");
  React.useEffect(() => {
    if (!status) return;
    const timer = setTimeout(() => setStatus(""), 2000);
    return () => clearTimeout(timer);
  }, [status]);
  const name = label ?? (language ? `${language} code` : "Code");
  return (
    <div data-slot="code-block" className={cn("flex min-w-0 flex-col overflow-hidden rounded-lg border bg-muted/40", className)} {...props}>
      <div className="flex items-center justify-between gap-3 border-b px-3 py-1.5">
        <span className={cn("min-w-0 truncate text-xs text-muted-foreground", !label && "font-mono")}>{label ?? language ?? "Code"}</span>
        <Button
          variant="ghost"
          size="xs"
          aria-label={label ? `Copy ${label}` : "Copy code"}
          onClick={async () => {
            const copied = await writeClipboard(children);
            setStatus(copied ? "Copied" : "Couldn't copy");
            if (copied) onCopied?.();
          }}
        >
          <HugeiconsIcon icon={status === "Copied" ? Tick02Icon : Copy01Icon} strokeWidth={2} data-icon="inline-start" />
          {status === "Copied" ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre
        role="region"
        tabIndex={0}
        aria-label={name}
        className={cn("m-0 overflow-auto px-3 py-3 font-mono text-xs leading-relaxed outline-none focus-visible:ring-3 focus-visible:ring-ring/50", wrap ? "whitespace-pre-wrap wrap-anywhere" : "whitespace-pre")}
      >
        <code>{children}</code>
      </pre>
      <span role="status" aria-live="polite" className={status === "Couldn't copy" ? "px-3 pb-2 text-xs text-muted-foreground" : "sr-only"}>
        {status}
      </span>
    </div>
  );
}
