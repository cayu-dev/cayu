import type * as React from "react";
import { toast as sonner } from "sonner";
import { Toaster as StockToaster } from "./ui/sonner";

export type ToastType = "success" | "error" | "info" | "warning" | "loading";

export type ToastOptions = {
  title?: React.ReactNode;
  description?: React.ReactNode;
  type?: ToastType;
  /** Milliseconds before it closes; 0 keeps it open. */
  timeout?: number;
  id?: string;
};

const show = { success: sonner.success, error: sonner.error, info: sonner.info, warning: sonner.warning, loading: sonner.loading };

/**
 * The toast manager over the stock shadcn Sonner toaster:
 * `toast.add({ title, description, type })` from anywhere, `toast.close(id)` to dismiss.
 */
export const toast = {
  add({ title, description, type, timeout, id }: ToastOptions): string {
    const open = type ? show[type] : sonner;
    return String(open(title, { description, id, duration: timeout === 0 ? Infinity : timeout }));
  },
  close(id?: string) {
    sonner.dismiss(id);
  },
};

/** Render once around the app: the app, then the stock Sonner toaster. */
export function Toaster({ children, timeout = 4000, ...props }: React.ComponentProps<typeof StockToaster> & { children?: React.ReactNode; timeout?: number }) {
  return (
    <>
      {children}
      <StockToaster duration={timeout} {...props} />
    </>
  );
}
