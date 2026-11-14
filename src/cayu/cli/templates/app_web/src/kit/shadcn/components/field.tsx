import * as React from "react";
import { Field as StockField, FieldLabel as StockFieldLabel } from "./ui/field";
import { Input as StockInput } from "./ui/input";
import { NativeSelect as StockNativeSelect } from "./ui/native-select";
import { Textarea as StockTextarea } from "./ui/textarea";

export { FieldContent, FieldDescription, FieldError, FieldGroup, FieldLegend, FieldSeparator, FieldSet, FieldTitle } from "./ui/field";
export { NativeSelectOptGroup, NativeSelectOption } from "./ui/native-select";

/*
 * The stock Field, FieldLabel, Input, Textarea and NativeSelect, joined by an id: a FieldLabel
 * inside a Field labels the Field's control without the caller wiring `htmlFor`.
 * An explicit `id` or `htmlFor` wins.
 */
const FieldControlId = React.createContext<string | undefined>(undefined);

export function Field(props: React.ComponentProps<typeof StockField>) {
  const id = React.useId();
  return (
    <FieldControlId.Provider value={id}>
      <StockField {...props} />
    </FieldControlId.Provider>
  );
}

export function FieldLabel({ htmlFor, ...props }: React.ComponentProps<typeof StockFieldLabel>) {
  const id = React.useContext(FieldControlId);
  return <StockFieldLabel htmlFor={htmlFor ?? id} {...props} />;
}

export function Input({ id, ...props }: React.ComponentProps<typeof StockInput>) {
  const fieldId = React.useContext(FieldControlId);
  return <StockInput id={id ?? fieldId} {...props} />;
}

export function NativeSelect({ id, ...props }: React.ComponentProps<typeof StockNativeSelect>) {
  const fieldId = React.useContext(FieldControlId);
  return <StockNativeSelect id={id ?? fieldId} {...props} />;
}

export function Textarea({ id, ...props }: React.ComponentProps<typeof StockTextarea>) {
  const fieldId = React.useContext(FieldControlId);
  return <StockTextarea id={id ?? fieldId} {...props} />;
}
