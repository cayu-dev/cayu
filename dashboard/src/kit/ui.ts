/**
 * The component kit the parts in ../parts render through, as `@kit/ui`.
 *
 * Here it is the dashboard's own stock shadcn components. An app that imports the parts
 * points `@kit` at its own kit instead (stock shadcn, or @cayu-tech/design), so a part
 * looks like the rest of that app. Export only what parts use, under the names every kit
 * shares.
 */
export { Button } from "@/components/ui/button"
export { Card, CardContent } from "@/components/ui/card"
export { Input } from "@/components/ui/input"
