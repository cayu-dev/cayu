/**
 * The icons the parts in ../parts render, as `@kit/icons` (see ./ui.ts).
 *
 * The kits apps use export the HugeIcons set and set `Icon` to `HugeiconsIcon`. The
 * dashboard draws with lucide, so here each HugeIcons name a part uses is the lucide icon
 * the dashboard already shows, and `Icon` renders it. Parts name icons only through this
 * module.
 */
import {
  CircleHelp,
  Lightbulb,
  type LucideIcon,
  type LucideProps,
  RotateCcw,
  ShieldCheck,
} from "lucide-react"
import { createElement } from "react"

/** An icon a part can pass to `Icon`. */
export type KitIcon = LucideIcon

export const BulbIcon: KitIcon = Lightbulb
export const HelpCircleIcon: KitIcon = CircleHelp
export const RotateLeft01Icon: KitIcon = RotateCcw
export const SecurityCheckIcon: KitIcon = ShieldCheck

export type IconProps = {
  icon: KitIcon
  className?: string
  strokeWidth?: number
  "aria-label"?: string
  "aria-hidden"?: boolean | "true" | "false"
}

export function Icon({ icon, ...props }: IconProps) {
  return createElement(icon, props satisfies LucideProps)
}
