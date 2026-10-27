/** The retained prototype has independent search schemas and a hash-history router.
 * Keep its deliberately permissive navigation outside the production wrappers. */
import {
  type AnyRouter,
  Link,
  type LinkComponentProps,
  type NavigateOptions,
  useRouter,
} from "@tanstack/react-router"

export function PrototypeLink(props: LinkComponentProps<"a", AnyRouter, string, string>) {
  return <Link<AnyRouter, string, string, string, "."> {...props} />
}
export function usePrototypeGo() {
  const router = useRouter<AnyRouter>()
  return (options: NavigateOptions<AnyRouter, string, string>) =>
    router.navigate<AnyRouter, string, string, string, ".">(options)
}
