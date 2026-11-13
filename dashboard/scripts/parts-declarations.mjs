/** Which of tsc's declarations the @cayu-tech/dashboard-parts package ships (see pack-parts.mjs). */
import path from "node:path"

/**
 * `file`'s path under `types` with "/" separators, so the same name comes out on Windows,
 * where `path.relative` separates with "\". `paths` is node:path or one of its flavours.
 */
export const declarationName = (types, file, paths = path) =>
  paths.relative(types, file).split(paths.sep).join("/")

/** Whether the package ships the declaration `declarationName` named `name`. */
export const shipsDeclaration = (name) =>
  name.startsWith("parts/") || name === "lib/release-metadata.d.ts"
