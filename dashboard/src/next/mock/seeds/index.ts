/**
 * The apps the prototype can switch between. Seeds from CAYU_PROTOTYPE_SEEDS (snapshots
 * of real deployments, kept outside the repository) follow the made-up default; see
 * PAGE-MAP.md.
 */
import extraSeeds from "@prototype-seeds"
import type { AppSeed } from "../model"
import { monthEnd } from "./month-end"
import { newApp } from "./new-app"

export const SEEDS: AppSeed[] = [monthEnd, ...(extraSeeds as AppSeed[]), newApp]
