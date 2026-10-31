import type { EvalSuiteTrialPolicyV1 } from "./generated/server-api/index.ts"

const TRIAL_POLICY_SCHEMA_VERSION = 1

/**
 * The server's `EvalSuiteTrialPolicyV1.create(trial_count=trialCount,
 * max_concurrency=maxConcurrency)`: every one of a case's ``trialCount`` trials must pass, and
 * at most ``maxConcurrency`` run at a time.
 *
 * Its revision is the SHA-256 of the policy's other fields as canonical JSON, sorted keys
 * without whitespace (`canonical_durable_json_bytes`); for these integers and booleans that is
 * exactly what `JSON.stringify` writes. The digest is Web Crypto's, which a secure context
 * (HTTPS or loopback) provides.
 */
export async function evalSuiteTrialPolicy(
  trialCount: number,
  maxConcurrency: number,
): Promise<EvalSuiteTrialPolicyV1> {
  const subtle = globalThis.crypto?.subtle
  if (subtle === undefined) {
    throw new Error("Secure browser hashing is unavailable, so the trial policy can't be made.")
  }
  const material = {
    schema_version: TRIAL_POLICY_SCHEMA_VERSION,
    trial_count: trialCount,
    minimum_passed_trials: trialCount,
    max_concurrency: maxConcurrency,
    require_zero_runtime_errors: true,
    require_zero_evaluator_errors: true,
    require_complete_required_evidence: true,
  } as const
  // A key list writes those keys in its order; the keys are ASCII, so code-unit order is
  // Python's sort_keys order.
  const canonical = JSON.stringify(material, Object.keys(material).sort())
  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(canonical))
  const hex = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0"))
  return { ...material, revision: `sha256:${hex.join("")}` }
}
