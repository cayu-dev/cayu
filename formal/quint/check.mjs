// Model-check the Runtime protocol models with Quint's TLC backend.
//
//   node check.mjs              run every check
//   node check.mjs cexHandoffReleasesSameNameReplacement ...  run only the named checks
//
// Each check is an invariant of one model instance with an expected result:
// "holds" (no reachable state violates it) or "violated" (a known
// counterexample, or a reachability witness). The run fails when any result
// differs from its expectation.
//
// The checks run only the models; they never import or execute src/cayu. A
// Runtime fix therefore leaves every known counterexample "violated" and CI
// green until someone updates the model to match the fix. Then the check
// reports "holds", this run fails, and its expectation moves to "holds".
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const quint = path.join(here, "node_modules", ".bin", "quint");

// TLC's heap and workers come from tlc.json (1 GB, 2 workers). The Apalache
// server that compiles Quint to TLA+ takes its JVM options from JVM_ARGS and
// otherwise defaults to a 4 GB heap; give it the same 1 GB cap unless the
// caller sets one.
const env = { ...process.env, JVM_ARGS: process.env.JVM_ARGS ?? "-Xmx1g" };

// No check takes more than about a minute; a hung JVM fails its check alone
// instead of the whole job's time bound.
const CHECK_TIMEOUT_MS = 180_000;

const SESSION = "session_settlement.qnt";
const LEASE = "task_lease.qnt";

/** [name, file, instance, invariant, expected] */
const CHECKS = [
  // Session settlement: the single settlement path holds.
  ["designScanSettles", SESSION, "session_settlement_design_scan", "designScanSettles", "holds"],
  ["designSettlesWithoutCrashAfterDecision", SESSION, "session_settlement_design", "designSettlesWithoutCrashAfterDecision", "holds"],
  // Session settlement: known counterexamples on main.
  ["cexResumeSetupGroupCancel", SESSION, "session_settlement_current", "cexResumeSetupGroupCancel", "violated"],
  ["cexResumeAdmissionCancel", SESSION, "session_settlement_current", "cexResumeAdmissionCancel", "violated"],
  ["cexRunEntranceCreate", SESSION, "session_settlement_current", "cexRunEntranceCreate", "violated"],
  ["cexResumePromotedClose", SESSION, "session_settlement_current", "cexResumePromotedClose", "violated"],
  ["cexCompletedInteractionClose", SESSION, "session_settlement_current", "cexCompletedInteractionClose", "violated"],
  ["cexTaskCompletionGroup", SESSION, "session_settlement_current", "cexTaskCompletionGroup", "violated"],
  ["cexGroupCancelHoldsTask", SESSION, "session_settlement_current", "cexGroupCancelHoldsTask", "violated"],
  ["cexTaskCrashTerminalize", SESSION, "session_settlement_current", "cexTaskCrashTerminalize", "violated"],
  ["cexTrailingFenceCancel", SESSION, "session_settlement_current", "cexTrailingFenceCancel", "violated"],
  ["cexTrailingFenceCrash", SESSION, "session_settlement_current", "cexTrailingFenceCrash", "violated"],
  ["cexOrphanedRunningNoSweep", SESSION, "session_settlement_current", "cexOrphanedRunningNoSweep", "violated"],
  ["cexOrphanedRunningTaskFailed", SESSION, "session_settlement_current", "cexOrphanedRunningTaskFailed", "violated"],
  // Session settlement: main violates every property.
  ["currentNoOrphanedRunning", SESSION, "session_settlement_current", "currentNoOrphanedRunning", "violated"],
  ["currentRunExitSettles", SESSION, "session_settlement_current", "currentRunExitSettles", "violated"],
  ["currentInteractionSettles", SESSION, "session_settlement_current", "currentInteractionSettles", "violated"],
  ["currentTaskMatchesSession", SESSION, "session_settlement_current", "currentTaskMatchesSession", "violated"],
  ["currentWorkerReleasesTask", SESSION, "session_settlement_current", "currentWorkerReleasesTask", "violated"],
  ["currentNoPermanentFence", SESSION, "session_settlement_current", "currentNoPermanentFence", "violated"],
  // Session settlement: the design without the decision scan.
  ["cexDesignNeedsScan", SESSION, "session_settlement_design", "cexDesignNeedsScan", "violated"],
  // Session settlement: reachability.
  ["reachCompletesCurrent", SESSION, "session_settlement_current", "reachCompletes", "violated"],
  ["reachCompletesDesign", SESSION, "session_settlement_design_scan", "reachCompletes", "violated"],
  ["reachRecoveredDecision", SESSION, "session_settlement_design_scan", "reachRecoveredDecision", "violated"],
  // Task lease: generation fencing holds.
  ["generationHbLastSafe", LEASE, "task_lease_generation_hb_last", "generationHbLastSafe", "holds"],
  ["generationFences", LEASE, "task_lease_generation", "generationFences", "holds"],
  // Task lease: known counterexamples on main.
  ["cexHandoffReleasesSameNameReplacement", LEASE, "task_lease_token", "cexHandoffReleasesSameNameReplacement", "violated"],
  ["cexRetrySettlementLosesToHeartbeats", LEASE, "task_lease_token", "cexRetrySettlementLosesToHeartbeats", "violated"],
  ["cexHeartbeatStoppedBeforeDisposition", LEASE, "task_lease_token", "cexHeartbeatStoppedBeforeDisposition", "violated"],
  // Task lease: main violates every property.
  ["tokenNoStaleOwnerWrite", LEASE, "task_lease_token", "tokenNoStaleOwnerWrite", "violated"],
  ["tokenNoFalseClaimLost", LEASE, "task_lease_token", "tokenNoFalseClaimLost", "violated"],
  ["tokenReportRecorded", LEASE, "task_lease_token", "tokenReportRecorded", "violated"],
  ["tokenLiveHolderKeepsClaim", LEASE, "task_lease_token", "tokenLiveHolderKeepsClaim", "violated"],
  // Task lease: generation fencing alone does not cover cexHeartbeatStoppedBeforeDisposition.
  ["cexGenerationNeedsHbLast", LEASE, "task_lease_generation", "cexGenerationNeedsHbLast", "violated"],
  // Task lease: reachability.
  ["reachTakeoverCompletes", LEASE, "task_lease_generation_hb_last", "reachTakeoverCompletes", "violated"],
  ["reachHandoff", LEASE, "task_lease_generation_hb_last", "reachHandoff", "violated"],
];

// The check that is running now. Its process group is outside the terminal's
// foreground group, so Ctrl-C would not reach it: stop it before exiting.
let current = null;
for (const [signal, code] of [["SIGINT", 130], ["SIGTERM", 143]]) {
  process.on(signal, () => {
    if (current !== null) {
      try {
        process.kill(-current.pid, "SIGKILL");
      } catch {
        // The group has already exited.
      }
    }
    process.exit(code);
  });
}

function run([name, file, main, invariant, expected]) {
  const started = Date.now();
  return new Promise((resolve) => {
    // Its own process group, so a timeout also stops the JVMs Quint starts.
    const child = spawn(
      quint,
      ["verify", "--backend=tlc", "--tlc-config=tlc.json", `--main=${main}`, `--invariant=${invariant}`, file],
      { cwd: here, env, detached: true },
    );
    current = child;
    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk;
    });
    child.stderr.on("data", (chunk) => {
      output += chunk;
    });
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      process.kill(-child.pid, "SIGKILL");
    }, CHECK_TIMEOUT_MS);
    child.on("close", () => {
      clearTimeout(timer);
      current = null;
      const seconds = ((Date.now() - started) / 1000).toFixed(1);
      if (timedOut) {
        console.log(`FAIL ${name}: no result within ${CHECK_TIMEOUT_MS / 1000}s; the check was stopped`);
        resolve(false);
        return;
      }
      let actual = "error";
      if (/^\[ok\] No violation found/m.test(output)) actual = "holds";
      else if (/^\[violation\]/m.test(output)) actual = "violated";
      const ok = actual === expected;
      console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${actual} (expected ${expected}, ${seconds}s)`);
      if (!ok) console.log(output);
      resolve(ok);
    });
  });
}

const selected = process.argv.slice(2);
const unknown = selected.filter((name) => !CHECKS.some(([check]) => check === name));
if (unknown.length > 0) {
  console.error(`Unknown checks: ${unknown.join(", ")}`);
  process.exit(2);
}
const checks = selected.length === 0 ? CHECKS : CHECKS.filter(([name]) => selected.includes(name));
// One at a time: each check starts its own JVMs, and Quint's compiler
// server listens on a fixed port.
let failures = 0;
for (const check of checks) {
  if (!(await run(check))) failures += 1;
}
console.log(`${checks.length - failures}/${checks.length} checks as expected`);
process.exit(failures === 0 ? 0 : 1);
