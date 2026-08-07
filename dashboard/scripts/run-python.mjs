import { spawnSync } from "node:child_process";

const args = process.argv.slice(2);

if (args.length === 0) {
  console.error("usage: node scripts/run-python.mjs <script> [args...]");
  process.exit(2);
}

const python = process.env.CAYU_PYTHON || "python";
const result = spawnSync(python, args, {
  shell: false,
  stdio: "inherit",
});

if (result.error) {
  console.error(`could not run Python at ${JSON.stringify(python)}: ${result.error.message}`);
  process.exit(1);
}

if (result.signal) {
  console.error(`Python exited because of signal ${result.signal}`);
  process.exit(1);
}

process.exit(result.status ?? 1);
