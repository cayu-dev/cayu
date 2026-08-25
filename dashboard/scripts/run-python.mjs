import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rawArgs = process.argv.slice(2);
const repositoryMode = rawArgs[0] === "--repository";
const args = repositoryMode ? rawArgs.slice(1) : rawArgs;

if (args.length === 0) {
  console.error("usage: node scripts/run-python.mjs <script> [args...]");
  process.exit(2);
}

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const repositoryPython = path.join(
  repositoryRoot,
  ".venv",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const python = process.env.CAYU_PYTHON || (repositoryMode ? repositoryPython : "python");
if (repositoryMode && !process.env.CAYU_PYTHON && !existsSync(python)) {
  console.error(`repository Python environment is missing: ${JSON.stringify(python)}`);
  process.exit(1);
}
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
