// Builds the app into a temporary folder and compares it with the committed ./static, which
// `cayu serve` serves: they must be the same files with the same bytes. Run after `npm ci`;
// `npm run build` rewrites ./static when they differ.
import { spawnSync } from "node:child_process";
import { mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const committed = path.join(root, "static");
const fresh = await mkdtemp(path.join(tmpdir(), "web-static-"));

// Runs a Node script with this Node, not through an npm command shim, which Node can't start
// without a shell on Windows. A failure throws, so the temporary folder is still removed.
function node(args, env = {}) {
  const result = spawnSync(process.execPath, args, {
    cwd: root,
    stdio: "inherit",
    env: { ...process.env, ...env },
  });
  if (result.status !== 0) throw new Error(`node ${args.join(" ")} failed`);
}

async function files(directory, prefix = "") {
  const found = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relative = path.join(prefix, entry.name);
    if (entry.isDirectory())
      found.push(...(await files(path.join(directory, entry.name), relative)));
    else found.push(relative);
  }
  return found.sort();
}

try {
  node(["node_modules/vite/bin/vite.js", "build", "--logLevel", "warn"], { WEB_OUT_DIR: fresh });
  node(["scripts/add-licenses.mjs", fresh]);
  const [want, have] = [await files(fresh), await files(committed)];
  const differences = [
    ...want.filter((name) => !have.includes(name)).map((name) => `missing from static: ${name}`),
    ...have.filter((name) => !want.includes(name)).map((name) => `not in a fresh build: ${name}`),
  ];
  for (const name of want.filter((name) => have.includes(name))) {
    const [a, b] = await Promise.all([
      readFile(path.join(fresh, name)),
      readFile(path.join(committed, name)),
    ]);
    if (!a.equals(b)) differences.push(`differs: ${name}`);
  }
  if (differences.length) {
    console.error(
      `web/static doesn't match web/src. Run \`npm run build\` and commit web/static.\n${differences.join("\n")}`,
    );
    process.exitCode = 1;
  } else {
    console.log(`web/static matches a fresh build (${want.length} files).`);
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
} finally {
  await rm(fresh, { recursive: true, force: true });
}
