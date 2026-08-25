import assert from "node:assert/strict"
import { spawnSync } from "node:child_process"
import {
  chmod,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  writeFile,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import test from "node:test"
import { fileURLToPath } from "node:url"

const runner = fileURLToPath(new URL("../scripts/run-python.mjs", import.meta.url))

test("Python launcher preserves executable and argument paths containing spaces", async () => {
  const temporaryDirectory = await mkdtemp(path.join(tmpdir(), "cayu python launcher "))
  try {
    const executable = path.join(
      temporaryDirectory,
      process.platform === "win32" ? "python executable.exe" : "python executable",
    )
    await copyFile(process.execPath, executable)
    if (process.platform !== "win32") {
      await chmod(executable, 0o755)
    }

    const fixture = path.join(temporaryDirectory, "record arguments.mjs")
    const output = path.join(temporaryDirectory, "recorded arguments.json")
    await writeFile(
      fixture,
      'import { writeFileSync } from "node:fs"\n' +
        "writeFileSync(process.argv[2], JSON.stringify(process.argv.slice(3)))\n",
    )

    const result = spawnSync(process.execPath, [runner, fixture, output, "argument with spaces"], {
      encoding: "utf8",
      env: { ...process.env, CAYU_PYTHON: executable },
    })

    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
    assert.deepEqual(JSON.parse(await readFile(output, "utf8")), ["argument with spaces"])
  } finally {
    await rm(temporaryDirectory, { force: true, recursive: true })
  }
})

test("repository mode selects the repository venv instead of ambient Python", async () => {
  const temporaryDirectory = await mkdtemp(path.join(tmpdir(), "cayu repository python "))
  try {
    const copiedRunner = path.join(temporaryDirectory, "dashboard", "scripts", "run-python.mjs")
    await mkdir(path.dirname(copiedRunner), { recursive: true })
    await copyFile(runner, copiedRunner)

    const repositoryPython = path.join(
      temporaryDirectory,
      ".venv",
      process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
    )
    const ambientDirectory = path.join(temporaryDirectory, "ambient")
    const ambientPython = path.join(
      ambientDirectory,
      process.platform === "win32" ? "python.exe" : "python",
    )
    await Promise.all([
      mkdir(path.dirname(repositoryPython), { recursive: true }),
      mkdir(ambientDirectory, { recursive: true }),
    ])
    await copyFile(process.execPath, repositoryPython)
    await copyFile(process.execPath, ambientPython)
    if (process.platform !== "win32") {
      await chmod(repositoryPython, 0o755)
      await chmod(ambientPython, 0o755)
    }

    const fixture = path.join(temporaryDirectory, "record-python.mjs")
    const output = path.join(temporaryDirectory, "selected-python.txt")
    await writeFile(
      fixture,
      'import { writeFileSync } from "node:fs"\nwriteFileSync(process.argv[2], process.execPath)\n',
    )
    const env = { ...process.env, PATH: `${ambientDirectory}${path.delimiter}${process.env.PATH}` }
    delete env.CAYU_PYTHON
    const result = spawnSync(process.execPath, [copiedRunner, "--repository", fixture, output], {
      encoding: "utf8",
      env,
    })

    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
    assert.equal(await realpath(await readFile(output, "utf8")), await realpath(repositoryPython))
  } finally {
    await rm(temporaryDirectory, { force: true, recursive: true })
  }
})
