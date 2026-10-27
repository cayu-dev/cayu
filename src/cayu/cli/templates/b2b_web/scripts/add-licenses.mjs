// Adds the licenses Vite's module graph can't see to the build's THIRD_PARTY_LICENSES.md:
// the packages only CSS imports, and the shadcn/ui components copied into src/components/ui.
import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.resolve(root, process.argv[2] ?? "static", "THIRD_PARTY_LICENSES.md");

async function packageLicense(name) {
  const directory = path.join(root, "node_modules", ...name.split("/"));
  const metadata = JSON.parse(await readFile(path.join(directory, "package.json"), "utf8"));
  const file = (await readdir(directory)).find((entry) => /^licen[cs]e(?:\.|$)/i.test(entry));
  if (file === undefined) throw new Error(`${name} has no license file`);
  const text = await readFile(path.join(directory, file), "utf8");
  return `## ${name} - ${metadata.version} (${metadata.license})\n\n${text.trim()}\n`;
}

const additions = [
  await packageLicense("tailwindcss"),
  await packageLicense("tw-animate-css"),
  `## shadcn/ui (MIT)\n\nSource: https://github.com/shadcn-ui/ui\n\n${(
    await readFile(path.join(root, "third_party", "shadcn-ui.LICENSE"), "utf8")
  ).trim()}\n`,
];
const existing = await readFile(output, "utf8");
await writeFile(output, `${existing.trimEnd()}\n\n${additions.join("\n")}`);
