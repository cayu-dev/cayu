# Benchmark packages

A benchmark package distributes an existing authored Evals suite, its scenario
documents, and named input files. It is a reusable Runtime interface: a benchmark
author supplies data and scoring definitions; the operator supplies the trusted
application, model, environment, and execution authority.

`BenchmarkPackageV1` uses the current `EvalSuiteDocumentV3` and
`EvalScenarioDocumentV2` contracts. There is no additional case format or runner.
Simple cases use ordinary corpus input. File and multi-stage cases use existing
scenario messages, artifact requirements, and lifecycle events. Static and
model-backed assertions retain their existing scorer identities and authority
rules.

## Distribution and identity

A package directory contains `benchmark.json` and its explicitly named input
files. The manifest records a package ID/version, scorer ID/version, exact authored
suite, exact referenced scenarios, file bindings, and required environment/tool
names. It contains no executable import paths or provider credentials. The target
key comes from the authored suite; loading a package does not resolve that key or
import code.

Create a manifest with `BenchmarkPackageV1.create(...)` and serialize it with
`benchmark_package_to_json(...)`, both exported by `cayu.evals`. Cases retain
their authored stable IDs. Scenario file requirements declare the filename,
media type, size, and SHA-256 digest; `BenchmarkFileV1` binds each exact requirement
to a normalized relative POSIX path. Packages must include every referenced
scenario and exactly the required files. Absolute paths, parent traversal, and
symbolic-link input paths are rejected.

`load_benchmark_package(path)` accepts a manifest or directory. It validates every
declared file before returning a bounded immutable byte snapshot. Missing or
changed material is rejected before candidate dispatch. Limits are 8 MiB for the
manifest, 16 MiB per input file, and 64 MiB total input bytes. Existing authored
suite/scenario case, assertion, message, and event limits still apply. Large
datasets should distribute explicitly selected packages within these limits.

The package revision commits all manifest contents, including exact scenario
and input-file digests, scorer versions, and requirements. It is content identity,
not a publisher signature. `benchmark_suite_selection(package, case_ids)` binds
that revision into existing case source identities and returns an authored suite
and `EvalSuiteSelectionV1`. Selection is deterministic by stable case ID; duplicate,
unknown, and empty explicit selections fail. Omitting `case_ids` selects the full
suite. Different package versions, files, assertions, or cohorts produce different
comparable identities.

`benchmark_package_scenarios(package)` returns the resolved scenario documents
referenced by that suite. Both sides carry matching package provenance, preserving
the native catalog's strict case/scenario identity checks. Save these scenarios
before saving their authored suite into an EvalStore.

## Input and scorer separation

The manifest and assertions are evaluator-owned material. Only declared scenario
input bytes are destined for the candidate environment. Do not mount the package
root into an agent workspace: it can contain reference answers. A file is not
execution authority; launch still needs an existing trusted target environment
and its normal ArtifactStore, file-access, secret, and execution-profile checks.
Required environment and tool names are preconditions, not provisioning requests.

Benchmark integrations can live in separate packages and repositories. Their
adapters create these public contracts and supply ordinary Cayu targets. No
specific benchmark, dataset, provider, or external scheduler is required.

## Installed synthetic material

The wheel includes a small credential-free package builder:

```python
from cayu.evals.benchmark_synthetic import write_synthetic_benchmark_package
from cayu.evals import load_benchmark_package, benchmark_suite_selection

write_synthetic_benchmark_package("synthetic-package")  # must be a new directory
loaded = load_benchmark_package("synthetic-package")
suite, selection = benchmark_suite_selection(loaded.package, ["echo", "attachment"])
```

It includes a text case, a native PNG attachment scenario, and an intentionally
wrong-answer case. The fixture proves packaging and contract behavior; it makes
no claim about model accuracy or external benchmark qualification.
