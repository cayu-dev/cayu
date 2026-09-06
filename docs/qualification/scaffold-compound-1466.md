# Compound scaffold qualification for Runtime #1466

The strict consumer gate passes with this Runtime change and the companion
Compound import-initialization changes. This is source/strict-deployment
qualification, not full composed lifecycle or production qualification.

## Exact consumer and installed-wheel evidence

The consumer is based on `cayu-tech/cayu-compound` PR #527, commit
`6b1db54c87929d896c66fed207d8cada6be3d789`. The companion branch is
`fix/1466-import-inertness`. Its entire canonical source remains inside the
normal scan; `tests/test_canonical_cli.py`, its strict assertions and its
60-second timeout are unchanged. Runtime's base is
`50fcad7456544e165648c2512e908f45ffb83724`.

A built wheel of Runtime revision `f82dede3d9af49a3f17604bbe7c77139df482026`
was installed into a disposable consumer
environment. Import provenance was `site-packages/cayu/__init__.py`, not a source
checkout. Wheel SHA-256:
`a24d90a8861bd451414ad5dcdeb7ff288e199bfb28547f17bc1d32bd41a634d3`
(6,449,898 bytes). Python was 3.14.2; consumer dependencies included Pydantic
2.14.0b1, pydantic-core 2.48.0, pytest 8.4.2 and pytest-asyncio 0.26.0.

The later PR #1469 review fix preserves namespace-package imports and validates
their explicit children, parent initializers and module/package precedence.
The consumer and wheel evidence below applies to the revision above; Compound
acceptance and wheel measurement were not rerun for that follow-up fix.

| Consumer source | Full static result with this wheel |
| --- | --- |
| Unchanged PR #527 commit above | 97 findings in 5.80 seconds: 79 unproven class bases and 18 unsupported expressions |
| Companion changes | Zero findings in 5.76 seconds |

The original source still constructs model defaults, computes two fingerprints
and acquires a logger at import. Those operations were not blanket-allowlisted.
Unsupported declarations can also cause schema-reference and inheritance
findings; diagnostic counts are not counts of independent effects.

The following consumer group passed: **44 tests in 23.48 seconds**.

```sh
.venv/bin/pytest -q \
  tests/test_canonical_cli.py \
  tests/test_public_service_security.py \
  tests/test_import_initialization.py \
  tests/test_canonical_ownership.py \
  tests/test_canonical_prompt_parity.py \
  tests/test_evaluation_strategy.py
```

The unchanged canonical CLI test prepares the synthetic lab and runs
`cayu check --deploy --fail-on warning --json`. It requires no diagnostics, an
available manifest fingerprint, authenticated control-plane evidence and the
maintained service contract. The group also checks preserved worker/transaction
method AST hashes, prompt bytes, model default values, log names/caller locations
and exceptions, and canonical synthetic fingerprint bytes.

## Runtime proof

The checker builds one bounded graph of explicit source imports and package
initializers. It establishes unique, ordered declaration bindings to a fixed
point, then invalidates unsafe dependency closures. Import cycles cannot
bootstrap a class-inheritance proof. Computed bindings and input caches belong
to one scan; editing a dependency cannot reuse stale trust on the next scan.
The graph has a 1,024-module bound and fails closed when it is exceeded.

The new proofs cover:

- Configured local Pydantic models, re-exports, schema aliases, typed enum members,
  `Literal` enum discriminants and `Annotated`/`Field` discriminated unions.
- Immutable literal aliases and bounded integer products used as field options.
- Deferred built-in, proven model, plain function and zero-argument lambda
  factories. A lambda body is not invoked at declaration; evaluated lambda
  defaults and opaque callable objects remain unsupported.
- Deferred `exclude_if` predicates and typed `computed_field` declarations.
- TypeVar bounds, exact `asyncio.Future` subscriptions, literal `timedelta`, empty
  `set`, and reviewed literal Runtime retry/identity metadata.
- The standard HTTP server/handler roots, with namespace checks before trusting
  an application subclass.

Hooks, descriptors (including leaf-class descriptors), class or import
rebinding, wildcard imports, shadowed external package prefixes, arbitrary
invocation and application/model/resource instance construction remain blocked.
A supported application type or callable reference never authorizes invoking it at import.

Diagnostics distinguish unsupported expressions from unproven bases. They
prefer the primary expression within a declaration, include failures in explicit
helper imports, and attach up to eight blocking module paths plus the full
blocking-module count to dependent-base findings. Diagnostic output contains
symbols and locations, not source literals. No errors are downgraded or hidden.

## Companion changes and remaining limits

Compound changes two model-instance defaults to `Field(default_factory=...)`,
computes synthetic fingerprint material at its explicit use site, and acquires
its logger inside explicit logging entry points. Canonical concern ownership,
worker method bodies, the scaffold contract and strict acceptance test remain
intact. The companion PR pins the exact Runtime revision and measured wheel
identity used for its final installed-package checks.

Runtime regression coverage passed 402 tests with one existing Windows-only
skip on macOS: the complete scaffold/check files and focused inheritance,
validator, factory and graph tests. The final graph file also passed all 43
nodes after the final import-target type guard; Ruff lint/format and focused
`ty check` passed. The generated composed
service exercises the same declarations through public strict checking, with
negative controls that prevent executing effectful source.

No full Runtime/Compound suite, PostgreSQL/Docker operational qualification,
paid evaluation or external deployment is claimed. Strict acceptance does not
replace execution/restart/settlement qualification.
