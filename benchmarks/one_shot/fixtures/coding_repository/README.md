# Safe command-selector regression fixture

This deterministic fixture exercises the coding-repository benchmark's
selector boundary without a provider, network access, or a real test runner.
It is an executable-specific example, not a reusable argv policy language.

The normal, linted `cayu.guides.command_selectors` module is the canonical
selector grammar. `cayu guide authoring` renders that module's marked source
region, while `safe_selector_check.py` imports and calls the same function. No
Markdown is executed, no second grammar is maintained, and no top-level runtime
policy API is exposed. The recipe accepts only relative Python files below
`tests/` plus simple `::node_id` suffixes. It rejects option-like, absolute,
traversal, empty,
normalized spellings such as `//`, `/./`, and a trailing slash, and other
malformed values before starting the process. The fixed
`fixture_check_program.py` documents and implements `--` as its end-of-options
delimiter, so the wrapper may safely use it. Do not infer that another
executable accepts a delimiter in the same position.

The fixture reports rejection, unavailable or malformed executable, timeout,
failed check, zero tests executed, and verified success as different process
statuses, with stable reasons and errno evidence for launch failures. It reports
full versus selected scope and preserves exact validated selectors. It snapshots
the workspace and explicit protected paths so tests compare the declared `none`
effect with observed writes instead of trusting a declaration or zero exit
status; effect mismatch remains separate from process status.

This fixture does not provide sandbox isolation. Production coding agents must
still run untrusted repository code in an appropriate container or microVM.
