# Send CLI evaluation results to Evals

`cayu eval run` can publish completed evaluation evidence to a running Cayu
server. Publication is opt-in. Without reporting flags, execution, local result
files and exit codes retain their ordinary behavior.

Reporting uses server contract 67 and storage revision 132. Configure an EvalStore
and target registry on the destination server and migrate its SQLite or PostgreSQL
store with the normal `cayu storage migrate` workflow. In-memory storage supports
the same operations but does not survive server restart. Custom EvalStore adapters
remain unsupported until they implement the reporting methods and capability.

## Configure a reporting credential

Give CI a dedicated random bearer credential through its secret manager. The
server stores only the SHA-256 digest and an explicit grant. Its environment
configuration is `CAYU_EVAL_REPORTERS`, a JSON array:

```json
[
  {
    "publisher_id": "github:example/reviewer",
    "token_sha256": "REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS",
    "target_keys": ["REPLACE_WITH_REGISTERED_TARGET_KEY"]
  }
]
```

Find the target key in the authenticated `/api/evals/targets` catalog. A Python
module name or suite name does not select a deployed specialist automatically.
To calculate the digest without putting the credential in a command argument:

```sh
python -c 'import hashlib, os; print(hashlib.sha256(os.environ["CAYU_EVAL_REPORT_TOKEN"].encode()).hexdigest())'
```

The credential can publish only to its configured targets. It cannot list
sessions, read results, launch native evaluation, promote a baseline or use
operator endpoints. Operator credentials do not authorize publication. To rotate
a reporting credential, temporarily configure both digests with the same stable
`publisher_id` and target grants; exact retries retain their original receipt.
Remove the old digest after clients have switched. Keep publisher IDs stable
across rotation and grant changes.

Hosts with an existing identity verifier can instead supply
`ServerConfig(eval_reporting=EvalReportingAccess(authenticate=...))`, or the same
`eval_reporting` argument to `mount_cayu`. The async callback receives a request
and returns `EvalReportPrincipal(publisher_id=..., target_keys=(...))` from
`cayu.server.eval_report_auth`, or `None` to deny access. For example, an existing
OIDC verifier can map its verified subject to this grant. The callback owns issuer,
audience and signature validation; caller JSON is never an authority source.
This extension does not add an issuer, browser login or global role.

## Publish a completed run

```sh
cayu eval run evals.reviewer:build \
  --output result.json \
  --report-to https://dashboard.example/api \
  --report-target YOUR_REGISTERED_TARGET_KEY \
  --report-file report.json
```

The CLI reads `CAYU_EVAL_REPORT_TOKEN`, or `--report-token-file FILE`. Credentials
are never stored in the report envelope. Remote destinations require HTTPS.
For deliberate local development, use both `--report-to http://localhost:8000/api`
and `--report-allow-loopback-http`. Redirects are not followed. This initial
workflow publishes through HTTP, including local servers; it does not open the
destination database directly.

Ordinary direct suites, corpus suites, workflows and multiprocess direct runs
support reporting. Benchmark campaigns reject reporting options before execution.
Multiprocess workers prepare sanitized evidence while they still own their
application redactor. Raw trajectories, private metadata and local session handles
are excluded from direct reports. Corpus evidence retains its existing published
contract. Both the client and server check configured secrets before publication.
The JSON report is bounded to 40 MiB plus 64 KiB; direct evidence permits at most
1,000 cases, 100 trials per case, 256 checks per trial and 64 KiB per text field.

`--report-program NAME` supplies a stable source program label; its default is the
resolved Python target. `--report-suite-revision sha256:...` optionally declares a
direct suite revision. Omission stays unknown. The CLI does not manufacture a suite
revision from the commit or result digest. Histories group reports by authenticated
publisher, destination target, program and suite, separately from native suites.

For corpus execution, `--report-associate-corpus` requests association with an
already stored corpus and suite. The server validates the exact target, corpus,
suite, cases and execution contract before accepting the association. An absent
or different contract rejects publication; it does not import a corpus or infer
an authored-suite relationship. Without this flag, the report stays in its
reported series. A successful association still does not prove that the deployed
live target executed the CI run.

Reports are external evidence. Their backend read views expose history, metadata,
case evidence and a JSON download, with comparison explicitly unavailable. They
cannot be promoted as native baselines or compared with native, captured or other
reported results. Existing native comparison rules remain owned by native APIs.
Case evidence above 64 KiB is omitted from the read view with an explicit marker;
the case status, score and total trial count remain visible. The complete evidence
is preserved unchanged in the report download, which requires developer access.
This also applies to corpus reports. Filters and run totals always use the full
evidence, not the bounded view. Structured outputs must contain finite JSON numbers.
Successful upload receipts are validated against the exact submitted identity and
screened for credential echoes before the CLI prints them.

## CI metadata

On GitHub Actions, the CLI reads repository, branch, `GITHUB_SHA`, run ID, attempt,
job and run URL. It reads the bounded GitHub event file to retain the PR number
and PR head SHA separately. For pull-request merge testing, `commit` therefore
retains the tested/event commit while `pr_head_commit` retains the contributor's
head. If checkout selects a different commit than `GITHUB_SHA`, explicitly pass
`--report-commit` with the checked-out SHA.

Each value can be overridden with `--report-provider`, `--report-repository`,
`--report-branch`, `--report-commit`, `--report-pr-head-commit`,
`--report-pull-request`, `--report-run-url`, `--report-run-id`,
`--report-run-attempt` or `--report-job`. The envelope records whether each retained
value came from a flag or environment. These fields are publisher-supplied
attribution, not verified authentication or evidence of native compatibility.
Run URLs require HTTPS without credentials, query parameters or fragments.

## Retain and retry the exact report

The CLI writes a new sanitized envelope before dispatching the POST. A supplied
report file must not already exist; without `--report-file`, the CLI chooses a
new file under `.cayu/evals/reports/`. Relative report and credential file paths,
like the result output path, resolve from the discovered project root for
`eval run`. The file binds the destination and immutable
report operation. If evaluation completed but saving or uploading its report
fails, the normal result output remains available and the command exits 2.
A failed or unavailable evaluation that uploads successfully retains its ordinary
exit status.

Retry a saved report with:

```sh
cayu eval publish report.json --report-to https://dashboard.example/api
```

Supply the API URL from trusted deployment configuration, independently of the
saved file. Before reading the credential, this command requires that URL to
match the saved destination. The unsigned artifact cannot authorize its own
upload destination. Loopback HTTP additionally requires an explicit
`--report-allow-loopback-http` on this retry command; the saved flag alone is
not permission. Retry paths resolve from the current working directory and do
not discover or import a project. The command never imports the Python target, calls the provider or reruns evaluation. The server atomically
binds the authenticated publisher and operation ID to the complete canonical
envelope. Exact duplicates return the original receipt, including after lost
acknowledgement, token rotation or store restart. Changed evidence, metadata,
target, program or declared revision under the same operation conflicts without
changing the original. There is no background queue or separate receipt lookup
requirement.

A file left only on an ephemeral CI runner does not survive teardown. Retain the
credential-free envelope as a CI artifact, especially on upload failure:

```yaml
- name: Run and report evals
  env:
    CAYU_EVAL_REPORT_TOKEN: ${{ secrets.CAYU_EVAL_REPORT_TOKEN }}
  run: >-
    cayu eval run --output result.json
    --report-to https://dashboard.example/api
    --report-target YOUR_REGISTERED_TARGET_KEY
    --report-file report.json
- name: Retain exact publication evidence
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: eval-publication
    path: report.json
    if-no-files-found: ignore
```

Choose artifact access and retention for the evaluation content. Download the
same envelope and run `cayu eval publish --report-to TRUSTED_API_URL` with a
currently authorized reporting credential to reconcile an uncertain upload. Keep the saved destination and
operation intact for an exact retry.
