# Recipe: GitHub CLI through virtual egress

**Goal:** let an agent use the unmodified GitHub CLI while keeping the real
GitHub token outside the runner and restricting which GitHub HTTP operations
the session may attempt.

The complete no-key proof is
[`examples/github_cli_virtual_egress.py`](../../examples/github_cli_virtual_egress.py):

```bash
uv run --extra egress python examples/github_cli_virtual_egress.py
```

It builds a checksum-pinned Linux `gh` image, runs `gh api user` inside Cayu's
enforced Docker egress topology, and routes the request to a fake GitHub
upstream. The proof verifies that the runner receives only a virtual `GH_TOKEN`,
the broker forwards only the real token, and neither token crosses the wrong
side of the broker.

## 1. Put `gh` in the runner image

Cayu routes an installed CLI; it does not install one into a running sandbox.
Build or select a Linux runner image containing `gh` before the session starts.
Pin and verify the CLI package or release in production. The no-key example
shows one checksum-pinned image build for `amd64` and `arm64`.

Do not allow a package manager merely to install `gh` at session time. That
would require a much broader destination policy for package indexes, mirrors,
signing keys, and transitive downloads.

## 2. Give the runner a virtual `GH_TOKEN`

Keep the real credential in the trusted application process or a production
vault. `LocalEnvVault` is convenient for local development:

```python
from cayu import (
    HttpEgressPolicy,
    LocalEnvVault,
    SecretRef,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)

OWNER = "acme"
REPO = "payments"
POLICY = "github-cli-read"

vault = LocalEnvVault({"github_cli_token": "GITHUB_TOKEN"})
factory = VirtualEgressEnvironmentFactory(
    resolver=vault,
    policies={
        POLICY: HttpEgressPolicy(
            name=POLICY,
            allowed_hosts=["api.github.com"],
            allowed_endpoints=[
                ("GET", "/user"),
                ("GET", f"/repos/{OWNER}/{REPO}"),
                ("GET", f"/repos/{OWNER}/{REPO}/issues"),
                ("GET", f"/repos/{OWNER}/{REPO}/pulls"),
            ],
        )
    },
    credentials=[
        VirtualCredentialSpec(
            env_name="GH_TOKEN",
            secret=SecretRef(name="github_cli_token"),
            destination="api.github.com",
            policy_name=POLICY,
            credential_kind="opaque_token",
        )
    ],
    runner_kind="docker",
    image="your-linux-image-with-gh",
)
```

That explicit Docker selection is suitable for trusted development and CI; its
network policy is enforced, but an ordinary container is not a security
boundary for hostile code. For untrusted agent-authored code, supply a
registered enforcing Microsandbox, E2B, or other sandbox adapter with the same
credential and policy declarations.

`opaque_token` is provider-neutral. It means an opaque secret carried with the
literal `Authorization: token …` scheme used by GitHub CLI. Use
`opaque_bearer` for CLIs that send `Authorization: Bearer …`. Cayu rejects a
request when the presented authorization scheme does not match the declared
credential kind; it does not silently reinterpret one shape as another.

The trusted process reads the real `GITHUB_TOKEN`; the runner receives a
generated virtual value in `GH_TOKEN`. Prefer a short-lived, least-privilege
GitHub App installation token in production. A fine-grained personal access
token is reasonable for a developer-owned local profile.

## 3. Start with explicit REST reads

The strict read profile uses `gh api` with REST `GET` endpoints whose method and
path Cayu can authorize directly:

```bash
gh api user
gh api repos/acme/payments
gh api --method GET repos/acme/payments/issues -f state=open
gh api --method GET repos/acme/payments/pulls -f state=open
```

Query strings do not change the policy path. A host, method, or path absent
from `allowed_endpoints` is denied before the vault resolves the real token.

High-level commands such as `gh repo view`, `gh issue list`, and `gh pr checks`
may use GitHub's GraphQL endpoint. GraphQL reads and mutations both travel as
`POST /graphql`, so Cayu's HTTP method/path policy cannot distinguish them.
There are two honest profiles:

- Keep the strict Cayu-enforced read profile above and use explicit REST `GET`
  commands.
- Allow `POST /graphql` only with a provider credential that itself has
  read-only permissions. In that profile GitHub, not the HTTP policy, is the
  authority preventing GraphQL mutations.

## 4. Treat mutations as a separate capability

Virtual egress protects the credential and authorizes HTTP destinations. It
does not approve business effects. For example, creating an issue requires a
separate policy containing the exact REST route:

```python
HttpEgressPolicy(
    name="github-issue-writer",
    allowed_hosts=["api.github.com"],
    allowed_endpoints=[("POST", "/repos/acme/payments/issues")],
)
```

Put that policy and its write-scoped credential in a separate environment or
trusted tool. Require the appropriate Cayu approval before selecting or
invoking that capability. Do not describe an allowed `POST` route as an
approval: network authorization and human/business authorization are
orthogonal.

## Compatibility boundary

- Proven: checksum-verified GitHub CLI 2.86.0 on Linux, `GH_TOKEN`, standard
  HTTPS proxying, Cayu's session CA, and `GET /user` through the real proxy and
  broker path.
- Linux runners honor the `SSL_CERT_FILE` contract Cayu injects. Go's
  [`SystemCertPool` documentation](https://pkg.go.dev/crypto/x509#SystemCertPool)
  explicitly excludes macOS from that environment-variable override; a
  host-installed macOS `gh` needs the session CA installed in the platform
  trust store and is not covered by this recipe.
- `gh repo clone`, `gh pr checkout`, and other Git operations introduce Git
  credential handling and additional `github.com` routes. Test and authorize
  them as a separate profile.
- Extensions are executable third-party code and may contact arbitrary hosts.
  Installing or running them is outside this default profile.
- GitHub Enterprise Server uses `GH_ENTERPRISE_TOKEN`, the enterprise hostname,
  and its REST path prefix. Give each enterprise host its own credential and
  destination policy; do not reuse the `api.github.com` declaration blindly.

See the [GitHub CLI environment reference](https://cli.github.com/manual/gh_help_environment),
[`gh api` reference](https://cli.github.com/manual/gh_api), and Cayu's
[virtual-egress contract](../virtual-egress.md).
