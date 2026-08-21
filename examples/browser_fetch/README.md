# Sandboxed browser inspection image

This image is the versioned guest half of `BrowserWebFetchAdapter`,
`ScreenshotPageTool`, and the opt-in `BrowserSessionTool`. It contains
Playwright 1.62.0, its matching Chromium build, NSS tooling for the per-session
Cayu CA, the closed `cayu.browser-fetch.v4` one-shot protocol, and the closed
`cayu.browser-session.v2` stateful protocol. The image runs as the
non-root `pwuser`; the root-owned, read-only worker refuses root execution and
launches Chromium with its browser sandbox enabled. Other commands running as
the guest user cannot replace the versioned worker between invocations.
An independent, uncredentialed guardian owns each temporary browser profile,
deletes it on normal release or worker exit, and turns a deletion that cannot
settle within the request's cleanup reserve into `cleanup_failed`. Cayu's
managed Docker runner starts the container with Docker's minimal init so a
guardian orphaned by a worker crash is reaped after cleanup; other admitted
runners must provide an equivalent reaper or supervisor.

Build from the repository root:

```bash
docker build \
  --file examples/browser_fetch/Dockerfile \
  --tag cayu-browser-fetch:6-playwright-1.62.0 \
  .
```

The Python slim base is pinned by multi-platform manifest digest. A multi-stage
build extracts the setuid sandbox helper from Playwright's matching full
Chromium revision, then ships only the matching headless shell and that helper.
This retains Chromium's browser sandbox without carrying the full headed
browser in the final image. Updating the base or Playwright requires changing
the Dockerfile, the three public browser version constants, the worker
handshake, and their tests together.

Applications opt in by selecting this image in a virtual-egress environment,
declaring every document and subresource host as an
`ApprovedEgressDestination`, and using a `BrowserEgressPolicy`. The tool keeps
the normal model-facing name and input schema:

```python
from cayu import (
    ApprovedEgressDestination,
    BrowserEgressPolicy,
    BrowserWebFetchAdapter,
    ScreenshotPageTool,
    WebFetchTool,
)

browser_policy = BrowserEgressPolicy(
    name="product-docs",
    allowed_hosts=["docs.example.com", "static.example.com"],
    allowed_path_prefixes=["/"],
)
approved_destinations = [
    ApprovedEgressDestination(
        destination=host,
        policy_name="product-docs",
        protocol="https",
        port=443,
    )
    for host in ("docs.example.com", "static.example.com")
]
web_fetch = WebFetchTool(adapter=BrowserWebFetchAdapter())
screenshot_page = ScreenshotPageTool()
```

Pass `policies={"product-docs": browser_policy}`,
`approved_destinations=approved_destinations`, `credentials=[]`, an adapter
constructed with
`DockerEgressAdapter(seccomp_profile="/absolute/path/to/examples/browser_fetch/seccomp_profile.json")`,
and `image="cayu-browser-fetch:6-playwright-1.62.0"` to
`VirtualEgressEnvironmentFactory`. Plain Docker proves the enforced networking
path for trusted development and CI; it is not Cayu's untrusted-code isolation
boundary. Applications that require a stronger boundary use the same adapter
with an admitted runner/image that implements the identical worker protocol.

The browser adapter fails closed when the runner lacks current capability
evidence for deny-by-default networking, brokered egress, confirmed
cancellation, or confirmed cleanup. It never falls back to host-process HTTP.
All browser destinations remain application-owned: redirects and subresource
hosts are not inferred or auto-approved.

Rendered pages without meaningful interactive or relational structure retain
the compact `text` result. Links, tables, forms, navigation landmarks, and
interactive labels—including controls in open shadow roots—deterministically
select an accessibility-tree-depth-, aggregate-composed-DOM-node-, and
byte-bounded `accessibility` representation. Trusted metadata identifies the
representation and truncation state before the untrusted model-facing page
envelope. Applications can lower the DOM-node ceiling with
`BrowserWebFetchAdapter(max_dom_nodes=...)`; it is host configuration and is
never exposed to the model. The worker freezes page-authored JavaScript after
the render-settle period and performs node accounting from a browser-owned
isolated world, keeping page-defined getters and prototype overrides outside
the inspection boundary. It aggregates up to 32 admitted main/child frame
documents in stable tree order, labels each frame section with its URL, and
shares the configured DOM/content limits across the complete page rather than
granting each frame a separate allowance. Frame subtrees ignored by Chromium's
accessibility tree remain counted for safety but are excluded from model-facing
evidence.

`ScreenshotPageTool` uses the same worker and egress boundary. Its only model
presentation option is `full_page`; PNG format, viewport, dimensions, pixels,
bytes, and all navigation/network limits are host configuration. The active
environment must include an artifact store. Successful captures are stored as
session-scoped PNG artifacts and returned as provider-neutral image attachments;
worker base64 is bounded internal transport and never appears in model-facing
text or structured output.

The checked-in seccomp profile is Playwright 1.62.0's
[published Docker profile](https://github.com/microsoft/playwright/blob/v1.62.0/utils/docker/seccomp_profile.json):
Docker's normal syscall allowlist plus `clone`, `setns`, and `unshare`, which
Chromium needs to create its own sandbox namespaces. The adapter requires an
absolute existing host path and passes it to Docker explicitly. It never uses
an unconfined profile, `SYS_ADMIN`, or privileged mode.
