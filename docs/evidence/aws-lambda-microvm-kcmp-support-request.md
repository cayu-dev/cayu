# AWS support request: Lambda MicroVM guest kcmp returns ENOSYS

Subject: Lambda MicroVM process checkpointing blocked by missing kcmp syscall

We are qualifying process-memory recovery on AWS Lambda MicroVMs. We need to
capture runtime process memory and matching files, terminate the source
allocation, and restore the captured state in a fresh allocation. Initialization
snapshots and suspend/resume of the same allocation do not meet this recovery
requirement.

## Observed environment

- Region: us-east-1; ARM64/aarch64.
- Service-managed base: al2023-1; workspace image reports baseImageVersion 1.0.
- Guest kernel: 6.1.166-24.303.amzn2023.aarch64.
- Trusted process UID: 0.
- CapEff: 000001ffffffffff; NoNewPrivs: 1; Seccomp: 0.
- Image additionalOsCapabilities: ALL.
- CRIU: upstream 4.2, built with CONFIG_NFTABLES=n.
- Source archive SHA-256:
  0c6e51af878e63df7391e6dffbbe5f0ced429bc9f1e5a603020bfd2503065c39.

A real CRIU capture of a quiescent synthetic Python process fails at:

~~~text
Error (criu/kcmp-ids.c:108): kcmp failed: pid (...) type 1 idx (0 0): Function not implemented
Dumping FAILED.
~~~

The direct minimal probe below independently returns -1 / ENOSYS. This happens
in the trusted command context with full elevated capabilities and no active
seccomp filter. The guest kernel configuration is not exposed, so we have not
inferred a particular disabled configuration option.

## Minimal guest reproduction

Run in the trusted ARM64 guest command context:

~~~python
import ctypes
import errno
import json
import os
import platform

assert platform.machine() == "aarch64"
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
ctypes.set_errno(0)
result = libc.syscall(
    ctypes.c_long(272),  # aarch64 __NR_kcmp
    ctypes.c_int(os.getpid()),
    ctypes.c_int(os.getpid()),
    ctypes.c_int(1),    # KCMP_VM
    ctypes.c_ulong(0),
    ctypes.c_ulong(0),
)
print(json.dumps({
    "kernel": platform.release(),
    "architecture": platform.machine(),
    "uid": os.getuid(),
    "kcmp_result": result,
    "errno": errno.errorcode.get(ctypes.get_errno()),
    "process_status": [
        line.strip() for line in open("/proc/self/status")
        if line.startswith(("CapEff:", "NoNewPrivs:", "Seccomp:"))
    ],
}))
~~~

Expected for a supported self/self KCMP_VM comparison: result 0.
Observed: result -1, errno ENOSYS, Seccomp 0.

A separate DMTCP 4.2.0 experiment now restores one pre-launched synthetic
process and matching files into fresh MicroVMs after source termination. That
result does not qualify arbitrary running workloads or a whole-VM capture API;
the CRIU syscall failure remains reproducible.

## Requested guidance

1. Is kcmp intentionally unavailable in the current managed MicroVM kernel?
   Is there a supported managed base/version with CRIU checkpoint primitives,
   and what are its supported process-checkpoint constraints?
2. If this requires a provider kernel change, can AWS enable the necessary
   checkpoint/restore primitives? Additional missing features may become
   visible after kcmp is available; enabling kcmp alone is not a complete
   qualification result.
3. Is there a supported API, preview or planned feature to capture an arbitrary
   running MicroVM's memory and disk into an immutable state that can initialize
   a fresh allocation after the original is terminated?

We can provide the exact private image/version and allocation identifiers
through the authenticated support case if needed. Snapshot contents and
credentials are excluded. All CRIU probe allocations were terminated.

No support case was submitted: DescribeServices returned
SubscriptionRequiredException for the staging account. This request is prepared
for manual submission as requested.
