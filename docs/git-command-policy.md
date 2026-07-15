# Git command policy

`GitCommandPolicy` is Cayu's argv-level authorization layer for a narrow local
repository workflow. It composes a `ProcessCommandPolicy`; the process policy
first decides executable identity, canonical cwd, model environment, stdin,
timeout, and shell capability. Only an allowed process whose exact `argv[0]`
appears in `git_executables` enters the Git parser.

This is deliberately not a general Git sandbox. The policy recognizes only the
matrix below, rejects abbreviated or unknown syntax, and never invokes Git to
interpret an untrusted alias during authorization.

## Secure configuration

```python
from cayu import ExecCommandTool, GitCommandPolicy, ProcessCommandPolicy

process_policy = ProcessCommandPolicy(
    # Match the exact spelling the runner will execute.
    allowed_executables={"/usr/bin/git"},
    allowed_cwds={"/workspace"},
    # Keep model-supplied Git configuration, helpers, proxies, and credentials out.
    allowed_env_names={"LANG"},
    max_timeout_s=30,
)

git_policy = GitCommandPolicy(
    process_policy=process_policy,
    git_executables={"/usr/bin/git"},
    allowed_repositories={"/workspace/repository"},
    max_commit_message_bytes=4096,
)

tool = ExecCommandTool(policy=git_policy)
```

Configuration paths are canonical absolute POSIX paths in the active runner's
namespace. `git_executables` and `allowed_repositories` must both be non-empty.
If the process policy denies or requires command approval, the Git layer returns
that result unchanged. After a general allow, the Git layer denies any command
that is not process form or whose exact executable is absent from
`git_executables`. This fail-closed check prevents a configured path spelling
from bypassing Git parsing by accidentally falling through the general
allowlist. An application exposing Git beside another executable can implement
one small application-owned `CommandPolicy` that delegates each exact identity
to its dedicated policy. Cayu does not introduce a generic policy graph or
command-protocol abstraction for that composition.

## Required invocation prefix

Every supported Git invocation carries these exact controls before the
subcommand:

```text
/usr/bin/git --no-pager -c core.fsmonitor=false <subcommand> ...
```

`--no-pager` prevents Git from dispatching a configured pager. The exact
`core.fsmonitor=false` override prevents `status` and index-refresh paths from
using a repository-configured filesystem-monitor hook. Git documents both the
global no-pager option and the fact that `core.fsmonitor` may name a hook
command. ([Git CLI](https://git-scm.com/docs/git),
[Git config](https://git-scm.com/docs/git-config))

The parser accepts only Git's separate-operand forms of `-C <path>` and
`-c <name=value>`. Attached spellings such as `-Cpath` and `-cname=value` are
rejected because the supported Git CLI does not recognize them. `-C` may
repeat; each operand is applied to the previous effective cwd.
The initial canonical cwd and every effective cwd must remain under the same
configured repository root. Empty operands, `.` or `..` normalization
components, absolute escape, missing operands, and a final path outside that
root fail closed. A global `--` before the subcommand is rejected; the
subcommand-level `--` path separator remains supported where the matrix says so.

All other global options are denied. That includes repository/worktree
redirection, config-from-environment, namespace and executable-path changes,
bare mode, pathspec-mode switches, pager aliases, and arbitrary `-c` values.

## Supported matrix

Arguments not listed here are denied.

| Subcommand | Supported shape |
| --- | --- |
| `status` | Any combination of `--short`, `--porcelain`, `--porcelain=v1`, `--branch`, and `--untracked-files=no|normal|all`; optional safe paths after `--`. |
| `ls-files` | Any combination of `--cached`, `--modified`, `--deleted`, `--others`, and `--exclude-standard`; optional safe paths after `--`. |
| `rev-parse` | Exactly one of `--show-toplevel`, `--show-prefix`, `--is-inside-work-tree`, or `--git-dir`; or exactly `--verify <revision>`. |
| `branch` | Exactly `--show-current`. No branch creation, deletion, rename, or force update. |
| `log` | The inspection-only global override `-c log.showSignature=false`; required `--oneline`; optional `--no-decorate`, `--decorate=short`, and `--max-count=N` where `1 <= N <= 1000`; at most two bounded revisions; optional safe paths after `--`. Patch-producing log options are not supported. |
| `show` | The inspection-only global override `-c log.showSignature=false`; required `--format=medium`, `--no-ext-diff`, and `--no-textconv`; exactly one bounded object expression; optional safe paths after `--`. |
| `cat-file` | Exactly one of `-e`, `-t`, `-s`, or `-p`, followed by one bounded object expression. Options that apply filters are denied. |
| `diff` | Both `--no-ext-diff` and `--no-textconv`; optional `--cached`, `--staged`, `--stat`, `--name-only`, `--name-status`, `--patch`, `--exit-code`, `--quiet`, or `--unified=N` for `0 <= N <= 20`; at most two bounded revisions; optional safe paths after `--`. |
| `add` | Exactly `add -- <path>...` with at least one explicit safe path. Broad `-A`, `--all`, `-u`, interactive/patch modes, pathspec files, stdin pathspecs, and implicit staging are denied. Explicit deleted paths are supported. |
| `commit` | The commit-only global overrides below, then exactly one `-m` or `--message` value plus `--no-verify` and `--no-gpg-sign`. No path arguments or other commit options. |

The diff flags are not cosmetic. Upstream Git documents that
`--no-ext-diff` prevents external diff drivers and `--no-textconv` prevents
external text-conversion filters. ([Git diff](https://git-scm.com/docs/git-diff))
The fixed `log` and `show` formats plus `log.showSignature=false` prevent local
configuration from requesting signature display and dispatching a
repository-selected `gpg.program` during these inspection commands.

### Path subset

Paths must appear after `--`, be non-empty normalized relative POSIX spellings,
and contain no NUL, empty component, `.` or `..` component, `.git` component
(case-insensitive), glob metacharacter, or Git pathspec-magic prefix. Absolute
paths and traversal are denied. Each path is limited to 4,096 UTF-8 bytes and a
single invocation to 256 explicit paths. A leading dash is allowed only because
it is already after `--`. The policy checks lexical containment; it does not
promise race-free filesystem or symlink containment.

### Revision subset

Revisions are byte-bounded and use a conservative ASCII subset for ordinary
ref names, object IDs, `~`/`^` ancestry suffixes, and two-dot or three-dot
ranges. `show` and `cat-file` additionally accept `<revision>:<safe-path>`.
Reflog selectors, leading-option spellings, regex selectors, and syntax the
parser does not recognize are denied rather than delegated to Git.

### Local commit shape and hooks

The commit prefix must contain exactly these three configuration values:

```text
-c core.fsmonitor=false
-c core.hooksPath=/dev/null
-c commit.gpgSign=false
```

The subcommand must also include `--no-verify`, `--no-gpg-sign`, and one bounded
explicit message. `--amend`, fixup/squash, message files, editor launch,
signing, pathspecs, reuse of existing messages, and every other commit option
are denied. `--no-verify` alone is insufficient because Git's `post-commit`
hook is still invoked after a commit; the exact `core.hooksPath=/dev/null`
override is what makes repository hooks undiscoverable for this invocation.
Git's hook documentation describes the pre-commit, commit-message, and
post-commit behavior. ([Git hooks](https://git-scm.com/docs/githooks),
[Git commit](https://git-scm.com/docs/git-commit))

Configure author and committer identity as trusted host or runner state before
starting the agent. The default Git policy does not let the model supply
`GIT_AUTHOR_*`, `GIT_COMMITTER_*`, editor, signing, or helper environment.

## Denied capabilities

Unlisted subcommands fail closed. In particular, the policy denies:

- remote and transport operations (`push`, `pull`, `fetch`, `clone`, remote
  mutation, submodules, upload/receive pack, arbitrary transport helpers);
- credential plumbing and model environment that can select Git config,
  credential helpers, SSH/askpass, pagers, editors, proxies, or alternate
  repository/worktree state;
- destructive or broader history mutation (`reset`, `clean`, checkout/switch,
  restore, rebase, merge, cherry-pick, revert, replace, notes/tag mutation,
  forced branch updates, worktree mutation, reflog expiry, GC/repack and
  maintenance);
- arbitrary aliases and `-c` configuration, option abbreviation, missing
  operands, shell source, and stdin.

Denial reasons identify only the rejected category. They never interpolate the
argv item, path, revision, commit message, environment value, stdin, repository
content, credential, URL, or command output. The terminal
`tool.call.blocked` result therefore stays category-only and carries
`denied_by=command_policy`. The ordinary Cayu
audit/replay contract still records the model's original tool arguments on
`tool.call.started`; do not place secrets in command argv, environment, stdin,
or commit messages when that event stream is not an appropriate secret store.

## Residual risk and isolation

This policy authorizes argv; it does not make Git or a repository trustworthy.

- Git's check-in conversion is used by `git add` and may also be consulted when
  Git compares worktree content. Repository-controlled `.gitattributes` plus
  local filter configuration can select a `clean` or long-running `process`
  command. Git documents these filter and diff/check-in interactions. There is
  no supported argv control in this matrix that disables every named clean
  filter. Run the whole workflow inside a container or microVM, and do not
  expose host credentials or writable paths to the guest. ([Git attributes](https://git-scm.com/docs/gitattributes),
  [Git add](https://git-scm.com/docs/git-add))
- Supported commands still read repository objects, refs, index data,
  attributes, and selected local configuration. `--no-ext-diff`,
  `--no-textconv`, no-pager, the fsmonitor override, and the commit hook/signing
  controls close the executable-dispatch paths covered by the matrix; they do
  not make arbitrary repository bytes benign.
- A path can change after policy evaluation. Runner root containment and OS
  isolation remain the enforcement boundary for filesystem races and symlinks.
- Denying Git transport operations is local authorization, not network-egress
  enforcement. Use the runner or deployment network policy as the network
  boundary.
- Allowing another programmable executable in the composed process policy can
  bypass Git-specific intent completely. Give each such executable its own
  justified policy or approval boundary.
- Git versions can add syntax or behavior. Exact allowlisting makes unknown
  syntax fail closed, but upgrades still require rerunning the adversarial and
  real-repository tests before expanding this matrix.

Without any command policy, Cayu preserves its compatibility contract and
passes valid model-controlled command arguments to the runner. Attaching this
policy does not replace container, microVM, filesystem, credential, or network
isolation.
