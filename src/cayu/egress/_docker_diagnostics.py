"""Docker stderr projection for durable setup failures.

Docker's message is shown so a developer can see why setup failed. Before it is
stored, known secrets are removed (the sidecar transport token and any caller
redactor), common credential shapes are masked (URL userinfo, authorization and
other secret header values, cookies, credential-named ``KEY=value`` pairs,
quoted and escaped ``"key": "value"`` mappings, ``password: x`` lines,
credential flags such as ``--password x`` and ``docker login -p x``, PEM and PGP
private keys), control characters are flattened, and only then is the text
bounded. Command arguments
are never copied; only a fixed operation name is.

Remaining risk: an unregistered secret that matches none of those shapes is
shown if Docker echoes it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from cayu.vaults import SecretRedactor

_MAX_STDERR_BYTES = 1024
_MAX_SCAN_CHARACTERS = 65536
_REDACTED = "[REDACTED]"
# Every pattern here runs on up to 64 KiB of daemon-controlled text on the event
# loop, so each one is linear: matches start only at a fixed anchor or at the
# beginning of a run, and repeated parts are possessive.
_CREDENTIAL_WORD = re.compile(r"(?i)token|secret|password|passwd|key|credential|auth|cookie")
# Short names count only as a whole name part: DB_PASS, PW, mysql-pwd, not passport.
_SHORT_CREDENTIAL_WORD = re.compile(r"(?i)(?:^|[_.-])(?:pass|pw|pwd)(?:$|[_.-])")
# Unquoted ``name: value`` lines: the name must end in a credential term, so
# messages such as "unauthorized: authentication required" stay readable.
_LINE_MAPPING_NAME = re.compile(
    r"(?i)(?:^|[_.-])(?:password|passwd|pass|pw|pwd|secret|token|key|apikey|credentials?)$"
)
_SECRET_HEADER = (
    r"x-api-key|api-key|apikey|x-auth-token|private-token|job-token|x-registry-auth|"
    r"x-amz-security-token|x-goog-api-key|x-vault-token"
)
_PEM = re.compile(
    r"-----BEGIN [A-Z ]{0,64}PRIVATE KEY(?: BLOCK)?-----.*?"
    r"(?:-----END [A-Z ]{0,64}PRIVATE KEY(?: BLOCK)?-----|\Z)",
    re.DOTALL,
)
# Userinfo before ``@`` in a URL: user:password, or a bare token.
_URL_USERINFO = re.compile(r"(?<=://)[^/\s@?#]++(?=@)")
# Quoted mappings, masked even when the value's closing quote is missing (for
# example when the scan limit cut it): JSON, Python reprs and escaped JSON.
_QUOTED_MAPPINGS = (
    re.compile(
        r'(?P<keep>\\"(?P<name>[A-Za-z0-9_.-]++)\\"[ \t]*+:[ \t]*+\\")'
        r'(?:[^\\\n]|\\[^"\n])*+(?P<close>(?:\\")?)'
    ),
    re.compile(
        r'(?P<keep>"(?P<name>[A-Za-z0-9_.-]++)"[ \t]*+:[ \t]*+")'
        r'(?:[^"\\\n]|\\.)*+(?P<close>"?)'
    ),
    re.compile(
        r"(?P<keep>'(?P<name>[A-Za-z0-9_.-]++)'[ \t]*+:[ \t]*+')"
        r"(?:[^'\\\n]|\\.)*+(?P<close>'?)"
    ),
)
_LINE_MAPPING = re.compile(
    r"(?m)^(?P<keep>[ \t]*+(?:-[ \t]++)?(?P<name>[A-Za-z0-9_.-]++):[ \t]++)\S[^\n]*+"
)
_HEADERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)(?P<keep>authorization:)\s*+\S++(?:[^\S\n]++\S++)?"),
        "\\g<keep> ",
    ),
    (re.compile(rf"(?i)(?P<keep>\b(?:{_SECRET_HEADER}):)\s*+\S++"), "\\g<keep> "),
    (re.compile(r"(?i)(?P<keep>\b(?:set-)?cookie:)[^\n]*+"), "\\g<keep> "),
)
# Whitespace runs (a masked value is a whole run), and the command words and
# separators inside them.
_WORD = re.compile(r"\S++")
_COMMAND_TOKEN = re.compile(r"[^\s;|&]++|[;|&]++")
_QUOTE = re.compile(r"[\"']")
_CONTAINER_CLIS = frozenset({"docker", "podman", "docker.exe", "podman.exe"})
# Global options of those CLIs that take a separate value before the subcommand.
_CLI_VALUE_OPTIONS = frozenset(
    {
        "-c",
        "-H",
        "-l",
        "--config",
        "--context",
        "--host",
        "--log-level",
        "--tlscacert",
        "--tlscert",
        "--tlskey",
        "--connection",
        "--url",
        "--identity",
        "--root",
        "--runroot",
        "--storage-driver",
    }
)
_CREDENTIAL_FLAGS = frozenset(
    {
        "--password",
        "--passwd",
        "--token",
        "--secret",
        "--api-key",
        "--access-token",
        "--auth-token",
        "--client-secret",
    }
)
# NAME= at the start of a name run, including after another ``=`` (--env=TOKEN=x).
_KEY_NAME = re.compile(r"(?<![A-Za-z0-9_.-])(?P<name>[A-Za-z0-9_.-]++)=")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
# Before masking: drop ANSI CSI sequences and turn other controls and
# zero-width characters into spaces, so they cannot hide a flag or key from
# the masks and then vanish when the text is flattened.
_ANSI_CSI = re.compile(r"(?:\x1b\[|\x9b)[0-?]*+[ -/]*+[@-~]")
_INVISIBLE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f\u200b-\u200d\u2060\ufeff]++")
_OPERATIONS = frozenset({"run", "exec", "start", "stop", "rm", "pause", "unpause", "inspect"})
_NETWORK_OPERATIONS = frozenset({"create", "connect", "disconnect", "rm", "inspect", "ls"})


def docker_setup_failure(
    argv: Sequence[str],
    exit_code: int,
    stderr: str,
    *,
    redactor: SecretRedactor | None = None,
) -> str:
    """Describe a failed Docker command with its stderr, minus known secrets."""
    operation = "operation"
    if argv and argv[0] in _OPERATIONS:
        operation = argv[0]
    elif len(argv) >= 2 and argv[0] == "network" and argv[1] in _NETWORK_OPERATIONS:
        operation = f"network {argv[1]}"
    return (
        f"docker {operation} failed while preparing egress (exit_code={exit_code}); "
        f"stderr: {_safe_stderr(stderr, redactor or SecretRedactor())}"
    )


def _safe_stderr(stderr: str, redactor: SecretRedactor) -> str:
    if not stderr.strip():
        return "[unavailable]"
    window = stderr[:_MAX_SCAN_CHARACTERS]
    if len(stderr) > _MAX_SCAN_CHARACTERS:
        # A secret cut at the scan limit, or one nested inside it, would survive
        # as a fragment: drop everything that starts within the longest secret's
        # length of the cut, by position, while redacting the original window.
        # Then back up to whitespace so an unregistered credential shape is not
        # left half-cut either.
        text = redactor.redact_cut_text(window, cut_tail=True)
        boundary = max(text.rfind(character) for character in " \t\r\n")
        text = text[:boundary] if boundary > 0 else ""
    else:
        text = redactor.redact_text(window)
    text = _mask_credentials(_INVISIBLE.sub(" ", _ANSI_CSI.sub("", text)))
    text = " ".join(_CONTROL.sub(" ", text).split())
    bounded, truncated = redactor.redact_text_bounded_with_marker(
        text,
        max_bytes=_MAX_STDERR_BYTES,
        truncation_marker="...[truncated]",
    )
    if len(stderr) > _MAX_SCAN_CHARACTERS and not truncated:
        bounded += "...[truncated]"
    return bounded or _REDACTED


def _mask_credentials(text: str) -> str:
    text = _PEM.sub(_REDACTED, text)
    text = _URL_USERINFO.sub(_REDACTED, text)
    for pattern in _QUOTED_MAPPINGS:
        text = pattern.sub(_mask_quoted_mapping, text)
    for pattern, keep in _HEADERS:
        text = pattern.sub(keep + _REDACTED, text)
    text = _LINE_MAPPING.sub(_mask_line_mapping, text)
    return "\n".join(_mask_key_values(_mask_flag_values(line)) for line in text.split("\n"))


def _is_credential_name(name: str) -> bool:
    return (
        _CREDENTIAL_WORD.search(name) is not None or _SHORT_CREDENTIAL_WORD.search(name) is not None
    )


def _mask_quoted_mapping(match: re.Match[str]) -> str:
    if not _is_credential_name(match.group("name")):
        return match.group()
    return f"{match.group('keep')}{_REDACTED}{match.group('close')}"


def _mask_line_mapping(match: re.Match[str]) -> str:
    value = match.group()[len(match.group("keep")) :]
    if _LINE_MAPPING_NAME.search(match.group("name")) is None or value.startswith(_REDACTED):
        # Header shapes already masked their value and keep the text after it.
        return match.group()
    return f"{match.group('keep')}{_REDACTED}"


def _mask_key_values(line: str) -> str:
    """Mask the value of every credential-named ``NAME=`` on the line."""

    masked: list[tuple[int, int]] = []
    consumed = 0
    for match in _KEY_NAME.finditer(line):
        if match.start() < consumed or not _is_credential_name(match.group("name")):
            continue
        end = _value_end(line, match.end())
        if end > match.end():
            masked.append((match.end(), end))
            consumed = end
    return _replace_spans(line, masked)


def _mask_flag_values(line: str) -> str:
    """Mask credential flag values, and ``-p`` only inside a registry login command.

    A login command is ``docker``/``podman`` (by basename, after optional global
    options) followed by ``login``, or ``login`` starting a command segment.
    Command segments end at ``;``, ``|`` and ``&`` wherever they appear in a word.
    """

    masked: list[tuple[int, int]] = []
    consumed = 0
    state = "start"
    mask_next = False
    for token in _COMMAND_TOKEN.finditer(line):
        start = token.start()
        if start < consumed:
            continue
        text = token.group()
        separator = text[0] in ";|&"
        if mask_next:
            mask_next = False
            run = _WORD.match(line, start)
            if not (separator and run is not None and run.end() == token.end()):
                consumed = _value_end(line, start)
                masked.append((start, consumed))
                if line[consumed - 1] in ";|&":
                    state = "start"
                continue
        if separator:
            state = "start"
            continue
        name, equals, _value = text.partition("=")
        if name.lower() in _CREDENTIAL_FLAGS:
            if equals:
                consumed = _value_end(line, start + len(name) + 1)
                masked.append((start + len(name) + 1, consumed))
            else:
                mask_next = True
            continue
        if state == "login" and text.startswith("-p") and not text.startswith("--"):
            if text == "-p":
                mask_next = True
            else:
                value_start = start + (3 if text.startswith("-p=") else 2)
                consumed = _value_end(line, value_start)
                masked.append((value_start, consumed))
            continue
        if text.rsplit("/", 1)[-1].lower() in _CONTAINER_CLIS:
            state = "cli"
        elif state == "start":
            state = "login" if text == "login" else "other"
        elif state == "cli":
            if text == "login":
                state = "login"
            elif text in _CLI_VALUE_OPTIONS:
                state = "cli_value"
            elif not text.startswith("-"):
                state = "other"
        elif state == "cli_value":
            state = "cli"
    return _replace_spans(line, masked)


def _value_end(line: str, start: int) -> int:
    """End of a value starting at *start*: its whitespace run, following quotes.

    A quote anywhere in the run extends the value to its closing quote's run,
    or to the end of the line when it never closes.
    """

    run = _WORD.match(line, start)
    end = start if run is None else run.end()
    position = start
    while (quote := _QUOTE.search(line, position, end)) is not None:
        closing = line.find(quote.group(), quote.end())
        if closing == -1:
            return len(line)
        position = closing + 1
        if closing >= end:
            tail = _WORD.match(line, closing)
            assert tail is not None  # The closing quote itself is a word character.
            end = tail.end()
    return end


def _replace_spans(line: str, masked: list[tuple[int, int]]) -> str:
    if not masked:
        return line
    pieces: list[str] = []
    position = 0
    for start, end in masked:
        if start < end:
            pieces.extend((line[position:start], _REDACTED))
            position = end
    pieces.append(line[position:])
    return "".join(pieces)
