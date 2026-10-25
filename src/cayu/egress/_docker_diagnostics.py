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
# Private key armor lines, to find a key whose BEGIN line fell before a cut.
_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z ]{0,64}PRIVATE KEY(?: BLOCK)?-----")
_PEM_END = re.compile(r"-----END [A-Z ]{0,64}PRIVATE KEY(?: BLOCK)?-----")
_PEM_MARKER_MAX = len("-----BEGIN ") + 64 + len("PRIVATE KEY BLOCK-----")
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
# Before masking (see _normalized): drop ANSI escape sequences and turn other
# controls and zero-width characters into spaces.
_ANSI_CSI = re.compile(r"(?:\x1b\[|\x9b)[0-?]*+[ -/]*+[@-~]")
# Operating system commands (titles, hyperlinks), ended by BEL or ST, or left
# unterminated up to the end of the line. Their introducer is ``ESC ]``, so
# dropping one never eats a character of the text around it. Other escape
# sequences are not dropped: their ESC becomes a space like any control, which
# keeps a flag and its value apart.
_ANSI_OSC = re.compile(r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c\n]*+(?:\x07|\x1b\\|\x9c)?")
# The character-set selections terminal resets emit (``tput sgr0`` writes
# ``ESC ( B``). Only the ASCII and line-drawing sets are matched, and the
# sequence becomes a space, so it cannot join or shorten the words around it.
_ANSI_CHARSET = re.compile(r"\x1b[()*+][B0]")
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


def docker_stderr_tail(
    stderr: str,
    *,
    max_lines: int = 5,
    redactor: SecretRedactor | None = None,
) -> str:
    """The last non-blank stderr lines, masked and bounded like a setup failure.

    Lines are joined with `` | ``. The scan window is the end of *stderr*, and
    masking runs on the whole window before lines are picked, so a multi-line
    secret is masked as a whole. Earlier lines are dropped first to stay
    within the byte bound; ``...[truncated]`` leads when stderr was longer
    than the window.
    """
    redactor = redactor or SecretRedactor()
    if not stderr.strip():
        return "[unavailable]"
    text, cut = _masked_tail(stderr, redactor)
    lines = [line for line in text.split("\n") if line]
    kept: list[str] = []
    for line in reversed(lines[-max_lines:]):
        if kept and len(" | ".join([line, *kept]).encode()) > _MAX_STDERR_BYTES:
            break
        kept.insert(0, line)
    bounded, _ = redactor.redact_text_bounded_with_marker(
        " | ".join(kept),
        max_bytes=_MAX_STDERR_BYTES,
        truncation_marker="...[truncated]",
    )
    if cut:
        bounded = "...[truncated] " + bounded
    return bounded or _REDACTED


def _safe_stderr(stderr: str, redactor: SecretRedactor) -> str:
    if not stderr.strip():
        return "[unavailable]"
    return _bounded(" ".join(_masked_window(stderr, redactor).split()), stderr, redactor)


def _bounded(text: str, stderr: str, redactor: SecretRedactor) -> str:
    bounded, truncated = redactor.redact_text_bounded_with_marker(
        text,
        max_bytes=_MAX_STDERR_BYTES,
        truncation_marker="...[truncated]",
    )
    if len(stderr) > _MAX_SCAN_CHARACTERS and not truncated:
        bounded += "...[truncated]"
    return bounded or _REDACTED


def _masked_tail(stderr: str, redactor: SecretRedactor) -> tuple[str, bool]:
    """Redact and mask the last 64 Ki characters; also say whether text was cut.

    Registered secrets are redacted in the whole string first, so the window
    cannot cut one. The string is then normalized the way the masks see it,
    so the private key scan finds the same armor lines the PEM mask would,
    and the window is taken from that normalized text.
    """
    text = _normalized(redactor.redact_text(stderr))
    if len(text) <= _MAX_SCAN_CHARACTERS:
        return _flattened(_mask_credentials(text)), False
    start = len(text) - _MAX_SCAN_CHARACTERS
    # A private key whose BEGIN line falls before the window has no armor
    # left for the PEM mask to find, so drop the window up to its END line,
    # or all of it when the key never ends.
    key_end = _private_key_end_after(text, start)
    window = "" if key_end is None else text[max(start, key_end) :]
    # Then drop the rest of the cut line, so no half-cut credential shape stays.
    window = window[window.find("\n") + 1 :] if "\n" in window else ""
    if key_end != start:
        window = _REDACTED + "\n" + window
    return _flattened(_mask_credentials(window)), True


def _masked_window(stderr: str, redactor: SecretRedactor) -> str:
    """Redact and mask the first 64 Ki characters; lines stay, each one flattened."""
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
    return _flattened(_mask_credentials(_normalized(text)))


def _normalized(text: str) -> str:
    """Drop CSI and OSC sequences; turn character-set selections, other
    controls and zero-width characters into spaces.

    This runs before masking, so they cannot hide a flag, key or armor line
    from the masks and then vanish when the text is flattened.
    """
    text = _ANSI_CHARSET.sub(" ", _ANSI_OSC.sub("", _ANSI_CSI.sub("", text)))
    return _INVISIBLE.sub(" ", text)


def _flattened(text: str) -> str:
    return "\n".join(" ".join(_CONTROL.sub(" ", line).split()) for line in text.split("\n"))


def _private_key_end_after(stderr: str, start: int) -> int | None:
    """Where a private key open at *start* ends: *start* itself when none is open.

    Returns the end of the key's END line, or None when the key never ends.
    Both scans are single passes over *stderr*.
    """
    begin = None
    # A BEGIN line that starts before *start* may end after it.
    for match in _PEM_BEGIN.finditer(stderr, 0, start + _PEM_MARKER_MAX):
        if match.start() >= start:
            break
        begin = match
    if begin is None:
        return start
    end = _PEM_END.search(stderr, begin.end())
    if end is None:
        return None
    return max(start, end.end())


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
