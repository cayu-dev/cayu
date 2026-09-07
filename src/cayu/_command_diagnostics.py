"""Closed public corrections for structured commands; never format caller input."""

from enum import StrEnum
from types import MappingProxyType


class CommandDenialCode(StrEnum):
    UNKNOWN_SELECTOR = "unknown_selector"
    ARGUMENT_SHAPE = "argument_shape"
    ARGUMENT_COUNT = "argument_count"
    DISALLOWED_PATH = "disallowed_path"
    WORKING_DIRECTORY = "working_directory"
    TIMEOUT_CEILING = "timeout_ceiling"
    OUTPUT_MODE = "output_mode"


COMMAND_DENIAL_HINTS = MappingProxyType(
    {
        CommandDenialCode.UNKNOWN_SELECTOR: "Choose a selector admitted by the active command profile.",
        CommandDenialCode.ARGUMENT_SHAPE: "Use the selector's declared fields, flags, and argument forms.",
        CommandDenialCode.ARGUMENT_COUNT: "Use the selector's declared minimum and maximum argument count.",
        CommandDenialCode.DISALLOWED_PATH: "Use a workspace-relative path within the selector's admitted path scope.",
        CommandDenialCode.WORKING_DIRECTORY: "Omit workingDirectory or choose an admitted working directory.",
        CommandDenialCode.TIMEOUT_CEILING: "Omit timeoutSeconds or use a positive integer within the selector's timeout ceiling.",
        CommandDenialCode.OUTPUT_MODE: "Use outputMode summary or summary_and_artifact.",
    }
)


class CommandValidationError(ValueError):
    def __init__(self, code: CommandDenialCode):
        self.code = code
        super().__init__(COMMAND_DENIAL_HINTS[code])
