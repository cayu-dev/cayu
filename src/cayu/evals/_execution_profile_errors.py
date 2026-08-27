"""Internal control-flow errors for exact Evals execution-profile enforcement."""


class EvalExecutionProfileChangedError(RuntimeError):
    """The server-owned execution identity changed after eval admission."""
