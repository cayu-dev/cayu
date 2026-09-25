"""Non-disclosing denial shared by independently authorized resources."""


class ResourceAccessDenied(PermissionError):
    def __init__(self) -> None:
        super().__init__("Resource operation is not authorized.")
