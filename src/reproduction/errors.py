"""D-class pipeline errors."""


class ReproductionError(RuntimeError):
    """Base error for the reproduction pipeline."""


class SourceGenerationError(ReproductionError):
    """The gateway did not return a usable implementation."""

    def __init__(self, message: str, *, upstream_status: int | None = None,
                 retryable: bool | None = None) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status
        self.retryable = retryable


class SelfCheckError(ReproductionError):
    """The public sample self-check failed."""


class PlatformError(ReproductionError):
    """The platform experiment client failed or returned an invalid answer."""


class PlatformUnavailableError(PlatformError):
    """Trusted execution/audit is unavailable; not an algorithm failure."""


class PlatformExecutionUnknownError(PlatformError):
    """A submitted experiment has no confirmed current result yet."""
