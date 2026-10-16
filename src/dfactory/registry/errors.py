"""Registry-specific errors."""


class DuplicateProviderError(ValueError):
    """Raised when two providers claim the same public name."""


class ProviderNotFoundError(KeyError):
    """Raised when a requested provider is not installed."""


__all__ = ["DuplicateProviderError", "ProviderNotFoundError"]
