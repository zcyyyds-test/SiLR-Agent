"""Exception hierarchy for SiLR framework."""


class SiLRError(Exception):
    """Base exception for all SiLR errors."""
    pass


class SystemNotLoadedError(SiLRError):
    """Raised when operation requires a loaded system."""
    pass


class SystemStateError(SiLRError):
    """Raised when operation is invalid for current system state."""
    pass


class DeviceNotFoundError(SiLRError):
    """Raised when a device ID is not found."""
    pass


class ConvergenceError(SiLRError):
    """Raised when steady-state solver fails to converge."""
    pass


class ValidationError(SiLRError):
    """Raised when parameter validation fails."""
    pass


class SnapshotError(SiLRError):
    """Raised on snapshot save/restore failure."""
    pass
