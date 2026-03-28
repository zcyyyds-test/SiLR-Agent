"""Parameter validation utilities."""

from silr.exceptions import ValidationError, DeviceNotFoundError


def validate_positive(value, name):
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value, name):
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def clamp(value, min_val, max_val):
    """Clamp value to [min_val, max_val].

    Returns (clamped_value, was_clamped).
    """
    clamped = max(min_val, min(max_val, value))
    return clamped, clamped != value


def validate_device_idx(model, idx, device_type="device"):
    """Check that idx exists in the ANDES model's idx list."""
    idx_list = list(model.idx.v)
    if idx not in idx_list:
        raise DeviceNotFoundError(
            f"{device_type} '{idx}' not found. Available: {idx_list}"
        )
