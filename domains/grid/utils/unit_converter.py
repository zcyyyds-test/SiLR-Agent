"""Unit conversion utilities: MW <-> p.u., rad <-> deg, omega <-> Hz."""

import numpy as np

from ..config import SYSTEM_BASE_MVA, SYSTEM_FREQ_HZ


def pu_to_mw(pu_val, base_mva=SYSTEM_BASE_MVA):
    """Convert per-unit power to MW (or MVAr)."""
    return pu_val * base_mva


def mw_to_pu(mw_val, base_mva=SYSTEM_BASE_MVA):
    """Convert MW (or MVAr) to per-unit power."""
    return mw_val / base_mva


def rad_to_deg(rad_val):
    """Convert radians to degrees."""
    return np.degrees(rad_val)


def deg_to_rad(deg_val):
    """Convert degrees to radians."""
    return np.radians(deg_val)


def omega_to_hz(omega_pu, f0=SYSTEM_FREQ_HZ):
    """Convert per-unit angular velocity to frequency in Hz.

    omega_pu=1.0 corresponds to synchronous speed (f0 Hz).
    """
    return omega_pu * f0


def omega_to_delta_f(omega_pu, f0=SYSTEM_FREQ_HZ):
    """Convert per-unit omega to frequency deviation in Hz."""
    return (omega_pu - 1.0) * f0
