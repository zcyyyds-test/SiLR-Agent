"""Core abstractions for the SiLR framework.

This module is a leaf node in the dependency graph — it imports nothing
from other silr subpackages, preventing circular dependencies.
"""

from .interfaces import BaseSystemManager, BaseConstraintChecker
from .config import DomainConfig

__all__ = ["BaseSystemManager", "BaseConstraintChecker", "DomainConfig"]
