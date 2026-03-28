"""Test core interfaces: ABC contracts and zero circular imports."""

import sys


class TestZeroCircularImports:
    """Verify that silr.core has no internal cross-imports."""

    def test_import_core_interfaces(self):
        from silr.core.interfaces import BaseSystemManager, BaseConstraintChecker
        assert BaseSystemManager is not None
        assert BaseConstraintChecker is not None

    def test_import_core_config(self):
        from silr.core.config import DomainConfig
        assert DomainConfig is not None

    def test_import_types(self):
        from silr.types import ToolResult, ViolationFlag
        assert ToolResult is not None
        assert ViolationFlag is not None

    def test_import_exceptions(self):
        from silr.exceptions import SiLRError, ValidationError
        assert issubclass(ValidationError, SiLRError)

    def test_import_verifier(self):
        from silr.verifier import SiLRVerifier, Verdict
        assert SiLRVerifier is not None
        assert Verdict is not None

    def test_import_agent(self):
        from silr.agent import ReActAgent, AgentConfig
        assert ReActAgent is not None
        assert AgentConfig is not None

    def test_import_tools_base(self):
        from silr.tools.base import BaseTool
        assert BaseTool is not None

    def test_core_no_silr_internal_imports(self):
        """silr.core modules should NOT import from other silr subpackages."""
        # Snapshot current modules
        before = set(sys.modules.keys())

        # Fresh import (remove cached core modules first)
        for mod_name in list(sys.modules):
            if mod_name.startswith("silr.core"):
                del sys.modules[mod_name]

        import silr.core.interfaces  # noqa: F811
        import silr.core.config      # noqa: F811

        # Only check NEW modules that were pulled in by these imports
        after = set(sys.modules.keys())
        new_modules = after - before

        # Filter to silr.* modules that aren't core/types/exceptions
        unexpected = [
            m for m in new_modules
            if m.startswith("silr.")
            and not m.startswith("silr.core")
            and not m.startswith("silr.types")
            and not m.startswith("silr.exceptions")
            and m != "silr"
        ]
        assert unexpected == [], f"silr.core pulled in unexpected modules: {unexpected}"


class TestABCContracts:
    """Verify ABC methods are abstract."""

    def test_base_system_manager_is_abstract(self):
        from silr.core.interfaces import BaseSystemManager
        import pytest

        with pytest.raises(TypeError):
            BaseSystemManager()

    def test_base_constraint_checker_is_abstract(self):
        from silr.core.interfaces import BaseConstraintChecker
        import pytest

        with pytest.raises(TypeError):
            BaseConstraintChecker()
