"""Report generator: structured VerificationResult -> human/LLM-readable text."""

from .types import VerificationResult, Verdict


class ReportGenerator:
    """Generate natural-language verification reports for LLM consumption."""

    def generate(self, result: VerificationResult) -> str:
        lines = []
        lines.append("=== SiLR Verification Report ===")

        # Action
        action = result.action
        tool_name = action.get("tool_name", "unknown")
        params = action.get("params", {})
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        lines.append(f"Action: {tool_name}({param_str})")
        lines.append(f"Verdict: {result.verdict.value}")
        lines.append("")

        # Failure reason
        if result.fail_reason:
            lines.append(f"Failure Reason: {result.fail_reason}")
            lines.append("")

        # Steady-state solver convergence
        if result.solver_converged is not None:
            status = "converged" if result.solver_converged else "FAILED"
            lines.append(f"Steady-State Solver: {status}")

        # Post-solve hook status
        if result.post_solve_passed is None:
            lines.append("Post-Solve Check: skipped")
        elif result.post_solve_passed:
            lines.append("Post-Solve Check: passed")
        else:
            lines.append("Post-Solve Check: FAILED")
        lines.append("")

        # Constraint results
        if result.check_results:
            lines.append("Constraint Results:")
            for cr in result.check_results:
                tag = "PASS" if cr.passed else "FAIL"
                summary_str = self._format_summary(cr)
                lines.append(f"  [{tag}] {cr.checker_name.title()}: {summary_str}")

                for v in cr.violations:
                    lines.append(f"    - {v.detail} ({v.severity})")

            lines.append("")

        # Timing
        lines.append(f"Verification completed in {result.elapsed_seconds:.2f}s")

        return "\n".join(lines)

    def _format_summary(self, cr) -> str:
        """Format checker summary. Falls back to str(summary) for unknown checkers."""
        s = cr.summary
        name = cr.checker_name

        # Power grid checkers (recognized by name)
        if name == "voltage":
            if cr.passed:
                return (
                    f"all buses within [{s.get('min_pu', '?'):.2f}, "
                    f"{s.get('max_pu', '?'):.2f}] p.u."
                )
            return f"{s.get('n_violations', 0)} violations"

        elif name == "frequency":
            if cr.passed:
                return (
                    f"max |delta_f| = {s.get('max_abs_delta_f_hz', 0):.3f} Hz "
                    f"(within limit)"
                )
            return f"{s.get('n_violations', 0)} violations"

        elif name == "line_loading":
            n_rated = s.get("n_rated_lines", 0)
            if n_rated == 0:
                return "no rated lines (rate_a data unavailable)"
            if cr.passed:
                max_l = s.get("max_loading_pct", 0)
                return f"max loading = {max_l:.1f}% (within limit)"
            return f"{s.get('n_violations', 0)} overloaded lines"

        elif name == "transient":
            method = s.get("method", "unknown")
            if method == "time_series":
                peak = s.get("peak_separation_deg", 0)
                t = s.get("peak_time", 0)
                if cr.passed:
                    return f"peak separation {peak:.1f} deg at t={t:.2f}s (stable)"
                return f"peak separation {peak:.1f} deg > {s.get('limit_deg', 180)} deg at t={t:.2f}s"
            else:
                sep = s.get("max_separation_deg", 0)
                if cr.passed:
                    return f"final separation {sep:.1f} deg (stable)"
                return f"final separation {sep:.1f} deg > {s.get('limit_deg', 180)} deg"

        # Generic fallback: show pass/fail count
        n_violations = s.get("n_violations", None)
        if n_violations is not None:
            if cr.passed:
                return "all checks passed"
            return f"{n_violations} violations"

        return str(s)
