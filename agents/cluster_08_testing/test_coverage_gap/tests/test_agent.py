"""Tests for TestCoverageGapAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_08_testing.test_coverage_gap.agent import TestCoverageGapAgent


@pytest.fixture
def agent() -> TestCoverageGapAgent:
    return TestCoverageGapAgent()


def _ctx(source: str, language: str = "python", fp: str = "module.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ---------------------------------------------------------------------------
# Helper to build a function with N decision points (if-statements)
# ---------------------------------------------------------------------------


def _complex_func(name: str = "process_data", n: int = 6) -> str:
    """Return Python source for a function with *n* if-statements."""
    lines = [f"def {name}(x):"]
    for i in range(n):
        lines.append(f"    if x > {i}:")
        lines.append(f"        x += {i}")
    lines.append("    return x")
    return "\n".join(lines) + "\n"


def _complex_func_js(name: str = "processData", n: int = 6) -> str:
    body = "".join(f"    if (x > {i}) {{ x += {i}; }}\n" for i in range(n))
    return f"function {name}(x) {{\n{body}    return x;\n}}\n"


def _complex_func_java(name: str = "processData", n: int = 6) -> str:
    body = "".join(f"        if (x > {i}) {{ x += {i}; }}\n" for i in range(n))
    return (
        f"public class Svc {{\n"
        f"    int {name}(int x) {{\n{body}        return x;\n    }}\n"
        f"}}\n"
    )


def _complex_func_go(name: str = "processData", n: int = 6) -> str:
    body = "".join(f"    if x > {i} {{ x += {i} }}\n" for i in range(n))
    return f"package main\n\nfunc {name}(x int) int {{\n{body}    return x\n}}\n"


# ── Positive cases (should produce findings) ─────────────────────────────


class TestPositiveCases:
    def test_complex_function_no_tests(self, agent: TestCoverageGapAgent) -> None:
        """A function with 6 decision points and no tests should be flagged."""
        src = _complex_func("process_data", 6)
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        assert "process_data" in findings[0].title
        assert findings[0].severity == "low"
        assert findings[0].confidence == 0.68

    def test_high_complexity_flagged(self, agent: TestCoverageGapAgent) -> None:
        """A function with 10 decision points should be flagged."""
        src = _complex_func("handle_request", 10)
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        assert "10 decision points" in findings[0].title

    def test_multiple_complex_functions(self, agent: TestCoverageGapAgent) -> None:
        """Multiple complex functions with no tests should all be flagged."""
        src = _complex_func("func_a", 7) + "\n" + _complex_func("func_b", 8)
        findings = agent.analyze(_ctx(src))
        names = {f.title for f in findings}
        assert len(findings) == 2
        assert any("func_a" in t for t in names)
        assert any("func_b" in t for t in names)

    def test_mixed_for_while_try(self, agent: TestCoverageGapAgent) -> None:
        """Decision points from for, while, and try should all count."""
        src = (
            "def complex(x):\n"
            "    if x > 0:\n"
            "        pass\n"
            "    for i in range(x):\n"
            "        pass\n"
            "    while x > 10:\n"
            "        x -= 1\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n"
            "    if x < 0:\n"
            "        pass\n"
            "    if x == 5:\n"
            "        pass\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        assert "complex" in findings[0].title

    def test_javascript_complex_no_tests(self, agent: TestCoverageGapAgent) -> None:
        """A complex JS function with no tests should be flagged."""
        src = _complex_func_js("processData", 7)
        findings = agent.analyze(_ctx(src, "javascript", "service.js"))
        assert len(findings) == 1
        assert "processData" in findings[0].title

    def test_java_complex_no_tests(self, agent: TestCoverageGapAgent) -> None:
        """A complex Java method with no tests should be flagged."""
        src = _complex_func_java("processData", 7)
        findings = agent.analyze(_ctx(src, "java", "Service.java"))
        assert len(findings) == 1
        assert "processData" in findings[0].title

    def test_go_complex_no_tests(self, agent: TestCoverageGapAgent) -> None:
        """A complex Go function with no tests should be flagged."""
        src = _complex_func_go("processData", 7)
        findings = agent.analyze(_ctx(src, "go", "service.go"))
        assert len(findings) == 1
        assert "processData" in findings[0].title

    def test_finding_fields(self, agent: TestCoverageGapAgent) -> None:
        """Verify all expected fields on a finding."""
        src = _complex_func("validate", 8)
        findings = agent.analyze(_ctx(src, fp="app/validate.py"))
        assert len(findings) == 1
        f = findings[0]
        assert f.agent_name == "test_coverage_gap"
        assert f.category == "testing"
        assert f.file_path == "app/validate.py"
        assert f.confidence == 0.68
        assert f.severity == "low"
        assert f.line_start >= 1
        assert f.line_end >= f.line_start
        assert "testing" in f.tags

    def test_typescript_complex_flagged(self, agent: TestCoverageGapAgent) -> None:
        """A complex TypeScript function should be flagged."""
        body = "".join(f"    if (x > {i}) {{ x += {i}; }}\n" for i in range(7))
        src = f"function handleEvent(x: number) {{\n{body}    return x;\n}}\n"
        findings = agent.analyze(_ctx(src, "typescript", "handler.ts"))
        assert len(findings) == 1

    def test_complex_with_unrelated_test(self, agent: TestCoverageGapAgent) -> None:
        """A complex function whose name does NOT appear in any test function."""
        src = (
            _complex_func("compute_score", 7)
            + "\n"
            + "def test_something_else():\n    assert 1 == 1\n"
        )
        findings = agent.analyze(_ctx(src, fp="module.py"))
        assert len(findings) == 1
        assert "compute_score" in findings[0].title

    def test_loops_contribute_to_complexity(self, agent: TestCoverageGapAgent) -> None:
        """for/while loops should each count as a decision point."""
        src = (
            "def pipeline(data):\n"
            "    for item in data:\n"
            "        pass\n"
            "    for item in data:\n"
            "        pass\n"
            "    while len(data) > 0:\n"
            "        data.pop()\n"
            "    for item in data:\n"
            "        pass\n"
            "    for item in data:\n"
            "        pass\n"
            "    for item in data:\n"
            "        pass\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        assert "pipeline" in findings[0].title


# ── Negative cases (should NOT produce findings) ─────────────────────────


class TestNegativeCases:
    def test_simple_function(self, agent: TestCoverageGapAgent) -> None:
        """A function with few decision points should not be flagged."""
        src = "def add(a, b):\n    return a + b\n"
        assert agent.analyze(_ctx(src)) == []

    def test_test_file_skipped(self, agent: TestCoverageGapAgent) -> None:
        """Files whose path contains 'test' should be skipped entirely."""
        src = _complex_func("process_data", 8)
        assert agent.analyze(_ctx(src, fp="tests/test_service.py")) == []

    def test_test_file_path_case_insensitive(self, agent: TestCoverageGapAgent) -> None:
        """Test file detection should be case-insensitive."""
        src = _complex_func("process_data", 8)
        assert agent.analyze(_ctx(src, fp="Tests/TestService.py")) == []

    def test_empty_source(self, agent: TestCoverageGapAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: TestCoverageGapAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_no_functions(self, agent: TestCoverageGapAgent) -> None:
        """Source with no functions should produce no findings."""
        src = "x = 1\ny = 2\nz = x + y\n"
        assert agent.analyze(_ctx(src)) == []

    def test_below_threshold(self, agent: TestCoverageGapAgent) -> None:
        """A function with exactly 5 decision points should not be flagged."""
        src = _complex_func("process", 5)
        assert agent.analyze(_ctx(src)) == []

    def test_complex_with_matching_test(self, agent: TestCoverageGapAgent) -> None:
        """A complex function with a corresponding test function should not be flagged."""
        src = (
            _complex_func("calculate_tax", 7)
            + "\n"
            + "def test_calculate_tax():\n    assert calculate_tax(100) == 110\n"
        )
        assert agent.analyze(_ctx(src, fp="module.py")) == []

    def test_unsupported_language(self, agent: TestCoverageGapAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "main.cob")) == []

    def test_single_if(self, agent: TestCoverageGapAgent) -> None:
        src = "def f():\n    if x:\n        pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_test_function_not_flagged(self, agent: TestCoverageGapAgent) -> None:
        """Test functions themselves should never be flagged, even if complex."""
        lines = ["def test_everything(x):"]
        for i in range(8):
            lines.append(f"    if x > {i}:")
            lines.append(f"        assert x != {i}")
        lines.append("    return x")
        src = "\n".join(lines) + "\n"
        assert agent.analyze(_ctx(src, fp="module.py")) == []

    def test_simple_js_function(self, agent: TestCoverageGapAgent) -> None:
        src = "function add(a, b) { return a + b; }\n"
        assert agent.analyze(_ctx(src, "javascript", "util.js")) == []


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_metadata(self, agent: TestCoverageGapAgent) -> None:
        m = agent.metadata()
        assert m.name == "test_coverage_gap"
        assert m.version == "0.1.0"
        assert m.axis_type == "aware"
        assert m.methodology == "testing"
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0
        assert "python" in m.languages
        assert "go" in m.languages
        assert "quality_engineering" in m.domains

    def test_explain(self, agent: TestCoverageGapAgent) -> None:
        src = _complex_func("validate_input", 7)
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        explanation = agent.explain(findings[0])
        assert len(explanation) > 20
        assert "validate_input" in explanation

    def test_file_with_only_test_functions(self, agent: TestCoverageGapAgent) -> None:
        """A file with only test functions and no production code (but not a test path)."""
        src = "def test_a():\n    assert True\n\ndef test_b():\n    assert False\n"
        assert agent.analyze(_ctx(src, fp="module.py")) == []

    def test_threshold_boundary(self, agent: TestCoverageGapAgent) -> None:
        """Exactly at threshold (5) should not flag; one above (6) should."""
        at_threshold = _complex_func("at_five", 5)
        above_threshold = _complex_func("at_six", 6)
        assert agent.analyze(_ctx(at_threshold)) == []
        findings = agent.analyze(_ctx(above_threshold))
        assert len(findings) == 1
        assert "at_six" in findings[0].title
