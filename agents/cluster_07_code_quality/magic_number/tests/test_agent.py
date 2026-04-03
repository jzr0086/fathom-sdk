"""Tests for MagicNumberAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_07_code_quality.magic_number.agent import MagicNumberAgent


@pytest.fixture
def agent() -> MagicNumberAgent:
    return MagicNumberAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ── Positive tests (magic numbers SHOULD be flagged) ───────────────────────


class TestPositiveCases:
    def test_multiply_by_magic(self, agent: MagicNumberAgent) -> None:
        """x = y * 86400  — 86400 is unexplained."""
        src = "def f():\n    x = y * 86400\n"
        findings = agent.analyze(_ctx(src))
        assert any(f.title and "86400" in f.title for f in findings)

    def test_comparison_magic(self, agent: MagicNumberAgent) -> None:
        """if count > 42: — 42 is unexplained."""
        src = "def f():\n    if count > 42:\n        pass\n"
        findings = agent.analyze(_ctx(src))
        assert any("42" in f.title for f in findings)

    def test_call_arg_magic(self, agent: MagicNumberAgent) -> None:
        """sleep(300) — 300 is a magic number."""
        src = "def f():\n    sleep(300)\n"
        findings = agent.analyze(_ctx(src))
        assert any("300" in f.title for f in findings)

    def test_return_magic(self, agent: MagicNumberAgent) -> None:
        """return 3600 — unexplained value."""
        src = "def f():\n    return 3600\n"
        findings = agent.analyze(_ctx(src))
        assert any("3600" in f.title for f in findings)

    def test_addition_magic(self, agent: MagicNumberAgent) -> None:
        """x = y + 255"""
        src = "def f():\n    x = y + 255\n"
        findings = agent.analyze(_ctx(src))
        assert any("255" in f.title for f in findings)

    def test_float_magic(self, agent: MagicNumberAgent) -> None:
        """ratio = value * 3.14 — a magic float."""
        src = "def f():\n    ratio = value * 3.14\n"
        findings = agent.analyze(_ctx(src))
        assert any("3.14" in f.title for f in findings)

    def test_javascript_magic(self, agent: MagicNumberAgent) -> None:
        """JS: const x = y * 86400;"""
        src = "function f() {\n    const x = y * 86400;\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert any("86400" in f.title for f in findings)

    def test_java_magic(self, agent: MagicNumberAgent) -> None:
        """Java: int x = y * 86400;"""
        src = "public class T {\n    void f() {\n        int x = y * 86400;\n    }\n}\n"
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert any("86400" in f.title for f in findings)

    def test_go_magic(self, agent: MagicNumberAgent) -> None:
        """Go: x := y * 86400"""
        src = "package main\n\nfunc f() {\n    x := y * 86400\n}\n"
        findings = agent.analyze(_ctx(src, "go", "t.go"))
        assert any("86400" in f.title for f in findings)

    def test_multiple_magic_numbers(self, agent: MagicNumberAgent) -> None:
        """Two distinct magic numbers produce two findings."""
        src = "def f():\n    a = 86400\n    b = 3600\n"
        findings = agent.analyze(_ctx(src))
        magic_vals = {f.title for f in findings}
        assert any("86400" in t for t in magic_vals)
        assert any("3600" in t for t in magic_vals)

    def test_lowercase_assignment_not_constant(self, agent: MagicNumberAgent) -> None:
        """timeout = 300 — lowercase name, not a constant."""
        src = "def f():\n    timeout = 300\n"
        findings = agent.analyze(_ctx(src))
        assert any("300" in f.title for f in findings)

    def test_finding_fields(self, agent: MagicNumberAgent) -> None:
        """Verify all Finding fields are populated correctly."""
        src = "def f():\n    x = 42\n"
        findings = agent.analyze(_ctx(src, fp="src/mod.py"))
        flagged = [f for f in findings if "42" in f.title]
        assert len(flagged) == 1
        f = flagged[0]
        assert f.agent_name == "magic_number"
        assert f.severity == "low"
        assert f.category == "quality"
        assert f.file_path == "src/mod.py"
        assert f.confidence == 0.75


# ── Negative tests (should NOT be flagged) ────────────────────────────────


class TestNegativeCases:
    def test_zero_not_magic(self, agent: MagicNumberAgent) -> None:
        """0 is a universally understood sentinel value."""
        src = "def f():\n    x = 0\n"
        findings = agent.analyze(_ctx(src))
        assert all("0" != f.title.split()[-1] for f in findings)
        # More precise: no finding whose literal text is exactly "0"
        assert not any(
            f.title == "Magic number 0 at line 2" for f in findings
        )

    def test_one_not_magic(self, agent: MagicNumberAgent) -> None:
        """1 is a universally understood base/increment value."""
        src = "def f():\n    x = 1\n"
        assert agent.analyze(_ctx(src)) == []

    def test_negative_one_not_magic(self, agent: MagicNumberAgent) -> None:
        """-1 is a common sentinel value."""
        src = "def f():\n    x = -1\n"
        findings = agent.analyze(_ctx(src))
        # Should not flag the '1' inside '-1'
        assert not any("Magic number 1 " in f.title for f in findings)

    def test_upper_case_constant(self, agent: MagicNumberAgent) -> None:
        """MAX_SIZE = 100 — named constant is fine."""
        src = "MAX_SIZE = 100\n"
        assert agent.analyze(_ctx(src)) == []

    def test_upper_case_constant_with_underscore(self, agent: MagicNumberAgent) -> None:
        """TIMEOUT_SECONDS = 300 — named constant is fine."""
        src = "TIMEOUT_SECONDS = 300\n"
        assert agent.analyze(_ctx(src)) == []

    def test_js_const_upper_case(self, agent: MagicNumberAgent) -> None:
        """JS: const MAX_RETRIES = 5; — named constant is fine."""
        src = "const MAX_RETRIES = 5;\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert not any("5" in f.title for f in findings)

    def test_empty_source(self, agent: MagicNumberAgent) -> None:
        """Empty input produces no findings."""
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: MagicNumberAgent) -> None:
        """Whitespace-only input produces no findings."""
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_no_numeric_literals(self, agent: MagicNumberAgent) -> None:
        """Code with no numbers produces no findings."""
        src = "def f():\n    return 'hello'\n"
        assert agent.analyze(_ctx(src)) == []

    def test_unsupported_language(self, agent: MagicNumberAgent) -> None:
        """Unsupported language returns empty results."""
        assert agent.analyze(_ctx("x = 42", "cobol", "t.cob")) == []

    def test_only_zeros_and_ones(self, agent: MagicNumberAgent) -> None:
        """Code using only 0 and 1 produces no findings."""
        src = "def f():\n    x = 0\n    y = 1\n    z = x + 1\n"
        assert agent.analyze(_ctx(src)) == []


# ── Edge cases ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_metadata(self, agent: MagicNumberAgent) -> None:
        m = agent.metadata()
        assert m.name == "magic_number"
        assert m.version == "0.1.0"
        assert m.axis_type == "aware"
        assert "python" in m.languages
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain(self, agent: MagicNumberAgent) -> None:
        src = "def f():\n    x = 42\n"
        findings = agent.analyze(_ctx(src))
        flagged = [f for f in findings if "42" in f.title]
        assert len(flagged) == 1
        explanation = agent.explain(flagged[0])
        assert len(explanation) > 20
        assert "magic" in explanation.lower() or "Magic" in explanation

    def test_typescript_magic(self, agent: MagicNumberAgent) -> None:
        """TypeScript is supported."""
        src = "function f(): void {\n    const x = 999;\n}\n"
        findings = agent.analyze(_ctx(src, "typescript", "t.ts"))
        assert any("999" in f.title for f in findings)

    def test_mixed_constants_and_magic(self, agent: MagicNumberAgent) -> None:
        """Constants are excluded, but magic numbers in the same file are caught."""
        src = "MAX_SIZE = 100\ndef f():\n    x = 42\n"
        findings = agent.analyze(_ctx(src))
        # 100 is in a constant — excluded
        assert not any("100" in f.title for f in findings)
        # 42 is magic — flagged
        assert any("42" in f.title for f in findings)
