"""Tests for AssertionQualityAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_08_testing.assertion_quality.agent import AssertionQualityAgent


@pytest.fixture
def agent() -> AssertionQualityAgent:
    return AssertionQualityAgent()


def _ctx(source: str, language: str = "python", fp: str = "test_example.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ── Positive cases (should produce findings) ─────────────────────────────


class TestPositiveCases:
    def test_assert_true_with_equality(self, agent: AssertionQualityAgent) -> None:
        src = "def test_foo():\n    assertTrue(x == y)\n"
        findings = agent.analyze(_ctx(src))
        matched = [f for f in findings if "assertTrue with comparison" in f.title]
        assert len(matched) >= 1

    def test_assert_true_with_inequality(self, agent: AssertionQualityAgent) -> None:
        src = "def test_foo():\n    assertTrue(x != y)\n"
        findings = agent.analyze(_ctx(src))
        matched = [f for f in findings if "assertTrue with comparison" in f.title]
        assert len(matched) >= 1

    def test_assert_true_equality_severity(self, agent: AssertionQualityAgent) -> None:
        src = "def test_foo():\n    assertTrue(a == b)\n"
        findings = agent.analyze(_ctx(src))
        comparison_findings = [f for f in findings if "assertTrue with comparison" in f.title]
        assert comparison_findings[0].severity == "low"

    def test_no_assertion_python(self, agent: AssertionQualityAgent) -> None:
        src = "def test_compute():\n    result = compute()\n    print(result)\n"
        findings = agent.analyze(_ctx(src))
        matched = [f for f in findings if "no assertions" in f.title]
        assert len(matched) == 1
        assert matched[0].severity == "medium"

    def test_no_assertion_function_name(self, agent: AssertionQualityAgent) -> None:
        src = "def test_add():\n    x = 1 + 2\n"
        findings = agent.analyze(_ctx(src))
        matched = [f for f in findings if "no assertions" in f.title]
        assert len(matched) == 1
        assert "test_add" in matched[0].title

    def test_no_assertion_go_test_prefix(self, agent: AssertionQualityAgent) -> None:
        src = "package main\n\nfunc TestAdd() {\n    x := 1 + 2\n}\n"
        findings = agent.analyze(_ctx(src, "go", "example_test.go"))
        matched = [f for f in findings if "no assertions" in f.title]
        assert len(matched) == 1

    def test_meaningless_assert_true(self, agent: AssertionQualityAgent) -> None:
        src = "def test_placeholder():\n    assert True\n"
        findings = agent.analyze(_ctx(src))
        matched = [f for f in findings if "Meaningless assertion" in f.title]
        assert len(matched) >= 1

    def test_meaningless_assert_true_call(self, agent: AssertionQualityAgent) -> None:
        src = "def test_placeholder():\n    assertTrue(True)\n"
        findings = agent.analyze(_ctx(src))
        matched = [f for f in findings if "Meaningless assertion" in f.title]
        assert len(matched) >= 1

    def test_meaningless_severity(self, agent: AssertionQualityAgent) -> None:
        src = "def test_placeholder():\n    assertTrue(True)\n"
        findings = agent.analyze(_ctx(src))
        meaningless = [f for f in findings if "Meaningless assertion" in f.title]
        assert meaningless[0].severity == "low"

    def test_multiple_anti_patterns(self, agent: AssertionQualityAgent) -> None:
        src = (
            "def test_a():\n"
            "    assertTrue(x == 1)\n"
            "\n"
            "def test_b():\n"
            "    result = compute()\n"
            "\n"
            "def test_c():\n"
            "    assert True\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 3

    def test_finding_fields(self, agent: AssertionQualityAgent) -> None:
        src = "def test_x():\n    assertTrue(a == b)\n"
        findings = agent.analyze(_ctx(src, fp="tests/test_math.py"))
        comparison = [f for f in findings if "assertTrue with comparison" in f.title]
        assert comparison[0].agent_name == "assertion_quality"
        assert comparison[0].category == "testing"
        assert comparison[0].file_path == "tests/test_math.py"
        assert comparison[0].confidence == 0.85

    def test_js_test_no_assertion(self, agent: AssertionQualityAgent) -> None:
        src = "function test_render() {\n    let x = render();\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        matched = [f for f in findings if "no assertions" in f.title]
        assert len(matched) == 1


# ── Negative cases (should NOT produce findings) ─────────────────────────


class TestNegativeCases:
    def test_proper_assert_equal(self, agent: AssertionQualityAgent) -> None:
        src = "def test_add():\n    assertEqual(add(1, 2), 3)\n"
        findings = agent.analyze(_ctx(src))
        comparison = [f for f in findings if "assertTrue with comparison" in f.title]
        assert comparison == []

    def test_proper_assert_statement(self, agent: AssertionQualityAgent) -> None:
        src = "def test_add():\n    assert add(1, 2) == 3\n"
        findings = agent.analyze(_ctx(src))
        no_assert = [f for f in findings if "no assertions" in f.title]
        assert no_assert == []

    def test_test_with_assert_in(self, agent: AssertionQualityAgent) -> None:
        src = "def test_membership():\n    assertIn(item, collection)\n"
        findings = agent.analyze(_ctx(src))
        no_assert = [f for f in findings if "no assertions" in f.title]
        assert no_assert == []

    def test_non_test_function_no_assert(self, agent: AssertionQualityAgent) -> None:
        src = "def compute():\n    return 1 + 2\n"
        findings = agent.analyze(_ctx(src))
        no_assert = [f for f in findings if "no assertions" in f.title]
        assert no_assert == []

    def test_non_test_function_helper(self, agent: AssertionQualityAgent) -> None:
        src = "def helper():\n    x = 1\n    return x\n"
        findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_empty_source(self, agent: AssertionQualityAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_no_functions(self, agent: AssertionQualityAgent) -> None:
        src = "x = 1\ny = 2\n"
        assert agent.analyze(_ctx(src)) == []

    def test_test_with_expect_js(self, agent: AssertionQualityAgent) -> None:
        src = "function test_render() {\n    expect(result).toBe(true);\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        no_assert = [f for f in findings if "no assertions" in f.title]
        assert no_assert == []

    def test_test_with_assert_raises(self, agent: AssertionQualityAgent) -> None:
        src = "def test_raises():\n    assertRaises(ValueError, bad_func)\n"
        findings = agent.analyze(_ctx(src))
        no_assert = [f for f in findings if "no assertions" in f.title]
        assert no_assert == []

    def test_assert_true_without_comparison(self, agent: AssertionQualityAgent) -> None:
        src = "def test_flag():\n    assertTrue(is_valid)\n"
        findings = agent.analyze(_ctx(src))
        comparison = [f for f in findings if "assertTrue with comparison" in f.title]
        assert comparison == []

    def test_assert_false_value(self, agent: AssertionQualityAgent) -> None:
        src = "def test_something():\n    assert result == False\n"
        findings = agent.analyze(_ctx(src))
        meaningless = [f for f in findings if "Meaningless assertion" in f.title]
        assert meaningless == []

    def test_java_test_with_assertion(self, agent: AssertionQualityAgent) -> None:
        src = (
            "public class MyTest {\n"
            "    public void test_add() {\n"
            "        assertEquals(3, add(1, 2));\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "MyTest.java"))
        no_assert = [f for f in findings if "no assertions" in f.title]
        assert no_assert == []


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unsupported_language(self, agent: AssertionQualityAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: AssertionQualityAgent) -> None:
        m = agent.metadata()
        assert m.name == "assertion_quality"
        assert m.version == "0.1.0"
        assert m.axis_type == "aware"
        assert m.methodology == "testing"
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain(self, agent: AssertionQualityAgent) -> None:
        src = "def test_x():\n    assertTrue(a == b)\n"
        findings = agent.analyze(_ctx(src))
        comparison = [f for f in findings if "assertTrue with comparison" in f.title]
        explanation = agent.explain(comparison[0])
        assert len(explanation) > 20

    def test_whitespace_only_source(self, agent: AssertionQualityAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []
