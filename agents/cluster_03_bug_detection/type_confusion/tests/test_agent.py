"""Tests for TypeConfusionAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_03_bug_detection.type_confusion.agent import TypeConfusionAgent


@pytest.fixture
def agent() -> TypeConfusionAgent:
    return TypeConfusionAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


# ======================================================================
# Positive cases (should produce at least one finding)
# ======================================================================


class TestPositiveCases:
    def test_python_mixed_return_string_and_int(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def convert(x):\n"
            "    if x > 0:\n"
            '        return "positive"\n'
            "    return 42\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "medium"
        assert findings[0].confidence == 0.78

    def test_python_mixed_return_none_and_int_no_optional(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def fetch(x):\n"
            "    if x:\n"
            "        return 10\n"
            "    return None\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_python_mixed_return_string_none_int(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def process(x):\n"
            "    if x == 1:\n"
            '        return "ok"\n'
            "    if x == 2:\n"
            "        return 42\n"
            "    return None\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_go_type_assertion_no_comma_ok(self, agent: TypeConfusionAgent) -> None:
        src = (
            "func handle(x interface{}) {\n"
            "    val := x.(MyType)\n"
            "    fmt.Println(val)\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1
        assert "comma-ok" in findings[0].title.lower() or "comma-ok" in findings[0].description.lower()

    def test_go_type_assertion_assigned_without_ok(self, agent: TypeConfusionAgent) -> None:
        src = "s := val.(String)\n"
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1

    def test_js_loose_equality(self, agent: TypeConfusionAgent) -> None:
        src = 'if (x == "5") { doSomething(); }\n'
        findings = agent.analyze(_ctx(src, "javascript", "app.js"))
        assert len(findings) >= 1
        assert "loose equality" in findings[0].title.lower() or "==" in findings[0].title

    def test_ts_loose_equality(self, agent: TypeConfusionAgent) -> None:
        src = "if (value == null) { return; }\n"
        findings = agent.analyze(_ctx(src, "typescript", "app.ts"))
        assert len(findings) >= 1

    def test_js_typeof_string_then_numeric_op(self, agent: TypeConfusionAgent) -> None:
        src = (
            'if (typeof x === "string") {\n'
            "    result = x * 2;\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "javascript", "app.js"))
        assert len(findings) >= 1

    def test_java_cast_without_instanceof(self, agent: TypeConfusionAgent) -> None:
        src = (
            "public void process(Object obj) {\n"
            "    String s = (String) obj;\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "App.java"))
        assert len(findings) >= 1
        assert "instanceof" in findings[0].description.lower()

    def test_java_multiple_unsafe_casts(self, agent: TypeConfusionAgent) -> None:
        src = (
            "public void process(Object a, Object b) {\n"
            "    String x = (String) a;\n"
            "    Integer y = (Integer) b;\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "App.java"))
        assert len(findings) >= 2

    def test_python_bare_return_and_value(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def maybe():\n"
            "    if True:\n"
            '        return "hello"\n'
            "    return\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1


# ======================================================================
# Negative cases (should produce zero findings)
# ======================================================================


class TestNegativeCases:
    def test_python_consistent_return_types(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def greet(name):\n"
            '    if name:\n'
            '        return "Hello " + name\n'
            '    return "Hello stranger"\n'
        )
        assert agent.analyze(_ctx(src)) == []

    def test_python_none_with_optional_annotation(self, agent: TypeConfusionAgent) -> None:
        src = (
            "from typing import Optional\n"
            "def find(x) -> Optional[int]:\n"
            "    if x:\n"
            "        return 42\n"
            "    return None\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_python_single_return(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def constant():\n"
            "    return 42\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_go_type_assertion_with_comma_ok(self, agent: TypeConfusionAgent) -> None:
        src = (
            "func handle(x interface{}) {\n"
            "    val, ok := x.(MyType)\n"
            "    if ok {\n"
            "        fmt.Println(val)\n"
            "    }\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src, "go", "main.go")) == []

    def test_go_type_assertion_with_underscore(self, agent: TypeConfusionAgent) -> None:
        src = "_, _ = x.(Foo)\n"
        assert agent.analyze(_ctx(src, "go", "main.go")) == []

    def test_js_strict_equality(self, agent: TypeConfusionAgent) -> None:
        src = 'if (x === "5") { doSomething(); }\n'
        assert agent.analyze(_ctx(src, "javascript", "app.js")) == []

    def test_java_instanceof_before_cast(self, agent: TypeConfusionAgent) -> None:
        src = (
            "if (obj instanceof String) {\n"
            "    String s = (String) obj;\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src, "java", "App.java")) == []

    def test_empty_source(self, agent: TypeConfusionAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_empty_whitespace(self, agent: TypeConfusionAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_unsupported_language(self, agent: TypeConfusionAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "ruby", "test.rb")) == []

    def test_js_comment_line(self, agent: TypeConfusionAgent) -> None:
        src = "// if (x == y) {}\n"
        assert agent.analyze(_ctx(src, "javascript", "app.js")) == []

    def test_go_comment_line(self, agent: TypeConfusionAgent) -> None:
        src = "// val := x.(Type)\n"
        assert agent.analyze(_ctx(src, "go", "main.go")) == []


# ======================================================================
# Edge cases & metadata
# ======================================================================


class TestEdgeCases:
    def test_metadata(self, agent: TypeConfusionAgent) -> None:
        m = agent.metadata()
        assert m.name == "type_confusion"
        assert m.version == "0.1.0"
        assert m.axis_type == "critical"
        assert m.methodology == "bug_detection"
        assert "type-confusion" in m.tags
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0
        assert "python" in m.languages
        assert "go" in m.languages

    def test_explain(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def convert(x):\n"
            "    if x > 0:\n"
            '        return "positive"\n'
            "    return 42\n"
        )
        findings = agent.analyze(_ctx(src))
        explanation = agent.explain(findings[0])
        assert "type confusion" in explanation.lower()

    def test_finding_tags(self, agent: TypeConfusionAgent) -> None:
        src = (
            "def convert(x):\n"
            "    if x > 0:\n"
            '        return "positive"\n'
            "    return 42\n"
        )
        findings = agent.analyze(_ctx(src))
        assert "bug" in findings[0].tags
        assert "type-confusion" in findings[0].tags

    def test_finding_category(self, agent: TypeConfusionAgent) -> None:
        src = 'if (x == "5") {}\n'
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert findings[0].category == "bug"
