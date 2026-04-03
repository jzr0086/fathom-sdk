"""Tests for LongMethodAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_07_code_quality.long_method.agent import LongMethodAgent


@pytest.fixture
def agent() -> LongMethodAgent:
    return LongMethodAgent()


def _make_context(source: str, language: str = "python", file_path: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=file_path, ast=ast)


def _make_func(name: str, body_lines: int, lang: str = "python") -> str:
    if lang == "python":
        lines = [f"def {name}():"]
        for i in range(body_lines):
            lines.append(f"    x_{i} = {i}")
        return "\n".join(lines) + "\n"
    elif lang in ("javascript", "typescript"):
        lines = [f"function {name}() {{"]
        for i in range(body_lines):
            lines.append(f"    let x_{i} = {i};")
        lines.append("}")
        return "\n".join(lines) + "\n"
    elif lang == "java":
        lines = ["public class Test {", f"    public void {name}() {{"]
        for i in range(body_lines):
            lines.append(f"        int x_{i} = {i};")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines) + "\n"
    elif lang == "go":
        lines = ["package main", "", f"func {name}() {{"]
        for i in range(body_lines):
            lines.append(f"    x_{i} := {i}")
        lines.append("}")
        return "\n".join(lines) + "\n"
    raise ValueError(f"Unsupported: {lang}")


class TestPositiveCases:
    def test_python_55_lines_high(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("big", 55))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_python_35_lines_medium(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("mid", 35))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        assert findings[0].severity == "medium"

    def test_python_22_lines_low(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("small", 22))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        assert findings[0].severity == "low"

    def test_javascript_long(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("big", 55, "javascript"), "javascript", "test.js")
        assert len(agent.analyze(ctx)) == 1

    def test_typescript_long(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("big", 55, "typescript"), "typescript", "test.ts")
        assert len(agent.analyze(ctx)) == 1

    def test_java_long(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("big", 55, "java"), "java", "Test.java")
        assert len(agent.analyze(ctx)) == 1

    def test_go_long(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("big", 55, "go"), "go", "test.go")
        assert len(agent.analyze(ctx)) == 1

    def test_multiple_long(self, agent: LongMethodAgent) -> None:
        src = _make_func("a", 55) + "\n" + _make_func("b", 35)
        ctx = _make_context(src)
        assert len(agent.analyze(ctx)) == 2

    def test_finding_fields(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("f", 55), file_path="src/mod.py")
        f = agent.analyze(ctx)[0]
        assert f.agent_name == "long_method"
        assert f.category == "quality"
        assert f.file_path == "src/mod.py"
        assert f.confidence == 0.95

    def test_exactly_51_lines(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("e", 51))
        findings = agent.analyze(ctx)
        assert len(findings) == 1


class TestNegativeCases:
    def test_5_line_function(self, agent: LongMethodAgent) -> None:
        assert agent.analyze(_make_context(_make_func("t", 4))) == []

    def test_10_line_function(self, agent: LongMethodAgent) -> None:
        assert agent.analyze(_make_context(_make_func("t", 9))) == []

    def test_19_body_lines(self, agent: LongMethodAgent) -> None:
        assert agent.analyze(_make_context(_make_func("t", 19))) == []

    def test_oneliner(self, agent: LongMethodAgent) -> None:
        assert agent.analyze(_make_context("def f():\n    return 1\n")) == []

    def test_multiple_short(self, agent: LongMethodAgent) -> None:
        src = "\n\n".join(_make_func(f"fn{i}", 5) for i in range(5))
        assert agent.analyze(_make_context(src)) == []

    def test_js_short(self, agent: LongMethodAgent) -> None:
        src = "function f() {\n    return 1;\n}\n"
        assert agent.analyze(_make_context(src, "javascript", "t.js")) == []

    def test_java_short(self, agent: LongMethodAgent) -> None:
        src = "public class T {\n    void f() {\n        int x = 1;\n    }\n}\n"
        assert agent.analyze(_make_context(src, "java", "T.java")) == []

    def test_go_short(self, agent: LongMethodAgent) -> None:
        src = "package main\n\nfunc f() {\n    x := 1\n}\n"
        assert agent.analyze(_make_context(src, "go", "t.go")) == []

    def test_no_functions(self, agent: LongMethodAgent) -> None:
        assert agent.analyze(_make_context("x = 1\ny = 2\n")) == []

    def test_class_no_long_methods(self, agent: LongMethodAgent) -> None:
        src = "class Foo:\n    def a(self):\n        pass\n    def b(self):\n        pass\n"
        assert agent.analyze(_make_context(src)) == []


class TestEdgeCases:
    def test_empty_source(self, agent: LongMethodAgent) -> None:
        assert agent.analyze(_make_context("")) == []

    def test_unsupported_language(self, agent: LongMethodAgent) -> None:
        assert agent.analyze(_make_context("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: LongMethodAgent) -> None:
        m = agent.metadata()
        assert m.name == "long_method"
        assert m.axis_type == "aware"

    def test_explain(self, agent: LongMethodAgent) -> None:
        ctx = _make_context(_make_func("f", 55))
        explanation = agent.explain(agent.analyze(ctx)[0])
        assert len(explanation) > 20
