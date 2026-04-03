"""Tests for CyclomaticComplexityAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_07_code_quality.cyclomatic_complexity.agent import CyclomaticComplexityAgent


@pytest.fixture
def agent() -> CyclomaticComplexityAgent:
    return CyclomaticComplexityAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_many_ifs_high(self, agent: CyclomaticComplexityAgent) -> None:
        src = "def f():\n" + "".join(f"    if x{i}:\n        pass\n" for i in range(22))
        findings = agent.analyze(_ctx(src))
        assert any(f.severity == "high" for f in findings)

    def test_medium_complexity(self, agent: CyclomaticComplexityAgent) -> None:
        src = "def f():\n" + "".join(f"    if x{i}:\n        pass\n" for i in range(12))
        findings = agent.analyze(_ctx(src))
        assert any(f.severity == "medium" for f in findings)

    def test_low_complexity(self, agent: CyclomaticComplexityAgent) -> None:
        src = "def f():\n" + "".join(f"    if x{i}:\n        pass\n" for i in range(8))
        findings = agent.analyze(_ctx(src))
        assert any(f.severity == "low" for f in findings)

    def test_for_loops_count(self, agent: CyclomaticComplexityAgent) -> None:
        src = "def f():\n" + "".join(f"    for i{n} in r:\n        pass\n" for n in range(12))
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_while_loops_count(self, agent: CyclomaticComplexityAgent) -> None:
        src = "def f():\n" + "".join(f"    while c{n}:\n        pass\n" for n in range(12))
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_except_clauses_count(self, agent: CyclomaticComplexityAgent) -> None:
        body = ""
        for i in range(12):
            body += f"    try:\n        pass\n    except E{i}:\n        pass\n"
        src = "def f():\n" + body
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_ifs(self, agent: CyclomaticComplexityAgent) -> None:
        src = "function f() {\n" + "".join(f"    if (x{i}) {{}}\n" for i in range(22)) + "}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_high(self, agent: CyclomaticComplexityAgent) -> None:
        body = "".join(f"        if (x{i}) {{}}\n" for i in range(22))
        src = f"public class T {{\n    void f() {{\n{body}    }}\n}}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_high(self, agent: CyclomaticComplexityAgent) -> None:
        body = "".join(f"    if x{i} > 0 {{}}\n" for i in range(22))
        src = f"package main\n\nfunc f() {{\n{body}}}\n"
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_boolean_ops(self, agent: CyclomaticComplexityAgent) -> None:
        src = "def f():\n    if a and b or c and d and e or f and g or h and i and j and k:\n        pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1


class TestNegativeCases:
    def test_simple_function(self, agent: CyclomaticComplexityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    return 1\n")) == []

    def test_single_if(self, agent: CyclomaticComplexityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    if x:\n        pass\n")) == []

    def test_two_ifs(self, agent: CyclomaticComplexityAgent) -> None:
        src = "def f():\n    if a:\n        pass\n    if b:\n        pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_functions(self, agent: CyclomaticComplexityAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_simple_for(self, agent: CyclomaticComplexityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    for x in y:\n        pass\n")) == []

    def test_simple_js(self, agent: CyclomaticComplexityAgent) -> None:
        assert agent.analyze(_ctx("function f() { return 1; }\n", "javascript", "t.js")) == []

    def test_simple_java(self, agent: CyclomaticComplexityAgent) -> None:
        src = "public class T {\n    void f() {\n        int x = 1;\n    }\n}\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_simple_go(self, agent: CyclomaticComplexityAgent) -> None:
        src = "package main\n\nfunc f() {\n    x := 1\n}\n"
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_multiple_simple_functions(self, agent: CyclomaticComplexityAgent) -> None:
        src = "\n".join(f"def f{i}():\n    return {i}\n" for i in range(5))
        assert agent.analyze(_ctx(src)) == []

    def test_three_ifs(self, agent: CyclomaticComplexityAgent) -> None:
        src = (
            "def f():\n    if a:\n        pass\n    if b:\n        pass\n    if c:\n        pass\n"
        )
        assert agent.analyze(_ctx(src)) == []


class TestEdgeCases:
    def test_empty_source(self, agent: CyclomaticComplexityAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_unsupported_language(self, agent: CyclomaticComplexityAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: CyclomaticComplexityAgent) -> None:
        m = agent.metadata()
        assert m.name == "cyclomatic_complexity"
        assert "python" in m.languages
