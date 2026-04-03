"""Tests for CognitiveComplexityAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_07_code_quality.cognitive_complexity.agent import CognitiveComplexityAgent


@pytest.fixture
def agent() -> CognitiveComplexityAgent:
    return CognitiveComplexityAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_deep_nesting_high(self, agent: CognitiveComplexityAgent) -> None:
        # if(+1) > if(+2) > if(+3) > if(+4) > if(+5) > if(+6) = 21
        src = "def f():\n    if a:\n        if b:\n            if c:\n                if d:\n                    if e:\n                        if f:\n                            pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_many_flat_ifs_medium(self, agent: CognitiveComplexityAgent) -> None:
        src = "def f():\n" + "".join(f"    if x{i}:\n        pass\n" for i in range(20))
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_nested_for_in_if(self, agent: CognitiveComplexityAgent) -> None:
        src = (
            "def f():\n"
            "    if a:\n"
            "        for x in y:\n"
            "            if b:\n"
            "                for z in w:\n"
            "                    if c:\n"
            "                        pass\n"
            + "".join(f"    if x{i}:\n        pass\n" for i in range(10))
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_js_nested(self, agent: CognitiveComplexityAgent) -> None:
        src = "function f() {\n    if (a) {\n        if (b) {\n            if (c) {\n                if (d) {\n                    if (e) {\n                        if (f) {}\n                    }\n                }\n            }\n        }\n    }\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 1

    def test_java_nested(self, agent: CognitiveComplexityAgent) -> None:
        inner = "".join(f"            if (x{i}) {{}}\n" for i in range(10))
        src = f"public class T {{\n    void f() {{\n        if (a) {{\n{inner}        }}\n    }}\n}}\n"
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_go_nested(self, agent: CognitiveComplexityAgent) -> None:
        inner = "".join(f"        if x{i} > 0 {{}}\n" for i in range(10))
        src = f"package main\n\nfunc f() {{\n    if a > 0 {{\n{inner}    }}\n}}\n"
        findings = agent.analyze(_ctx(src, "go", "t.go"))
        assert len(findings) >= 1

    def test_try_adds_nesting(self, agent: CognitiveComplexityAgent) -> None:
        src = "def f():\n" + "".join(
            f"    try:\n        if x{i}:\n            pass\n    except:\n        pass\n"
            for i in range(10)
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_while_loops(self, agent: CognitiveComplexityAgent) -> None:
        src = "def f():\n" + "".join(
            f"    while c{i}:\n        if x{i}:\n            pass\n" for i in range(10)
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_severity_high(self, agent: CognitiveComplexityAgent) -> None:
        # Very deep nesting to get high severity (>25)
        src = "def f():\n"
        indent = "    "
        for i in range(8):
            src += indent + f"if x{i}:\n"
            indent += "    "
        src += indent + "pass\n"
        findings = agent.analyze(_ctx(src))
        assert any(f.severity == "high" for f in findings)

    def test_multiple_functions(self, agent: CognitiveComplexityAgent) -> None:
        src = ""
        for fn in ["a", "b"]:
            src += (
                f"def {fn}():\n"
                + "".join(f"    if x{i}:\n        pass\n" for i in range(20))
                + "\n"
            )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 2


class TestNegativeCases:
    def test_simple_function(self, agent: CognitiveComplexityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    return 1\n")) == []

    def test_single_if(self, agent: CognitiveComplexityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    if x:\n        pass\n")) == []

    def test_few_flat_ifs(self, agent: CognitiveComplexityAgent) -> None:
        src = "def f():\n" + "".join(f"    if x{i}:\n        pass\n" for i in range(5))
        assert agent.analyze(_ctx(src)) == []

    def test_no_functions(self, agent: CognitiveComplexityAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_simple_for(self, agent: CognitiveComplexityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    for x in y:\n        pass\n")) == []

    def test_simple_js(self, agent: CognitiveComplexityAgent) -> None:
        assert agent.analyze(_ctx("function f() { return 1; }\n", "javascript", "t.js")) == []

    def test_simple_java(self, agent: CognitiveComplexityAgent) -> None:
        src = "public class T {\n    void f() {\n        int x = 1;\n    }\n}\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_simple_go(self, agent: CognitiveComplexityAgent) -> None:
        src = "package main\n\nfunc f() {\n    x := 1\n}\n"
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_three_ifs(self, agent: CognitiveComplexityAgent) -> None:
        src = (
            "def f():\n    if a:\n        pass\n    if b:\n        pass\n    if c:\n        pass\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_class_simple(self, agent: CognitiveComplexityAgent) -> None:
        src = "class Foo:\n    def a(self):\n        if x:\n            pass\n"
        assert agent.analyze(_ctx(src)) == []


class TestEdgeCases:
    def test_empty(self, agent: CognitiveComplexityAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_unsupported_lang(self, agent: CognitiveComplexityAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: CognitiveComplexityAgent) -> None:
        m = agent.metadata()
        assert m.name == "cognitive_complexity"
