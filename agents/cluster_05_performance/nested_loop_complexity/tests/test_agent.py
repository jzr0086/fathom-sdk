"""Tests for NestedLoopComplexityAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_05_performance.nested_loop_complexity.agent import NestedLoopComplexityAgent


@pytest.fixture
def agent() -> NestedLoopComplexityAgent:
    return NestedLoopComplexityAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_nested_2_low(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for i in range(10):\n    for j in range(10):\n        pass\n"
        findings = agent.analyze(_ctx(src))
        nested = [f for f in findings if f.severity == "low"]
        assert len(nested) >= 1

    def test_nested_3_medium(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for i in r:\n    for j in r:\n        for k in r:\n            pass\n"
        findings = agent.analyze(_ctx(src))
        assert any(f.severity == "medium" for f in findings)

    def test_nested_4_high(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for a in r:\n    for b in r:\n        for c in r:\n            for d in r:\n                pass\n"
        findings = agent.analyze(_ctx(src))
        assert any(f.severity == "high" for f in findings)

    def test_js_nested(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for (let i = 0; i < n; i++) {\n    for (let j = 0; j < n; j++) {\n        x++;\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_nested(self, agent: NestedLoopComplexityAgent) -> None:
        src = "public class T {\n    void f() {\n        for (int i = 0; i < n; i++) {\n            for (int j = 0; j < n; j++) {\n                x++;\n            }\n        }\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_nested(self, agent: NestedLoopComplexityAgent) -> None:
        src = "package main\n\nfunc f() {\n    for i := 0; i < n; i++ {\n        for j := 0; j < n; j++ {\n            x++\n        }\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_while_nested(self, agent: NestedLoopComplexityAgent) -> None:
        src = "while a:\n    while b:\n        pass\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_for_while_nested(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for x in r:\n    while y:\n        pass\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_title_shows_depth(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for i in r:\n    for j in r:\n        pass\n"
        findings = agent.analyze(_ctx(src))
        assert any("depth 2" in f.title for f in findings)

    def test_multiple_nested_groups(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for a in r:\n    for b in r:\n        pass\nfor c in r:\n    for d in r:\n        pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 2


class TestNegativeCases:
    def test_single_for(self, agent: NestedLoopComplexityAgent) -> None:
        assert agent.analyze(_ctx("for x in r:\n    pass\n")) == []

    def test_single_while(self, agent: NestedLoopComplexityAgent) -> None:
        assert agent.analyze(_ctx("while x:\n    pass\n")) == []

    def test_sibling_loops(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for a in r:\n    pass\nfor b in r:\n    pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_loops(self, agent: NestedLoopComplexityAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_single_js_loop(self, agent: NestedLoopComplexityAgent) -> None:
        src = "for (let i = 0; i < n; i++) { x++; }\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_single_java_loop(self, agent: NestedLoopComplexityAgent) -> None:
        src = "public class T {\n    void f() {\n        for (int i = 0; i < n; i++) {}\n    }\n}\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_single_go_loop(self, agent: NestedLoopComplexityAgent) -> None:
        src = (
            "package main\n\nfunc f() {\n    for i := 0; i < 10; i++ {\n        x := i\n    }\n}\n"
        )
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_function_with_single_loop(self, agent: NestedLoopComplexityAgent) -> None:
        src = "def f():\n    for x in range(10):\n        print(x)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_list_comprehension(self, agent: NestedLoopComplexityAgent) -> None:
        src = "x = [i for i in range(10)]\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_function(self, agent: NestedLoopComplexityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    pass\n")) == []


class TestEdgeCases:
    def test_empty(self, agent: NestedLoopComplexityAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_unsupported_lang(self, agent: NestedLoopComplexityAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: NestedLoopComplexityAgent) -> None:
        m = agent.metadata()
        assert m.name == "nested_loop_complexity"
        assert m.methodology == "performance"
