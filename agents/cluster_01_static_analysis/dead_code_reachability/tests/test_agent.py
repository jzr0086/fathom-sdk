"""Tests for DeadCodeReachabilityAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_01_static_analysis.dead_code_reachability.agent import DeadCodeReachabilityAgent


@pytest.fixture
def agent() -> DeadCodeReachabilityAgent:
    return DeadCodeReachabilityAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_code_after_return(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "def f():\n    return 1\n    x = 2\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_code_after_raise(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "def f():\n    raise ValueError('bad')\n    x = 2\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_after_return(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "function f() {\n    return 1;\n    let x = 2;\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_js_after_throw(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "function f() {\n    throw new Error('bad');\n    let x = 2;\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_after_return(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "public class T {\n    int f() {\n        return 1;\n        int x = 2;\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_after_return(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "package main\n\nfunc f() int {\n    return 1\n    x := 2\n}\n"
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_multiple_dead_lines(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "def f():\n    return 1\n    x = 2\n    y = 3\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_severity_low(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "def f():\n    return 1\n    x = 2\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "low"

    def test_two_functions(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "def a():\n    return 1\n    x = 2\ndef b():\n    return 3\n    y = 4\n"
        assert len(agent.analyze(_ctx(src))) >= 2

    def test_ts_after_throw(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "function f(): never {\n    throw new Error('x');\n    let x = 1;\n}\n"
        assert len(agent.analyze(_ctx(src, "typescript", "t.ts"))) >= 1


class TestNegativeCases:
    def test_return_at_end(self, agent: DeadCodeReachabilityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    x = 1\n    return x\n")) == []

    def test_no_return(self, agent: DeadCodeReachabilityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    x = 1\n    y = 2\n")) == []

    def test_conditional_return(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "def f():\n    if x:\n        return 1\n    return 2\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_functions(self, agent: DeadCodeReachabilityAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_empty(self, agent: DeadCodeReachabilityAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_pass_function(self, agent: DeadCodeReachabilityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    pass\n")) == []

    def test_js_return_at_end(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "function f() {\n    let x = 1;\n    return x;\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_try_except(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "def f():\n    try:\n        return 1\n    except:\n        return 2\n"
        assert agent.analyze(_ctx(src)) == []

    def test_single_return(self, agent: DeadCodeReachabilityAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    return None\n")) == []

    def test_class_method(self, agent: DeadCodeReachabilityAgent) -> None:
        src = "class C:\n    def m(self):\n        return self.x\n"
        assert agent.analyze(_ctx(src)) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: DeadCodeReachabilityAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: DeadCodeReachabilityAgent) -> None:
        m = agent.metadata()
        assert m.name == "dead_code_reachability"
