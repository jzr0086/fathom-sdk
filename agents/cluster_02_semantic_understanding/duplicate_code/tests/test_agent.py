"""Tests for DuplicateCodeAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_02_semantic_understanding.duplicate_code.agent import DuplicateCodeAgent


@pytest.fixture
def agent() -> DuplicateCodeAgent:
    return DuplicateCodeAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_identical_functions(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def func_a(x):\n    y = x + 1\n    z = y * 2\n    w = z - 3\n    return w + 4\n\n"
            "def func_b(a):\n    b = a + 1\n    c = b * 2\n    d = c - 3\n    return d + 4\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 2

    def test_three_duplicates(self, agent: DuplicateCodeAgent) -> None:
        template = "def {name}(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    d = c + 4\n    return d\n\n"
        src = template.format(name="f1") + template.format(name="f2") + template.format(name="f3")
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 3

    def test_js_duplicates(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "function calcA(x) {\n    let y = x + 1;\n    let z = y * 2;\n    let w = z - 3;\n    return w + 4;\n}\n\n"
            "function calcB(a) {\n    let b = a + 1;\n    let c = b * 2;\n    let d = c - 3;\n    return d + 4;\n}\n"
        )
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 2

    def test_severity_medium(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def f1(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    return c + 4\n\n"
            "def f2(y):\n    a = y + 1\n    b = a * 2\n    c = b - 3\n    return c + 4\n"
        )
        findings = agent.analyze(_ctx(src))
        assert all(f.severity == "medium" for f in findings)

    def test_different_var_names(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def process_a(data):\n    result = data + 1\n    output = result * 2\n    final = output - 3\n    value = final + 4\n    return value\n\n"
            "def process_b(info):\n    temp = info + 1\n    calc = temp * 2\n    res = calc - 3\n    ans = res + 4\n    return ans\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 2

    def test_different_strings(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def log_a(msg):\n    x = 'hello'\n    y = x + msg\n    z = y.upper()\n    w = z.strip()\n    return w\n\n"
            "def log_b(text):\n    x = 'world'\n    y = x + text\n    z = y.upper()\n    w = z.strip()\n    return w\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 2

    def test_java_duplicates(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "public class T {\n"
            "    int calcA(int x) {\n        int a = x + 1;\n        int b = a * 2;\n        int c = b - 3;\n        return c + 4;\n    }\n"
            "    int calcB(int y) {\n        int a = y + 1;\n        int b = a * 2;\n        int c = b - 3;\n        return c + 4;\n    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 2

    def test_finding_fields(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def f1(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    return c + 4\n\n"
            "def f2(y):\n    a = y + 1\n    b = a * 2\n    c = b - 3\n    return c + 4\n"
        )
        findings = agent.analyze(_ctx(src))
        assert all(f.agent_name == "duplicate_code" for f in findings)
        assert all(f.category == "quality" for f in findings)

    def test_go_duplicates(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "package main\n\n"
            "func calcA(x int) int {\n    a := x + 1\n    b := a * 2\n    c := b - 3\n    return c + 4\n}\n\n"
            "func calcB(y int) int {\n    a := y + 1\n    b := a * 2\n    c := b - 3\n    return c + 4\n}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "t.go"))
        assert len(findings) >= 2

    def test_ts_duplicates(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "function calcA(x: number): number {\n    let y = x + 1;\n    let z = y * 2;\n    let w = z - 3;\n    return w + 4;\n}\n\n"
            "function calcB(a: number): number {\n    let b = a + 1;\n    let c = b * 2;\n    let d = c - 3;\n    return d + 4;\n}\n"
        )
        findings = agent.analyze(_ctx(src, "typescript", "t.ts"))
        assert len(findings) >= 2


class TestNegativeCases:
    def test_different_logic(self, agent: DuplicateCodeAgent) -> None:
        src = "def f(x):\n    return x + 1\n\ndef g(x):\n    return x * 2\n"
        assert agent.analyze(_ctx(src)) == []

    def test_single_function(self, agent: DuplicateCodeAgent) -> None:
        src = "def f(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    return c\n"
        assert agent.analyze(_ctx(src)) == []

    def test_short_functions(self, agent: DuplicateCodeAgent) -> None:
        src = "def a():\n    return 1\n\ndef b():\n    return 1\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_functions(self, agent: DuplicateCodeAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_empty(self, agent: DuplicateCodeAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_different_structures(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def f(x):\n    if x:\n        return 1\n    return 2\n    a = 3\n\n"
            "def g(x):\n    for i in x:\n        print(i)\n    return None\n    b = 4\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_one_short_one_long(self, agent: DuplicateCodeAgent) -> None:
        src = "def short():\n    return 1\n\ndef long_fn(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    d = c + 4\n    return d\n"
        assert agent.analyze(_ctx(src)) == []

    def test_different_operations(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def add(x):\n    a = x + 1\n    b = a + 2\n    c = b + 3\n    d = c + 4\n    return d\n\n"
            "def mul(x):\n    a = x * 1\n    b = a * 2\n    c = b * 3\n    d = c * 4\n    return d\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_class_methods(self, agent: DuplicateCodeAgent) -> None:
        src = "class C:\n    def a(self):\n        return 1\n    def b(self):\n        return 2\n"
        assert agent.analyze(_ctx(src)) == []

    def test_different_returns(self, agent: DuplicateCodeAgent) -> None:
        src = (
            "def f1(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    return c\n\n"
            "def f2(x):\n    a = x + 1\n    b = a * 2\n    c = b - 3\n    print(c)\n"
        )
        # Different last statements should prevent matching
        assert agent.analyze(_ctx(src)) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: DuplicateCodeAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: DuplicateCodeAgent) -> None:
        m = agent.metadata()
        assert m.name == "duplicate_code"
        assert m.axis_type == "agnostic"
