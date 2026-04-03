"""Tests for GcPressureAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_05_performance.gc_pressure.agent import GcPressureAgent


@pytest.fixture
def agent() -> GcPressureAgent:
    return GcPressureAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_python_string_concat_loop(self, agent: GcPressureAgent) -> None:
        src = "result = ''\nfor x in items:\n    result += 'line'\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_string_add_loop(self, agent: GcPressureAgent) -> None:
        src = "s = ''\nfor x in items:\n    s = s + 'text'\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_java_string_concat(self, agent: GcPressureAgent) -> None:
        src = 'public class T {\n    void f() {\n        String s = "";\n        for (int i = 0; i < 100; i++) {\n            s += "item";\n        }\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_java_autoboxing(self, agent: GcPressureAgent) -> None:
        src = "public class T {\n    void f() {\n        for (int i = 0; i < 100; i++) {\n            Integer x = Integer.valueOf(i);\n        }\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_string_concat(self, agent: GcPressureAgent) -> None:
        src = 'package main\n\nfunc f() {\n    s := ""\n    for i := 0; i < 100; i++ {\n        s += "item"\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_severity_medium(self, agent: GcPressureAgent) -> None:
        src = "result = ''\nfor x in items:\n    result += 'text'\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "medium"

    def test_python_var_concat(self, agent: GcPressureAgent) -> None:
        src = "s = ''\nfor x in items:\n    s += x\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_java_new_integer(self, agent: GcPressureAgent) -> None:
        src = "public class T {\n    void f() {\n        for (int i = 0; i < n; i++) {\n            Integer x = new Integer(i);\n        }\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_var_concat(self, agent: GcPressureAgent) -> None:
        src = 'package main\n\nfunc f() {\n    result := ""\n    for _, s := range items {\n        result += s\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_critical_axis_type(self, agent: GcPressureAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "critical"


class TestNegativeCases:
    def test_string_concat_outside_loop(self, agent: GcPressureAgent) -> None:
        assert agent.analyze(_ctx("s = 'a' + 'b'\n")) == []

    def test_list_append_in_loop(self, agent: GcPressureAgent) -> None:
        src = "items = []\nfor x in data:\n    items.append(x)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_loops(self, agent: GcPressureAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_empty(self, agent: GcPressureAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_join_pattern(self, agent: GcPressureAgent) -> None:
        src = "parts = []\nfor x in items:\n    parts.append(str(x))\nresult = ''.join(parts)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_js_not_supported(self, agent: GcPressureAgent) -> None:
        src = "let s = '';\nfor (let x of items) {\n    s += 'text';\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_int_accumulation(self, agent: GcPressureAgent) -> None:
        src = "total = 0\nfor x in items:\n    total += x\n"
        # += with non-string might match pattern, but it's about string concat
        # Our regex checks for += with string literals specifically
        assert agent.analyze(_ctx(src)) == []

    def test_python_stringbuilder_pattern(self, agent: GcPressureAgent) -> None:
        src = "import io\nbuf = io.StringIO()\nfor x in items:\n    buf.write(str(x))\n"
        assert agent.analyze(_ctx(src)) == []

    def test_go_strings_builder(self, agent: GcPressureAgent) -> None:
        src = 'package main\n\nimport "strings"\n\nfunc f() {\n    var b strings.Builder\n    for _, s := range items {\n        b.WriteString(s)\n    }\n}\n'
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_simple_function(self, agent: GcPressureAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    return 1\n")) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: GcPressureAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: GcPressureAgent) -> None:
        m = agent.metadata()
        assert m.name == "gc_pressure"
        assert m.axis_type == "critical"
