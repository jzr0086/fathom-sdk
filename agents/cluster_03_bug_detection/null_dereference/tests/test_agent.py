"""Tests for NullDereferenceAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_03_bug_detection.null_dereference.agent import NullDereferenceAgent


@pytest.fixture
def agent() -> NullDereferenceAgent:
    return NullDereferenceAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


class TestPositiveCases:
    def test_python_none_deref(self, agent: NullDereferenceAgent) -> None:
        src = "x = None\nx.method()\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_none_attr(self, agent: NullDereferenceAgent) -> None:
        src = "result = None\nprint(result.value)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_null_deref(self, agent: NullDereferenceAgent) -> None:
        src = "let x = null;\nx.toString();\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_js_undefined_deref(self, agent: NullDereferenceAgent) -> None:
        src = "let x = undefined;\nx.value;\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_null_deref(self, agent: NullDereferenceAgent) -> None:
        src = "String x = null;\nx.length();\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_nil_deref(self, agent: NullDereferenceAgent) -> None:
        src = "var x = nil\nx.Method()\n"
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_severity_high(self, agent: NullDereferenceAgent) -> None:
        f = agent.analyze(_ctx("x = None\nx.method()\n"))[0]
        assert f.severity == "high"

    def test_multiple_nulls(self, agent: NullDereferenceAgent) -> None:
        src = "a = None\na.method()\nb = None\nb.method()\n"
        assert len(agent.analyze(_ctx(src))) >= 2

    def test_null_then_later_use(self, agent: NullDereferenceAgent) -> None:
        src = "x = None\ny = 1\nz = 2\nx.attr\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_ts_null(self, agent: NullDereferenceAgent) -> None:
        src = "let x = null;\nx.toString();\n"
        assert len(agent.analyze(_ctx(src, "typescript", "t.ts"))) >= 1


class TestNegativeCases:
    def test_null_check_before_use(self, agent: NullDereferenceAgent) -> None:
        src = "x = None\nif x is not None:\n    x.method()\n"
        assert agent.analyze(_ctx(src)) == []

    def test_reassigned_before_use(self, agent: NullDereferenceAgent) -> None:
        src = "x = None\nx = get_value()\nx.method()\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_null_assignments(self, agent: NullDereferenceAgent) -> None:
        assert agent.analyze(_ctx("x = 1\nx.bit_length()\n")) == []

    def test_js_null_check(self, agent: NullDereferenceAgent) -> None:
        src = "let x = null;\nif (x !== null) {\n    x.method();\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_truthy_check(self, agent: NullDereferenceAgent) -> None:
        src = "x = None\nif x:\n    x.method()\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty(self, agent: NullDereferenceAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_no_dereference(self, agent: NullDereferenceAgent) -> None:
        src = "x = None\nprint(x)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_comment(self, agent: NullDereferenceAgent) -> None:
        assert agent.analyze(_ctx("# x = None\n# x.method()\n")) == []

    def test_go_nil_check(self, agent: NullDereferenceAgent) -> None:
        src = "x = nil\nif x != nil {\n    x.Method()\n}\n"
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_no_assignments(self, agent: NullDereferenceAgent) -> None:
        assert agent.analyze(_ctx("print('hello')\n")) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: NullDereferenceAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: NullDereferenceAgent) -> None:
        m = agent.metadata()
        assert m.name == "null_dereference"
        assert "npe" in m.tags
