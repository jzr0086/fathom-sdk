"""Tests for UnnecessaryRecomputationAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_05_performance.unnecessary_recomputation.agent import (
    UnnecessaryRecomputationAgent,
)


@pytest.fixture
def agent() -> UnnecessaryRecomputationAgent:
    return UnnecessaryRecomputationAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_sorted_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for i in range(10):\n    s = sorted(data)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_compile_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for line in lines:\n    pat = re.compile(pattern)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_load_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for x in items:\n    data = json.load(f)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_parse_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for (let i = 0; i < n; i++) {\n    const d = JSON.parse(str);\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_decode_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for item in items:\n    text = data.decode('utf-8')\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_findall_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for line in lines:\n    matches = re.findall(pattern, text)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_severity_medium(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for i in range(10):\n    s = sorted(data)\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "medium"

    def test_read_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for x in items:\n    content = f.read()\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_deepcopy_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for x in items:\n    obj = copy.deepcopy(template)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_encode_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for x in items:\n    b = text.encode('utf-8')\n"
        assert len(agent.analyze(_ctx(src))) >= 1


class TestNegativeCases:
    def test_sorted_outside_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("s = sorted(data)\nfor x in s:\n    pass\n")) == []

    def test_simple_call_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("for x in items:\n    print(x)\n")) == []

    def test_no_loops(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("x = sorted(data)\n")) == []

    def test_empty(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_simple_math(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("for i in range(10):\n    x = i + 1\n")) == []

    def test_no_expensive_call(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = "for x in items:\n    y = len(x)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_append_in_loop(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("for x in items:\n    result.append(x)\n")) == []

    def test_string_format(self, agent: UnnecessaryRecomputationAgent) -> None:
        src = 'for x in items:\n    s = f"item: {x}"\n'
        assert agent.analyze(_ctx(src)) == []

    def test_function_call(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("for x in items:\n    process(x)\n")) == []

    def test_method_call(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("for x in items:\n    x.validate()\n")) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: UnnecessaryRecomputationAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: UnnecessaryRecomputationAgent) -> None:
        m = agent.metadata()
        assert m.name == "unnecessary_recomputation"
