"""Tests for OffByOneAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_03_bug_detection.off_by_one.agent import OffByOneAgent


@pytest.fixture
def agent() -> OffByOneAgent:
    return OffByOneAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


class TestPositiveCases:
    def test_array_at_len(self, agent: OffByOneAgent) -> None:
        assert len(agent.analyze(_ctx("x = arr[len(arr)]\n"))) >= 1

    def test_array_at_length(self, agent: OffByOneAgent) -> None:
        src = "let x = arr[arr.length];\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_le_len(self, agent: OffByOneAgent) -> None:
        assert len(agent.analyze(_ctx("while i <= len(arr):\n    pass\n"))) >= 1

    def test_le_length(self, agent: OffByOneAgent) -> None:
        src = "for (let i = 0; i <= arr.length; i++) {}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_range_len_plus_1(self, agent: OffByOneAgent) -> None:
        assert len(agent.analyze(_ctx("for i in range(len(arr)+1):\n    pass\n"))) >= 1

    def test_range_1_len(self, agent: OffByOneAgent) -> None:
        assert len(agent.analyze(_ctx("for i in range(1, len(arr)):\n    pass\n"))) >= 1

    def test_range_len_minus_1(self, agent: OffByOneAgent) -> None:
        assert len(agent.analyze(_ctx("for i in range(len(arr)-1):\n    pass\n"))) >= 1

    def test_le_size(self, agent: OffByOneAgent) -> None:
        src = "public class T {\n    void f() {\n        while (i <= list.size()) {}\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_array_at_size(self, agent: OffByOneAgent) -> None:
        src = "list.get(list.size());\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_high_severity_for_access(self, agent: OffByOneAgent) -> None:
        f = agent.analyze(_ctx("x = arr[len(arr)]\n"))[0]
        assert f.severity == "high"


class TestNegativeCases:
    def test_range_len(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("for i in range(len(arr)):\n    pass\n")) == []

    def test_lt_len(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("while i < len(arr):\n    pass\n")) == []

    def test_lt_length(self, agent: OffByOneAgent) -> None:
        src = "for (let i = 0; i < arr.length; i++) {}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_normal_access(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("x = arr[0]\n")) == []

    def test_len_minus_1_access(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("x = arr[len(arr) - 1]\n")) == []

    def test_no_arrays(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_comment(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("# arr[len(arr)]\n")) == []

    def test_simple_for(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("for x in items:\n    pass\n")) == []

    def test_empty(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_enumerate(self, agent: OffByOneAgent) -> None:
        assert agent.analyze(_ctx("for i, x in enumerate(arr):\n    pass\n")) == []


class TestEdgeCases:
    def test_metadata(self, agent: OffByOneAgent) -> None:
        m = agent.metadata()
        assert m.name == "off_by_one"
        assert "bug" in m.tags
