"""Tests for InfiniteLoopAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_03_bug_detection.infinite_loop.agent import InfiniteLoopAgent


@pytest.fixture
def agent() -> InfiniteLoopAgent:
    return InfiniteLoopAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# -----------------------------------------------------------------------
# Positive cases (should produce findings)
# -----------------------------------------------------------------------


class TestPositiveCases:
    def test_python_while_true_no_break(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    print('forever')\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_python_while_true_pass(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_js_while_true_no_break(self, agent: InfiniteLoopAgent) -> None:
        src = "while (true) {\n    console.log('forever');\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_js_for_ever(self, agent: InfiniteLoopAgent) -> None:
        src = "for (;;) {\n    console.log('forever');\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_java_while_true_no_break(self, agent: InfiniteLoopAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        while (true) {\n"
            "            System.out.println(\"forever\");\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_java_for_ever(self, agent: InfiniteLoopAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        for (;;) {\n"
            "            System.out.println(\"forever\");\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_python_recursion_no_base_case(self, agent: InfiniteLoopAgent) -> None:
        src = "def factorial(n):\n    return n * factorial(n - 1)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("base case" in f.title for f in findings)
        assert any(f.severity == "medium" for f in findings)

    def test_js_recursion_no_base_case(self, agent: InfiniteLoopAgent) -> None:
        src = "function countdown(n) {\n    console.log(n);\n    countdown(n - 1);\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 1

    def test_python_unmodified_loop_var(self, agent: InfiniteLoopAgent) -> None:
        src = "x = 10\nwhile x > 0:\n    print(x)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("never modified" in f.title for f in findings)

    def test_ts_while_true_no_break(self, agent: InfiniteLoopAgent) -> None:
        src = "while (true) {\n    console.log('forever');\n}\n"
        findings = agent.analyze(_ctx(src, "typescript", "t.ts"))
        assert len(findings) >= 1

    def test_finding_agent_name(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].agent_name == "infinite_loop"

    def test_finding_category(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].category == "bug"

    def test_finding_confidence(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].confidence == 0.78


# -----------------------------------------------------------------------
# Negative cases (should NOT produce findings)
# -----------------------------------------------------------------------


class TestNegativeCases:
    def test_python_while_true_with_break(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    x = input()\n    if x == 'q':\n        break\n"
        assert agent.analyze(_ctx(src)) == []

    def test_python_while_true_with_return(self, agent: InfiniteLoopAgent) -> None:
        src = "def f():\n    while True:\n        data = read()\n        if data:\n            return data\n"
        assert agent.analyze(_ctx(src)) == []

    def test_python_while_true_with_raise(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    x = next(it)\n    if x is None:\n        raise StopIteration\n"
        assert agent.analyze(_ctx(src)) == []

    def test_js_while_true_with_break(self, agent: InfiniteLoopAgent) -> None:
        src = "while (true) {\n    if (done) break;\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_js_for_ever_with_return(self, agent: InfiniteLoopAgent) -> None:
        src = "function f() {\n    for (;;) {\n        if (ok) return;\n    }\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_python_proper_recursion(self, agent: InfiniteLoopAgent) -> None:
        src = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_js_proper_recursion(self, agent: InfiniteLoopAgent) -> None:
        src = (
            "function fib(n) {\n"
            "    if (n <= 1) return n;\n"
            "    return fib(n - 1) + fib(n - 2);\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_normal_for_loop(self, agent: InfiniteLoopAgent) -> None:
        src = "for i in range(10):\n    print(i)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_normal_while_with_modification(self, agent: InfiniteLoopAgent) -> None:
        src = "x = 10\nwhile x > 0:\n    x -= 1\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_loops_or_recursion(self, agent: InfiniteLoopAgent) -> None:
        src = "x = 1\ny = 2\nprint(x + y)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_while_false(self, agent: InfiniteLoopAgent) -> None:
        src = "while False:\n    pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_js_normal_for(self, agent: InfiniteLoopAgent) -> None:
        src = "for (let i = 0; i < 10; i++) {\n    console.log(i);\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_java_while_true_with_break(self, agent: InfiniteLoopAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        while (true) {\n"
            "            if (done) break;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src, "java", "T.java")) == []


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_empty(self, agent: InfiniteLoopAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_unsupported_lang(self, agent: InfiniteLoopAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: InfiniteLoopAgent) -> None:
        m = agent.metadata()
        assert m.name == "infinite_loop"
        assert m.version == "0.1.0"
        assert m.axis_type == "aware"
        assert m.methodology == "bug_detection"
        assert "python" in m.languages
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        explanation = agent.explain(findings[0])
        assert "Infinite" in explanation or "infinite" in explanation

    def test_file_path_propagated(self, agent: InfiniteLoopAgent) -> None:
        src = "while True:\n    pass\n"
        findings = agent.analyze(_ctx(src, fp="src/server.py"))
        assert findings[0].file_path == "src/server.py"
