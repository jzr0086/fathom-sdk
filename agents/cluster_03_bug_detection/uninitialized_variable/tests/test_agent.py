"""Tests for UninitializedVariableAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_03_bug_detection.uninitialized_variable.agent import (
    UninitializedVariableAgent,
)


@pytest.fixture
def agent() -> UninitializedVariableAgent:
    return UninitializedVariableAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


# ── Positive cases (should produce findings) ─────────────────────────────


class TestPositiveCases:
    def test_use_before_assignment_python(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(x)\n    x = 1\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].agent_name == "uninitialized_variable"

    def test_conditional_only_assignment_python(self, agent: UninitializedVariableAgent) -> None:
        src = (
            "def foo(flag):\n"
            "    if flag:\n"
            "        result = 42\n"
            "    print(result)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert "result" in findings[0].title

    def test_try_only_assignment_python(self, agent: UninitializedVariableAgent) -> None:
        src = (
            "def foo():\n"
            "    try:\n"
            "        value = compute()\n"
            "    except Exception:\n"
            "        pass\n"
            "    print(value)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert "value" in findings[0].title

    def test_use_before_assignment_javascript(self, agent: UninitializedVariableAgent) -> None:
        src = "function foo() {\n    console.log(x);\n    let x = 1;\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert len(findings) >= 1

    def test_use_before_assignment_typescript(self, agent: UninitializedVariableAgent) -> None:
        src = "function foo() {\n    console.log(x);\n    let x = 1;\n}\n"
        findings = agent.analyze(_ctx(src, "typescript", "test.ts"))
        assert len(findings) >= 1

    def test_use_before_assignment_java(self, agent: UninitializedVariableAgent) -> None:
        src = (
            "public class Test {\n"
            "    public void foo() {\n"
            "        System.out.println(x);\n"
            "        int x = 1;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "Test.java"))
        assert len(findings) >= 1

    def test_use_before_assignment_go(self, agent: UninitializedVariableAgent) -> None:
        src = (
            "package main\n\n"
            "func foo() {\n"
            "    fmt.Println(x)\n"
            "    x := 1\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "test.go"))
        assert len(findings) >= 1

    def test_severity_is_medium(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(x)\n    x = 1\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "medium"

    def test_category_is_bug(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(x)\n    x = 1\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.category == "bug"

    def test_confidence_value(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(x)\n    x = 1\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.confidence == 0.75

    def test_tags(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(x)\n    x = 1\n"
        f = agent.analyze(_ctx(src))[0]
        assert "uninitialized" in f.tags

    def test_file_path_preserved(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(x)\n    x = 1\n"
        f = agent.analyze(_ctx(src, fp="src/module.py"))[0]
        assert f.file_path == "src/module.py"

    def test_conditional_if_no_else(self, agent: UninitializedVariableAgent) -> None:
        """Variable assigned in if-branch only, used after the block."""
        src = (
            "def process(data):\n"
            "    if data:\n"
            "        msg = 'found'\n"
            "    print(msg)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1


# ── Negative cases (should produce no findings) ──────────────────────────


class TestNegativeCases:
    def test_properly_initialized_python(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    x = 1\n    print(x)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_function_parameter(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo(x):\n    print(x)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_builtin_names(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(len([1, 2, 3]))\n"
        assert agent.analyze(_ctx(src)) == []

    def test_import_names(self, agent: UninitializedVariableAgent) -> None:
        src = "import os\n\ndef foo():\n    os.path.exists('x')\n"
        assert agent.analyze(_ctx(src)) == []

    def test_for_loop_variable(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    for i in range(10):\n        print(i)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: UninitializedVariableAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: UninitializedVariableAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_comments_only(self, agent: UninitializedVariableAgent) -> None:
        assert agent.analyze(_ctx("# just a comment\n# another\n")) == []

    def test_assigned_before_use_js(self, agent: UninitializedVariableAgent) -> None:
        src = "function foo() {\n    let x = 1;\n    console.log(x);\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_uppercase_constant(self, agent: UninitializedVariableAgent) -> None:
        """UPPER_CASE names are assumed to be constants/globals."""
        src = "def foo():\n    print(MAX_SIZE)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_class_name_not_flagged(self, agent: UninitializedVariableAgent) -> None:
        """Capitalized names (likely classes/types) are not flagged."""
        src = "def foo():\n    obj = MyClass()\n"
        assert agent.analyze(_ctx(src)) == []

    def test_self_access(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo(self):\n    print(self.x)\n"
        assert agent.analyze(_ctx(src)) == []


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unsupported_language(self, agent: UninitializedVariableAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "test.cob")) == []

    def test_metadata_name(self, agent: UninitializedVariableAgent) -> None:
        m = agent.metadata()
        assert m.name == "uninitialized_variable"

    def test_metadata_version(self, agent: UninitializedVariableAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages(self, agent: UninitializedVariableAgent) -> None:
        m = agent.metadata()
        assert "python" in m.languages
        assert "javascript" in m.languages
        assert "go" in m.languages

    def test_metadata_axis_type(self, agent: UninitializedVariableAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "aware"

    def test_metadata_model_not_required(self, agent: UninitializedVariableAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False

    def test_metadata_zero_cost(self, agent: UninitializedVariableAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: UninitializedVariableAgent) -> None:
        src = "def foo():\n    print(x)\n    x = 1\n"
        f = agent.analyze(_ctx(src))[0]
        explanation = agent.explain(f)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
