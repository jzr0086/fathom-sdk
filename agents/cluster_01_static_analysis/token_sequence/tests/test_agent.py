"""Tests for TokenSequenceAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_01_static_analysis.token_sequence.agent import TokenSequenceAgent


@pytest.fixture
def agent() -> TokenSequenceAgent:
    return TokenSequenceAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


# -----------------------------------------------------------------------
# Positive cases (10+)
# -----------------------------------------------------------------------


class TestPositiveCases:
    def test_consecutive_duplicate_lines(self, agent: TokenSequenceAgent) -> None:
        src = "x = 1\nx = 1\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].title == "Consecutive duplicate statement"

    def test_self_assignment_simple(self, agent: TokenSequenceAgent) -> None:
        src = "x = x\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].title == "Self-assignment has no effect"

    def test_self_assignment_attribute(self, agent: TokenSequenceAgent) -> None:
        src = "self.value = self.value\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].title == "Self-assignment has no effect"

    def test_self_comparison_equal(self, agent: TokenSequenceAgent) -> None:
        src = "if x == x:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].title == "Comparison of identical operands"

    def test_self_comparison_not_equal(self, agent: TokenSequenceAgent) -> None:
        src = "if x != x:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert "Comparison of identical operands" in findings[0].title

    def test_self_comparison_greater(self, agent: TokenSequenceAgent) -> None:
        src = "if count > count:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_return_followed_by_code(self, agent: TokenSequenceAgent) -> None:
        src = "def f():\n    return 1\n    x = 2\n"
        findings = agent.analyze(_ctx(src))
        assert any(f.severity == "medium" for f in findings)
        assert any("Unreachable" in f.title for f in findings)

    def test_raise_followed_by_code(self, agent: TokenSequenceAgent) -> None:
        src = "def f():\n    raise ValueError('bad')\n    x = 2\n"
        findings = agent.analyze(_ctx(src))
        assert any("Unreachable" in f.title for f in findings)

    def test_duplicate_dict_keys(self, agent: TokenSequenceAgent) -> None:
        src = 'data = {\n    "name": "Alice",\n    "name": "Bob",\n}\n'
        findings = agent.analyze(_ctx(src))
        assert any("Duplicate dictionary" in f.title for f in findings)

    def test_js_throw_followed_by_code(self, agent: TokenSequenceAgent) -> None:
        src = "function f() {\n    throw new Error('x');\n    let y = 1;\n}\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert any("Unreachable" in f.title for f in findings)

    def test_repeated_catch_blocks(self, agent: TokenSequenceAgent) -> None:
        src = (
            "try:\n"
            "    x = 1\n"
            "except TypeError:\n"
            "    log(e)\n"
            "except ValueError:\n"
            "    log(e)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert any("Repeated identical catch" in f.title for f in findings)

    def test_severity_medium_for_unreachable(self, agent: TokenSequenceAgent) -> None:
        src = "def f():\n    return 1\n    x = 2\n"
        findings = agent.analyze(_ctx(src))
        unreachable = [f for f in findings if "Unreachable" in f.title]
        assert len(unreachable) >= 1
        assert unreachable[0].severity == "medium"

    def test_severity_low_for_redundant(self, agent: TokenSequenceAgent) -> None:
        src = "x = x\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].severity == "low"

    def test_confidence_085(self, agent: TokenSequenceAgent) -> None:
        src = "x = x\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].confidence == 0.85


# -----------------------------------------------------------------------
# Negative cases (10+)
# -----------------------------------------------------------------------


class TestNegativeCases:
    def test_normal_code(self, agent: TokenSequenceAgent) -> None:
        src = "x = 1\ny = 2\nz = x + y\n"
        assert agent.analyze(_ctx(src)) == []

    def test_different_variables(self, agent: TokenSequenceAgent) -> None:
        src = "x = y\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: TokenSequenceAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: TokenSequenceAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_return_at_end_of_function(self, agent: TokenSequenceAgent) -> None:
        src = "def f():\n    x = 1\n    return x\n"
        assert agent.analyze(_ctx(src)) == []

    def test_different_assignments(self, agent: TokenSequenceAgent) -> None:
        src = "x = 1\ny = 1\n"
        assert agent.analyze(_ctx(src)) == []

    def test_comparison_different_operands(self, agent: TokenSequenceAgent) -> None:
        src = "if x == y:\n    pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_return_followed_by_except(self, agent: TokenSequenceAgent) -> None:
        src = "try:\n    return 1\nexcept:\n    pass\n"
        assert not any(
            "Unreachable" in f.title for f in agent.analyze(_ctx(src))
        )

    def test_return_followed_by_else(self, agent: TokenSequenceAgent) -> None:
        src = "if True:\n    return 1\nelse:\n    return 2\n"
        assert not any(
            "Unreachable" in f.title for f in agent.analyze(_ctx(src))
        )

    def test_return_followed_by_closing_brace(self, agent: TokenSequenceAgent) -> None:
        src = "function f() {\n    return 1;\n}\n"
        assert not any(
            "Unreachable" in f.title
            for f in agent.analyze(_ctx(src, "javascript", "t.js"))
        )

    def test_unique_dict_keys(self, agent: TokenSequenceAgent) -> None:
        src = 'data = {\n    "name": "Alice",\n    "age": 30,\n}\n'
        assert not any(
            "Duplicate dictionary" in f.title for f in agent.analyze(_ctx(src))
        )

    def test_similar_but_not_identical_lines(self, agent: TokenSequenceAgent) -> None:
        src = "x = 1\nx = 2\n"
        assert not any(
            "Consecutive duplicate" in f.title for f in agent.analyze(_ctx(src))
        )


# -----------------------------------------------------------------------
# Edge cases / metadata
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_metadata_name(self, agent: TokenSequenceAgent) -> None:
        m = agent.metadata()
        assert m.name == "token_sequence"
        assert m.version == "0.1.0"

    def test_metadata_axis_type(self, agent: TokenSequenceAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "aware"

    def test_metadata_languages(self, agent: TokenSequenceAgent) -> None:
        m = agent.metadata()
        assert "python" in m.languages
        assert "javascript" in m.languages
        assert "go" in m.languages

    def test_metadata_model_not_required(self, agent: TokenSequenceAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_file_path_propagated(self, agent: TokenSequenceAgent) -> None:
        src = "x = x\n"
        findings = agent.analyze(_ctx(src, fp="src/module.py"))
        assert findings[0].file_path == "src/module.py"

    def test_category_quality(self, agent: TokenSequenceAgent) -> None:
        src = "x = x\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].category == "quality"

    def test_explain_returns_string(self, agent: TokenSequenceAgent) -> None:
        src = "x = x\n"
        findings = agent.analyze(_ctx(src))
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0
