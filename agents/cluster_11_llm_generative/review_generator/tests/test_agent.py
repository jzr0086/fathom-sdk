"""Tests for ReviewGeneratorAgent."""

from __future__ import annotations

import os

# Force mock mode so tests never call the real LLM API.
os.environ["FATHOM_LLM_MOCK"] = "1"

import pytest

from fathom_sdk import CodeContext

from agents.cluster_11_llm_generative.review_generator.agent import ReviewGeneratorAgent


@pytest.fixture
def agent() -> ReviewGeneratorAgent:
    return ReviewGeneratorAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


# ---------------------------------------------------------------------------
# Positive cases — code with issues should produce findings in mock mode
# ---------------------------------------------------------------------------


class TestPositiveCases:
    """Non-empty source should produce mock findings."""

    def test_simple_function(self, agent: ReviewGeneratorAgent) -> None:
        src = "def add(a, b):\n    return a + b\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_class_definition(self, agent: ReviewGeneratorAgent) -> None:
        src = "class Foo:\n    def bar(self):\n        pass\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_javascript_code(self, agent: ReviewGeneratorAgent) -> None:
        src = "function greet(name) { return 'Hello ' + name; }\n"
        assert len(agent.analyze(_ctx(src, "javascript", "app.js"))) >= 1

    def test_multi_line_code(self, agent: ReviewGeneratorAgent) -> None:
        src = (
            "import os\n"
            "def process(data):\n"
            "    result = data.strip()\n"
            "    return result\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_findings_have_correct_agent_name(self, agent: ReviewGeneratorAgent) -> None:
        src = "x = 1\ny = 2\n"
        findings = agent.analyze(_ctx(src))
        for f in findings:
            assert f.agent_name == "review_generator"

    def test_findings_severity_is_medium(self, agent: ReviewGeneratorAgent) -> None:
        src = "x = 1\ny = 2\n"
        findings = agent.analyze(_ctx(src))
        for f in findings:
            assert f.severity == "medium"

    def test_findings_confidence(self, agent: ReviewGeneratorAgent) -> None:
        src = "x = 1\n"
        findings = agent.analyze(_ctx(src))
        for f in findings:
            assert f.confidence == pytest.approx(0.65)

    def test_findings_category_quality(self, agent: ReviewGeneratorAgent) -> None:
        src = "x = 1\n"
        findings = agent.analyze(_ctx(src))
        for f in findings:
            assert f.category == "quality"

    def test_file_path_propagated(self, agent: ReviewGeneratorAgent) -> None:
        src = "x = 1\n"
        findings = agent.analyze(_ctx(src, fp="src/main.py"))
        for f in findings:
            assert f.file_path == "src/main.py"

    def test_mock_returns_three_findings(self, agent: ReviewGeneratorAgent) -> None:
        src = "def foo():\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 3

    def test_findings_have_tags(self, agent: ReviewGeneratorAgent) -> None:
        src = "x = 1\n"
        findings = agent.analyze(_ctx(src))
        for f in findings:
            assert "llm" in f.tags

    def test_go_code(self, agent: ReviewGeneratorAgent) -> None:
        src = 'package main\nfunc main() { fmt.Println("hi") }\n'
        assert len(agent.analyze(_ctx(src, "go", "main.go"))) >= 1

    def test_java_code(self, agent: ReviewGeneratorAgent) -> None:
        src = "public class Main {\n    public static void main(String[] args) {}\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "Main.java"))) >= 1


# ---------------------------------------------------------------------------
# Negative cases — empty or whitespace-only code should produce no findings
# ---------------------------------------------------------------------------


class TestNegativeCases:
    """Empty / blank input should yield zero findings."""

    def test_empty_string(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_only_spaces(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("     ")) == []

    def test_only_newlines(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("\n\n\n")) == []

    def test_only_tabs(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("\t\t\t")) == []

    def test_mixed_whitespace(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("  \n  \t  \n  ")) == []

    def test_single_newline(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("\n")) == []

    def test_carriage_return_only(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("\r\n")) == []

    def test_unicode_whitespace(self, agent: ReviewGeneratorAgent) -> None:
        # non-breaking spaces
        assert agent.analyze(_ctx("\u00a0\u00a0")) == []

    def test_space_tab_newline_combo(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("   \t\n  \t\n   ")) == []

    def test_empty_with_different_language(self, agent: ReviewGeneratorAgent) -> None:
        assert agent.analyze(_ctx("", "javascript", "app.js")) == []


# ---------------------------------------------------------------------------
# Metadata and structural tests
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert m.name == "review_generator"

    def test_version(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_languages_agnostic(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert m.languages == ["*"]

    def test_domains(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert "web_development" in m.domains
        assert "api_integration" in m.domains
        assert "enterprise_engineering" in m.domains

    def test_methodology(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert m.methodology == "llm_generative"

    def test_axis_type(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "agnostic"

    def test_model_required(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert m.model_required is True

    def test_cost(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == pytest.approx(5.0)

    def test_tags(self, agent: ReviewGeneratorAgent) -> None:
        m = agent.metadata()
        assert "llm" in m.tags
        assert "review" in m.tags
        assert "generative" in m.tags

    def test_explain(self, agent: ReviewGeneratorAgent) -> None:
        src = "x = 1\n"
        findings = agent.analyze(_ctx(src))
        assert findings
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0
