"""Tests for FixSuggestionAgent (mock mode)."""

from __future__ import annotations

import os

import pytest

# Ensure mock mode is enabled before importing the agent.
os.environ["FATHOM_LLM_MOCK"] = "1"

from fathom_sdk import CodeContext, CodeFix, Finding

from agents.cluster_11_llm_generative.fix_suggestion.agent import FixSuggestionAgent


@pytest.fixture
def agent() -> FixSuggestionAgent:
    return FixSuggestionAgent()


def _ctx(
    source: str,
    language: str = "python",
    fp: str = "test.py",
    historical_findings: list[Finding] | None = None,
) -> CodeContext:
    return CodeContext(
        source_code=source,
        language=language,
        file_path=fp,
        historical_findings=historical_findings,
    )


# ---------------------------------------------------------------------------
# Sample code snippets with known issues
# ---------------------------------------------------------------------------

_UNVALIDATED_INPUT = """\
def process(data):
    return data * 2
"""

_SQL_INJECTION = """\
def get_user(user_id):
    query = "SELECT * FROM users WHERE id=" + user_id
    return db.execute(query)
"""

_RESOURCE_LEAK = """\
def read_file(path):
    f = open(path)
    contents = f.read()
    return contents
"""

_BROAD_EXCEPT = """\
def do_work():
    try:
        risky_call()
    except:
        pass
"""

_NO_RETURN_TYPE = """\
def add(a, b):
    return a + b
"""

_HARDCODED_PASSWORD = """\
password = "s3cretP@ss!"
db.connect(password=password)
"""

_MUTABLE_DEFAULT = """\
def append_item(item, items=[]):
    items.append(item)
    return items
"""

_GLOBAL_STATE = """\
counter = 0

def increment():
    global counter
    counter += 1
"""

_UNUSED_IMPORT = """\
import os
import sys

def hello():
    print("hello")
"""

_MAGIC_NUMBER = """\
def calculate_price(quantity):
    return quantity * 9.99 + 5.0
"""


# ---------------------------------------------------------------------------
# Positive tests — code with known issues should produce findings
# ---------------------------------------------------------------------------

class TestPositiveCases:
    """Code snippets with known issues — the agent should produce findings."""

    def test_unvalidated_input(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        assert len(findings) >= 1

    def test_sql_injection(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_SQL_INJECTION))
        assert len(findings) >= 1

    def test_resource_leak(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_RESOURCE_LEAK))
        assert len(findings) >= 1

    def test_broad_except(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_BROAD_EXCEPT))
        assert len(findings) >= 1

    def test_no_return_type(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_NO_RETURN_TYPE))
        assert len(findings) >= 1

    def test_hardcoded_password(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_HARDCODED_PASSWORD))
        assert len(findings) >= 1

    def test_mutable_default(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_MUTABLE_DEFAULT))
        assert len(findings) >= 1

    def test_global_state(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_GLOBAL_STATE))
        assert len(findings) >= 1

    def test_unused_import(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNUSED_IMPORT))
        assert len(findings) >= 1

    def test_magic_number(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_MAGIC_NUMBER))
        assert len(findings) >= 1

    def test_findings_have_fix_available(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        for f in findings:
            assert f.fix_available is True

    def test_findings_have_suggested_fix(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_SQL_INJECTION))
        for f in findings:
            assert f.suggested_fix is not None
            assert isinstance(f.suggested_fix, CodeFix)

    def test_suggested_fix_has_all_fields(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_RESOURCE_LEAK))
        fix = findings[0].suggested_fix
        assert fix is not None
        assert fix.description
        assert fix.original_code
        assert fix.fixed_code
        assert fix.explanation

    def test_with_historical_findings(self, agent: FixSuggestionAgent) -> None:
        prior = [
            Finding(
                agent_name="test_agent",
                severity="high",
                category="security",
                title="SQL injection risk",
                description="String concatenation in query",
                file_path="test.py",
                line_start=2,
                line_end=2,
                confidence=0.90,
            )
        ]
        findings = agent.analyze(_ctx(_SQL_INJECTION, historical_findings=prior))
        assert len(findings) >= 1


# ---------------------------------------------------------------------------
# Negative tests — empty or clean code should produce no findings
# ---------------------------------------------------------------------------

class TestNegativeCases:
    """Clean or empty code should produce no findings."""

    def test_empty_string(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_empty_with_newlines(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("\n\n\n")) == []

    def test_single_comment(self, agent: FixSuggestionAgent) -> None:
        # The mock always returns fixes for non-empty source, but empty-ish
        # source still counts as non-empty.  Only truly blank triggers the
        # early return.
        # This test verifies the agent at least doesn't crash on comments.
        result = agent.analyze(_ctx("# just a comment\n"))
        # Mock mode always returns findings for non-empty code, which is fine.
        assert isinstance(result, list)

    def test_empty_tab_space(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("\t  \t  ")) == []

    def test_empty_file_path(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("", fp="empty.py")) == []

    def test_empty_javascript(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("", language="javascript", fp="test.js")) == []

    def test_empty_java(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("", language="java", fp="Test.java")) == []

    def test_empty_go(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("", language="go", fp="test.go")) == []

    def test_empty_typescript(self, agent: FixSuggestionAgent) -> None:
        assert agent.analyze(_ctx("", language="typescript", fp="test.ts")) == []


# ---------------------------------------------------------------------------
# Edge cases and metadata
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_metadata_name(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert m.name == "fix_suggestion"

    def test_metadata_version(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert m.languages == ["*"]

    def test_metadata_domains(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert "web_development" in m.domains
        assert "api_integration" in m.domains
        assert "enterprise_engineering" in m.domains

    def test_metadata_methodology(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert m.methodology == "llm_generative"

    def test_metadata_axis_type(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "agnostic"

    def test_metadata_tags(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert "llm" in m.tags
        assert "fix" in m.tags
        assert "generative" in m.tags

    def test_metadata_model_required(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert m.model_required is True

    def test_metadata_cost(self, agent: FixSuggestionAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 8.0

    def test_finding_severity_info(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        for f in findings:
            assert f.severity == "info"

    def test_finding_category_fix(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        for f in findings:
            assert f.category == "fix"

    def test_finding_confidence(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        for f in findings:
            assert f.confidence == 0.60

    def test_finding_agent_name(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_SQL_INJECTION))
        for f in findings:
            assert f.agent_name == "fix_suggestion"

    def test_finding_file_path(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_SQL_INJECTION, fp="src/db.py"))
        for f in findings:
            assert f.file_path == "src/db.py"

    def test_explain_returns_description(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_suggest_fix_returns_code_fix(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_RESOURCE_LEAK))
        fix = agent.suggest_fix(findings[0])
        assert fix is not None
        assert isinstance(fix, CodeFix)

    def test_mock_produces_three_findings(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        assert len(findings) == 3

    def test_finding_tags(self, agent: FixSuggestionAgent) -> None:
        findings = agent.analyze(_ctx(_UNVALIDATED_INPUT))
        for f in findings:
            assert "llm" in f.tags
            assert "fix" in f.tags

    def test_multiple_historical_findings(self, agent: FixSuggestionAgent) -> None:
        prior = [
            Finding(
                agent_name="agent_a",
                severity="high",
                category="security",
                title="Issue A",
                description="First issue",
                file_path="test.py",
                line_start=1,
                line_end=1,
                confidence=0.9,
            ),
            Finding(
                agent_name="agent_b",
                severity="medium",
                category="bug",
                title="Issue B",
                description="Second issue",
                file_path="test.py",
                line_start=3,
                line_end=3,
                confidence=0.85,
            ),
        ]
        findings = agent.analyze(
            _ctx(_SQL_INJECTION, historical_findings=prior)
        )
        assert len(findings) >= 1

    def test_no_historical_findings_still_works(
        self, agent: FixSuggestionAgent
    ) -> None:
        findings = agent.analyze(_ctx(_MUTABLE_DEFAULT, historical_findings=None))
        assert len(findings) >= 1

    def test_empty_historical_findings_list(
        self, agent: FixSuggestionAgent
    ) -> None:
        findings = agent.analyze(_ctx(_GLOBAL_STATE, historical_findings=[]))
        assert len(findings) >= 1
