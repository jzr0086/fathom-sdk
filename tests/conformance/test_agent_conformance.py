"""Agent conformance test suite.

Discovers all registered agents and verifies they implement
the BaseReviewAgent interface correctly.
"""

from __future__ import annotations

import pytest

from fathom_sdk.agent.base import BaseReviewAgent
from fathom_sdk.agent.registry import AgentRegistry
from fathom_sdk.context.code_context import CodeContext
from fathom_sdk.schema.finding import CodeFix, Finding
from fathom_sdk.schema.metadata import AgentMetadata


@pytest.fixture
def discovered_registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.discover()
    return reg


@pytest.fixture
def agent_classes(discovered_registry: AgentRegistry) -> list[type[BaseReviewAgent]]:
    return list(discovered_registry.get_all().values())


@pytest.fixture
def simple_context() -> CodeContext:
    return CodeContext(
        source_code="x = 1\n",
        language="python",
        file_path="test.py",
    )


@pytest.fixture
def empty_context() -> CodeContext:
    return CodeContext(
        source_code="",
        language="python",
        file_path="empty.py",
    )


class TestAgentConformance:
    """Run against all discovered agents to verify interface compliance."""

    def _get_agent_classes(self) -> list[type[BaseReviewAgent]]:
        reg = AgentRegistry()
        reg.discover()
        return list(reg.get_all().values())

    def test_metadata_returns_valid_agent_metadata(self) -> None:
        for agent_class in self._get_agent_classes():
            agent = agent_class()
            meta = agent.metadata()
            assert isinstance(meta, AgentMetadata), (
                f"{agent_class.__name__}.metadata() must return AgentMetadata"
            )
            assert meta.name, f"{agent_class.__name__} must have a non-empty name"
            assert meta.version, f"{agent_class.__name__} must have a version"
            assert meta.languages, f"{agent_class.__name__} must declare languages"
            assert meta.methodology, f"{agent_class.__name__} must declare a methodology"
            assert meta.axis_type in ("agnostic", "aware", "critical")
            assert meta.estimated_cost_cents >= 0

    def test_analyze_returns_list_of_findings(self, simple_context: CodeContext) -> None:
        for agent_class in self._get_agent_classes():
            agent = agent_class()
            findings = agent.analyze(simple_context)
            assert isinstance(findings, list), (
                f"{agent_class.__name__}.analyze() must return a list"
            )
            for f in findings:
                assert isinstance(f, Finding), (
                    f"{agent_class.__name__}.analyze() must return Finding objects"
                )

    def test_analyze_handles_empty_source(self, empty_context: CodeContext) -> None:
        for agent_class in self._get_agent_classes():
            agent = agent_class()
            findings = agent.analyze(empty_context)
            assert isinstance(findings, list)

    def test_explain_returns_nonempty_string(self, simple_context: CodeContext) -> None:
        for agent_class in self._get_agent_classes():
            agent = agent_class()
            findings = agent.analyze(simple_context)
            if findings:
                explanation = agent.explain(findings[0])
                assert isinstance(explanation, str)
                assert len(explanation) > 0

    def test_suggest_fix_returns_none_or_codefix(self, simple_context: CodeContext) -> None:
        for agent_class in self._get_agent_classes():
            agent = agent_class()
            findings = agent.analyze(simple_context)
            for f in findings:
                fix = agent.suggest_fix(f)
                assert fix is None or isinstance(fix, CodeFix)

    def test_confidence_returns_valid_float(self, simple_context: CodeContext) -> None:
        for agent_class in self._get_agent_classes():
            agent = agent_class()
            findings = agent.analyze(simple_context)
            for f in findings:
                conf = agent.confidence(f)
                assert isinstance(conf, float)
                assert 0.0 <= conf <= 1.0
