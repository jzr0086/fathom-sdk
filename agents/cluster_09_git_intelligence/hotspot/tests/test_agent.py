"""Tests for HotspotAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext, Commit
from fathom_sdk.context.ast_parser import parse

from agents.cluster_09_git_intelligence.hotspot.agent import HotspotAgent


@pytest.fixture
def agent() -> HotspotAgent:
    return HotspotAgent()


def _ctx(
    source: str,
    language: str = "python",
    file_path: str = "app.py",
    commit_history: list[Commit] | None = None,
) -> CodeContext:
    """Build a CodeContext with optional commit history."""
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(
        source_code=source,
        language=language,
        file_path=file_path,
        ast=ast,
        commit_history=commit_history,
    )


def _commits(n: int, authors: int = 3) -> list[Commit]:
    """Generate *n* commits spread across *authors* distinct developers."""
    return [
        Commit(
            sha=f"abc{i:04d}",
            message="fix",
            author=f"dev{i % authors}",
            timestamp="2024-01-01",
        )
        for i in range(n)
    ]


def _complex_python(func_count: int = 5, branches_per_func: int = 4) -> str:
    """Generate Python source with *func_count* functions, each containing decision points."""
    parts: list[str] = []
    for f in range(func_count):
        lines = [f"def func_{f}(x):"]
        for b in range(branches_per_func):
            lines.append(f"    if x > {b}:")
            lines.append(f"        x += {b}")
        lines.append("    return x")
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"


def _simple_python(func_count: int = 2) -> str:
    """Generate simple Python source with no branching."""
    parts: list[str] = []
    for f in range(func_count):
        parts.append(f"def func_{f}():\n    return {f}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Positive cases (10+)
# ---------------------------------------------------------------------------


class TestPositiveCases:
    """Scenarios where the agent SHOULD emit findings."""

    def test_high_churn_complex_code(self, agent: HotspotAgent) -> None:
        """15 commits * 3 authors = churn 45; avg complexity ~5 => score ~225 => high."""
        src = _complex_python(func_count=5, branches_per_func=4)
        ctx = _ctx(src, commit_history=_commits(15, authors=3))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_moderate_churn_complex_code_medium(self, agent: HotspotAgent) -> None:
        """8 commits * 2 authors = churn 16; avg complexity ~3 => score ~48 => medium."""
        src = _complex_python(func_count=3, branches_per_func=2)
        ctx = _ctx(src, commit_history=_commits(8, authors=2))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        assert findings[0].severity == "medium"

    def test_low_severity_churn(self, agent: HotspotAgent) -> None:
        """11 commits * 1 author = churn 11; avg complexity ~2 => score ~22 => medium."""
        src = _complex_python(func_count=2, branches_per_func=1)
        ctx = _ctx(src, commit_history=_commits(11, authors=1))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        assert findings[0].severity in ("medium", "low")

    def test_many_authors_high_churn(self, agent: HotspotAgent) -> None:
        """20 commits * 5 authors = churn 100; even simple code crosses threshold."""
        src = _simple_python(func_count=1)
        ctx = _ctx(src, commit_history=_commits(20, authors=5))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        # churn_score = 100, avg_complexity = 1.0, hotspot = 100 => high
        assert findings[0].severity == "high"

    def test_finding_fields(self, agent: HotspotAgent) -> None:
        """Verify the structure of emitted findings."""
        src = _complex_python(func_count=5, branches_per_func=4)
        ctx = _ctx(src, file_path="src/core.py", commit_history=_commits(15, authors=3))
        f = agent.analyze(ctx)[0]
        assert f.agent_name == "hotspot"
        assert f.category == "maintenance"
        assert f.file_path == "src/core.py"
        assert f.confidence == 0.70
        assert f.line_start == 1
        assert "churn" in f.title.lower() or "hotspot" in f.title.lower()

    def test_javascript_high_churn(self, agent: HotspotAgent) -> None:
        """JavaScript with high churn and branching."""
        src = (
            "function handler(req) {\n"
            "    if (req.a) { return 1; }\n"
            "    if (req.b) { return 2; }\n"
            "    if (req.c) { return 3; }\n"
            "    if (req.d) { return 4; }\n"
            "    return 0;\n"
            "}\n"
        )
        ctx = _ctx(src, language="javascript", file_path="handler.js",
                    commit_history=_commits(20, authors=4))
        findings = agent.analyze(ctx)
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_java_high_churn(self, agent: HotspotAgent) -> None:
        """Java with high churn."""
        src = (
            "public class Service {\n"
            "    public void process(int x) {\n"
            "        if (x > 0) { x++; }\n"
            "        if (x > 1) { x++; }\n"
            "        if (x > 2) { x++; }\n"
            "    }\n"
            "}\n"
        )
        ctx = _ctx(src, language="java", file_path="Service.java",
                    commit_history=_commits(25, authors=5))
        findings = agent.analyze(ctx)
        assert len(findings) == 1

    def test_go_high_churn(self, agent: HotspotAgent) -> None:
        """Go with high churn and control flow."""
        src = (
            "package main\n\n"
            "func run(x int) int {\n"
            "    if x > 0 { x++ }\n"
            "    if x > 1 { x++ }\n"
            "    if x > 2 { x++ }\n"
            "    return x\n"
            "}\n"
        )
        ctx = _ctx(src, language="go", file_path="main.go",
                    commit_history=_commits(25, authors=5))
        findings = agent.analyze(ctx)
        assert len(findings) == 1

    def test_complexity_fallback_many_complex_funcs(self, agent: HotspotAgent) -> None:
        """No commit history, but highly complex file triggers complexity-based risk."""
        src = _complex_python(func_count=10, branches_per_func=5)
        ctx = _ctx(src)  # no commit_history
        findings = agent.analyze(ctx)
        # 10 funcs * avg_complexity ~6 = risk ~60 => high
        assert len(findings) == 1
        assert findings[0].severity == "high"

    def test_complexity_fallback_medium(self, agent: HotspotAgent) -> None:
        """No commit history, moderate complexity triggers medium severity."""
        src = _complex_python(func_count=5, branches_per_func=2)
        ctx = _ctx(src)  # no commit_history
        findings = agent.analyze(ctx)
        # 5 funcs * avg_complexity ~3 = risk ~15 => low (>10)
        assert len(findings) == 1
        assert findings[0].severity in ("medium", "low")

    def test_tags_present(self, agent: HotspotAgent) -> None:
        """Finding should include all expected tags."""
        src = _complex_python(func_count=5, branches_per_func=4)
        ctx = _ctx(src, commit_history=_commits(15, authors=3))
        f = agent.analyze(ctx)[0]
        assert "git" in f.tags
        assert "churn" in f.tags
        assert "hotspot" in f.tags
        assert "maintenance" in f.tags


# ---------------------------------------------------------------------------
# Negative cases (10+)
# ---------------------------------------------------------------------------


class TestNegativeCases:
    """Scenarios where the agent should NOT emit findings."""

    def test_no_history_simple_code(self, agent: HotspotAgent) -> None:
        """Simple code with no commit history -- risk too low."""
        src = _simple_python(func_count=2)
        ctx = _ctx(src)
        assert agent.analyze(ctx) == []

    def test_low_churn_simple_code(self, agent: HotspotAgent) -> None:
        """Only 3 commits * 1 author = churn 3 -- below threshold."""
        src = _simple_python(func_count=2)
        ctx = _ctx(src, commit_history=_commits(3, authors=1))
        assert agent.analyze(ctx) == []

    def test_low_churn_complex_code(self, agent: HotspotAgent) -> None:
        """Complex code but very low churn -- not a hotspot."""
        src = _complex_python(func_count=5, branches_per_func=4)
        ctx = _ctx(src, commit_history=_commits(2, authors=1))
        assert agent.analyze(ctx) == []

    def test_empty_source(self, agent: HotspotAgent) -> None:
        """Empty source file."""
        ctx = _ctx("")
        assert agent.analyze(ctx) == []

    def test_whitespace_only(self, agent: HotspotAgent) -> None:
        """Whitespace-only source file."""
        ctx = _ctx("   \n\n  \n")
        assert agent.analyze(ctx) == []

    def test_empty_commit_history(self, agent: HotspotAgent) -> None:
        """Explicit empty commit list behaves like absent history."""
        src = _simple_python(func_count=1)
        ctx = _ctx(src, commit_history=[])
        assert agent.analyze(ctx) == []

    def test_single_trivial_function_no_history(self, agent: HotspotAgent) -> None:
        """One trivial function, no history -- risk too low."""
        ctx = _ctx("def f():\n    return 1\n")
        assert agent.analyze(ctx) == []

    def test_no_functions_at_all(self, agent: HotspotAgent) -> None:
        """Module-level code only, no function definitions."""
        ctx = _ctx("x = 1\ny = 2\nprint(x + y)\n")
        assert agent.analyze(ctx) == []

    def test_no_functions_low_churn_with_history(self, agent: HotspotAgent) -> None:
        """Module-level code with low churn and no functions -- below threshold."""
        # 5 commits * 2 authors = churn 10, which is <= threshold
        ctx = _ctx("x = 1\ny = 2\n", commit_history=_commits(5, authors=2))
        assert agent.analyze(ctx) == []

    def test_low_churn_no_functions(self, agent: HotspotAgent) -> None:
        """Module-level code with low churn and no functions."""
        ctx = _ctx("x = 1\ny = 2\n", commit_history=_commits(5, authors=1))
        assert agent.analyze(ctx) == []

    def test_unsupported_language_no_history(self, agent: HotspotAgent) -> None:
        """Unsupported language without history returns no findings."""
        ctx = _ctx("(define (f x) (+ x 1))", language="scheme", file_path="t.scm")
        assert agent.analyze(ctx) == []

    def test_threshold_boundary_churn_exactly_10(self, agent: HotspotAgent) -> None:
        """Churn score exactly at threshold (10) should NOT fire (needs > 10)."""
        # 10 commits * 1 author = churn 10
        src = _simple_python(func_count=1)
        ctx = _ctx(src, commit_history=_commits(10, authors=1))
        assert agent.analyze(ctx) == []


# ---------------------------------------------------------------------------
# Edge cases and metadata
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_metadata(self, agent: HotspotAgent) -> None:
        m = agent.metadata()
        assert m.name == "hotspot"
        assert m.version == "0.1.0"
        assert m.languages == ["*"]
        assert m.axis_type == "agnostic"
        assert m.methodology == "git_intelligence"
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain(self, agent: HotspotAgent) -> None:
        src = _complex_python(func_count=5, branches_per_func=4)
        ctx = _ctx(src, commit_history=_commits(15, authors=3))
        explanation = agent.explain(agent.analyze(ctx)[0])
        assert len(explanation) > 20
        assert "hotspot" in explanation.lower() or "complex" in explanation.lower()

    def test_line_end_matches_file_length(self, agent: HotspotAgent) -> None:
        src = _complex_python(func_count=5, branches_per_func=4)
        ctx = _ctx(src, commit_history=_commits(15, authors=3))
        f = agent.analyze(ctx)[0]
        assert f.line_end == len(src.splitlines())

    def test_severity_for_score_none(self, agent: HotspotAgent) -> None:
        """Score at or below 10 returns None from the severity mapper."""
        assert agent._severity_for_score(10.0) is None
        assert agent._severity_for_score(5.0) is None
        assert agent._severity_for_score(0.0) is None

    def test_severity_for_score_low(self, agent: HotspotAgent) -> None:
        assert agent._severity_for_score(11.0) == "low"

    def test_severity_for_score_medium(self, agent: HotspotAgent) -> None:
        assert agent._severity_for_score(21.0) == "medium"

    def test_severity_for_score_high(self, agent: HotspotAgent) -> None:
        assert agent._severity_for_score(51.0) == "high"
