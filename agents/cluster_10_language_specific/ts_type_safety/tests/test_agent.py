"""Tests for TsTypeSafetyAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_10_language_specific.ts_type_safety.agent import TsTypeSafetyAgent


@pytest.fixture
def agent() -> TsTypeSafetyAgent:
    return TsTypeSafetyAgent()


def _ctx(source: str, language: str = "typescript", fp: str = "test.ts") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ---------------------------------------------------------------------------
# Positive cases (10+)
# ---------------------------------------------------------------------------


class TestPositiveCases:
    def test_any_annotation_abuse(self, agent: TsTypeSafetyAgent) -> None:
        """More than 3 `: any` annotations should trigger a finding."""
        src = (
            "let a: any = 1;\n"
            "let b: any = 'x';\n"
            "let c: any = true;\n"
            "let d: any = null;\n"
        )
        findings = agent.analyze(_ctx(src))
        any_findings = [f for f in findings if "any" in f.title.lower() and "annotation" in f.title.lower()]
        assert len(any_findings) >= 1
        assert any_findings[0].severity == "medium"

    def test_as_any_cast(self, agent: TsTypeSafetyAgent) -> None:
        """Each `as any` cast should produce a high-severity finding."""
        src = "const x = someValue as any;\n"
        findings = agent.analyze(_ctx(src))
        as_any = [f for f in findings if "as any" in f.title.lower()]
        assert len(as_any) >= 1
        assert as_any[0].severity == "high"

    def test_multiple_as_any_casts(self, agent: TsTypeSafetyAgent) -> None:
        """Multiple `as any` casts should produce multiple findings."""
        src = (
            "const x = foo as any;\n"
            "const y = bar as any;\n"
            "const z = baz as any;\n"
        )
        findings = agent.analyze(_ctx(src))
        as_any = [f for f in findings if "as any" in f.title.lower()]
        assert len(as_any) == 3

    def test_ts_ignore(self, agent: TsTypeSafetyAgent) -> None:
        """@ts-ignore should be flagged."""
        src = "// @ts-ignore\nconst x: number = 'not a number';\n"
        findings = agent.analyze(_ctx(src))
        ignore_findings = [f for f in findings if "@ts-ignore" in f.title]
        assert len(ignore_findings) >= 1
        assert ignore_findings[0].severity == "medium"

    def test_ts_nocheck(self, agent: TsTypeSafetyAgent) -> None:
        """@ts-nocheck should be flagged."""
        src = "// @ts-nocheck\nlet a = 'hello';\n"
        findings = agent.analyze(_ctx(src))
        nocheck_findings = [f for f in findings if "@ts-nocheck" in f.title]
        assert len(nocheck_findings) >= 1
        assert nocheck_findings[0].severity == "medium"

    def test_non_null_assertion_overuse(self, agent: TsTypeSafetyAgent) -> None:
        """More than 5 non-null assertions should trigger a finding."""
        src = (
            "const a = obj1!.prop;\n"
            "const b = obj2!.prop;\n"
            "const c = obj3!.prop;\n"
            "const d = obj4!.prop;\n"
            "const e = obj5!.prop;\n"
            "const f = obj6!.prop;\n"
        )
        findings = agent.analyze(_ctx(src))
        nna = [f for f in findings if "non-null" in f.title.lower()]
        assert len(nna) >= 1
        assert nna[0].severity == "low"

    def test_type_assertion_chain(self, agent: TsTypeSafetyAgent) -> None:
        """Chained type assertions should be flagged as high severity."""
        src = "const x = (value as unknown) as string;\n"
        findings = agent.analyze(_ctx(src))
        chain = [f for f in findings if "assertion chain" in f.title.lower()]
        assert len(chain) >= 1
        assert chain[0].severity == "high"

    def test_as_any_with_surrounding_code(self, agent: TsTypeSafetyAgent) -> None:
        """`as any` in a larger expression should still be caught."""
        src = "return (response.data as any).items.map(fn);\n"
        findings = agent.analyze(_ctx(src))
        as_any = [f for f in findings if "as any" in f.title.lower()]
        assert len(as_any) >= 1

    def test_any_annotation_at_threshold_boundary(self, agent: TsTypeSafetyAgent) -> None:
        """Exactly 4 `: any` annotations (above threshold of 3) should trigger."""
        src = (
            "function f(a: any, b: any, c: any): any {\n"
            "  return a;\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        any_findings = [f for f in findings if "any" in f.title.lower() and "annotation" in f.title.lower()]
        assert len(any_findings) >= 1

    def test_ts_ignore_inline(self, agent: TsTypeSafetyAgent) -> None:
        """@ts-ignore on the same line as code should be caught."""
        src = "const x = bad; // @ts-ignore\n"
        findings = agent.analyze(_ctx(src))
        ignore_findings = [f for f in findings if "@ts-ignore" in f.title]
        assert len(ignore_findings) >= 1

    def test_confidence_value(self, agent: TsTypeSafetyAgent) -> None:
        """All findings should have confidence 0.88."""
        src = "const x = foo as any;\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        for f in findings:
            assert f.confidence == 0.88

    def test_category_is_quality(self, agent: TsTypeSafetyAgent) -> None:
        """All findings should have category 'quality'."""
        src = "const x = foo as any;\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        for f in findings:
            assert f.category == "quality"


# ---------------------------------------------------------------------------
# Negative cases (10+)
# ---------------------------------------------------------------------------


class TestNegativeCases:
    def test_proper_types(self, agent: TsTypeSafetyAgent) -> None:
        """Properly typed code should produce no findings."""
        src = (
            "const x: string = 'hello';\n"
            "const y: number = 42;\n"
            "function add(a: number, b: number): number {\n"
            "  return a + b;\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_no_any(self, agent: TsTypeSafetyAgent) -> None:
        """Code without any type-safety issues should produce no findings."""
        src = (
            "interface User {\n"
            "  name: string;\n"
            "  age: number;\n"
            "}\n"
            "const user: User = { name: 'Alice', age: 30 };\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_no_ts_ignore(self, agent: TsTypeSafetyAgent) -> None:
        """Code without @ts-ignore should not flag that rule."""
        src = "const x: number = 42;\n"
        findings = agent.analyze(_ctx(src))
        ignore_findings = [f for f in findings if "ts-ignore" in f.title.lower()]
        assert len(ignore_findings) == 0

    def test_non_typescript_language(self, agent: TsTypeSafetyAgent) -> None:
        """Non-TypeScript code should return empty findings."""
        src = "let x: any = 1; // @ts-ignore\n"
        ctx = CodeContext(source_code=src, language="javascript", file_path="test.js")
        assert agent.analyze(ctx) == []

    def test_python_language(self, agent: TsTypeSafetyAgent) -> None:
        """Python code should return empty findings."""
        src = "x: any = 1\n"
        ctx = CodeContext(source_code=src, language="python", file_path="test.py")
        assert agent.analyze(ctx) == []

    def test_empty_source(self, agent: TsTypeSafetyAgent) -> None:
        """Empty source should return no findings."""
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: TsTypeSafetyAgent) -> None:
        """Whitespace-only source should return no findings."""
        assert agent.analyze(_ctx("   \n\n  \n")) == []

    def test_few_any_annotations_below_threshold(self, agent: TsTypeSafetyAgent) -> None:
        """Exactly 3 `: any` annotations (at threshold) should NOT trigger."""
        src = (
            "let a: any = 1;\n"
            "let b: any = 2;\n"
            "let c: any = 3;\n"
        )
        findings = agent.analyze(_ctx(src))
        any_findings = [f for f in findings if "annotation" in f.title.lower()]
        assert len(any_findings) == 0

    def test_ts_expect_error_not_flagged(self, agent: TsTypeSafetyAgent) -> None:
        """@ts-expect-error (the preferred alternative) should NOT be flagged."""
        src = "// @ts-expect-error: intentional for test\nconst x = bad;\n"
        findings = agent.analyze(_ctx(src))
        ts_findings = [f for f in findings if "ts-ignore" in f.title.lower() or "ts-nocheck" in f.title.lower()]
        assert len(ts_findings) == 0

    def test_non_null_below_threshold(self, agent: TsTypeSafetyAgent) -> None:
        """5 or fewer non-null assertions should NOT trigger."""
        src = (
            "const a = obj1!.prop;\n"
            "const b = obj2!.prop;\n"
            "const c = obj3!.prop;\n"
            "const d = obj4!.prop;\n"
            "const e = obj5!.prop;\n"
        )
        findings = agent.analyze(_ctx(src))
        nna = [f for f in findings if "non-null" in f.title.lower()]
        assert len(nna) == 0

    def test_as_keyword_in_import(self, agent: TsTypeSafetyAgent) -> None:
        """The `as` keyword in imports should not trigger `as any` detection."""
        src = "import * as React from 'react';\n"
        findings = agent.analyze(_ctx(src))
        as_any = [f for f in findings if "as any" in f.title.lower()]
        assert len(as_any) == 0

    def test_commented_out_any(self, agent: TsTypeSafetyAgent) -> None:
        """Commented-out code should not be flagged for `as any`."""
        src = "// const x = foo as any;\n"
        findings = agent.analyze(_ctx(src))
        as_any = [f for f in findings if "as any" in f.title.lower()]
        assert len(as_any) == 0


# ---------------------------------------------------------------------------
# Edge cases & metadata
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_metadata_name(self, agent: TsTypeSafetyAgent) -> None:
        m = agent.metadata()
        assert m.name == "ts_type_safety"

    def test_metadata_version(self, agent: TsTypeSafetyAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages(self, agent: TsTypeSafetyAgent) -> None:
        m = agent.metadata()
        assert m.languages == ["typescript"]

    def test_metadata_axis_type(self, agent: TsTypeSafetyAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "critical"

    def test_metadata_tags(self, agent: TsTypeSafetyAgent) -> None:
        m = agent.metadata()
        assert "typescript" in m.tags
        assert "type-safety" in m.tags
        assert "any" in m.tags

    def test_metadata_model_not_required(self, agent: TsTypeSafetyAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: TsTypeSafetyAgent) -> None:
        src = "const x = foo as any;\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0
