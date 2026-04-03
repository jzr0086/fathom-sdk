"""Tests for the AST Pattern Matching agent.

Covers all rule categories with 10+ positive (should detect) and
10+ negative (should NOT detect) cases.
"""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_01_static_analysis.ast_pattern_matching.agent import (
    AstPatternMatchingAgent,
)


def _ctx(source: str, language: str = "python", file_path: str = "test.py") -> CodeContext:
    """Build a minimal CodeContext with a pre-parsed AST."""
    root = parse(source, language)
    return CodeContext(
        source_code=source,
        language=language,
        file_path=file_path,
        ast=root,
    )


@pytest.fixture
def agent() -> AstPatternMatchingAgent:
    return AstPatternMatchingAgent()


# ===================================================================
# Metadata
# ===================================================================


class TestMetadata:
    def test_name_and_version(self, agent: AstPatternMatchingAgent):
        meta = agent.metadata()
        assert meta.name == "ast_pattern_matching"
        assert meta.version == "0.1.0"

    def test_languages(self, agent: AstPatternMatchingAgent):
        meta = agent.metadata()
        assert set(meta.languages) == {
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
        }

    def test_methodology(self, agent: AstPatternMatchingAgent):
        meta = agent.metadata()
        assert meta.methodology == "static_analysis"
        assert meta.axis_type == "aware"

    def test_no_model_required(self, agent: AstPatternMatchingAgent):
        meta = agent.metadata()
        assert meta.model_required is False
        assert meta.estimated_cost_cents == 0.0


# ===================================================================
# Positive tests — patterns that SHOULD be detected
# ===================================================================


class TestPositivePythonNoneComparison:
    """== None / != None should be flagged."""

    def test_eq_none(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("x = 1\nif x == None:\n    pass"))
        assert len(findings) >= 1
        assert any("is None" in f.title for f in findings)

    def test_ne_none(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x != None:\n    pass"))
        assert len(findings) >= 1
        assert any("is None" in f.title or "is not None" in f.title for f in findings)

    def test_none_eq_x(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if None == x:\n    pass"))
        assert len(findings) >= 1

    def test_severity_is_medium(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x == None:\n    pass"))
        none_findings = [f for f in findings if "None" in f.title]
        assert all(f.severity == "medium" for f in none_findings)


class TestPositivePythonBoolComparison:
    """== True / == False should be flagged."""

    def test_eq_true(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x == True:\n    pass"))
        assert len(findings) >= 1
        assert any("True" in f.description or "False" in f.description for f in findings)

    def test_eq_false(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x == False:\n    pass"))
        assert len(findings) >= 1

    def test_severity_is_low(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x == True:\n    pass"))
        bool_findings = [f for f in findings if "True" in f.description or "False" in f.description]
        assert all(f.severity == "low" for f in bool_findings)


class TestPositivePythonMutableDefaults:
    """Mutable default arguments should be flagged."""

    def test_empty_list_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x=[]):\n    pass"))
        assert len(findings) >= 1
        assert any("Mutable default" in f.title for f in findings)

    def test_empty_dict_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x={}):\n    pass"))
        assert len(findings) >= 1
        assert any("Mutable default" in f.title for f in findings)

    def test_set_call_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x=set()):\n    pass"))
        assert len(findings) >= 1

    def test_severity_is_medium(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x=[]):\n    pass"))
        mut_findings = [f for f in findings if "Mutable" in f.title]
        assert all(f.severity == "medium" for f in mut_findings)


class TestPositiveJSLooseEquality:
    """== / != in JS should be flagged (should use === / !==)."""

    def test_loose_eq(self, agent: AstPatternMatchingAgent):
        code = 'if (x == 1) { console.log("hi"); }'
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        assert len(findings) >= 1
        assert any("===" in f.title for f in findings)

    def test_loose_neq(self, agent: AstPatternMatchingAgent):
        code = 'if (x != null) { console.log("hi"); }'
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        assert len(findings) >= 1
        assert any("!==" in f.title or "===" in f.title for f in findings)


class TestPositiveJSVarDeclarations:
    """var declarations should be flagged."""

    def test_var_declaration(self, agent: AstPatternMatchingAgent):
        code = 'var x = 1;'
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        assert len(findings) >= 1
        assert any("var" in f.title.lower() for f in findings)

    def test_var_severity_low(self, agent: AstPatternMatchingAgent):
        code = 'var x = 1;'
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        var_findings = [f for f in findings if "var" in f.title.lower()]
        assert all(f.severity == "low" for f in var_findings)


class TestPositiveEmptyCatch:
    """Empty except/catch blocks should be flagged."""

    def test_python_empty_except(self, agent: AstPatternMatchingAgent):
        code = "try:\n    x = 1\nexcept:\n    pass"
        findings = agent.analyze(_ctx(code, "python"))
        assert len(findings) >= 1
        assert any("empty" in f.title.lower() or "catch" in f.title.lower() for f in findings)

    def test_js_empty_catch(self, agent: AstPatternMatchingAgent):
        code = "try { x = 1; } catch (e) { }"
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        assert len(findings) >= 1
        assert any("catch" in f.title.lower() or "empty" in f.title.lower() for f in findings)


# ===================================================================
# Negative tests — patterns that should NOT be detected
# ===================================================================


class TestNegativePythonNoneComparison:
    """'is None' / 'is not None' should NOT be flagged."""

    def test_is_none(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x is None:\n    pass"))
        none_findings = [f for f in findings if "is None" in f.title]
        assert len(none_findings) == 0

    def test_is_not_none(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x is not None:\n    pass"))
        none_findings = [f for f in findings if "is None" in f.title]
        assert len(none_findings) == 0


class TestNegativePythonBoolComparison:
    """Direct truthiness usage should NOT be flagged."""

    def test_truthiness(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x:\n    pass"))
        bool_findings = [f for f in findings if "True" in f.description or "False" in f.description]
        assert len(bool_findings) == 0

    def test_not_x(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if not x:\n    pass"))
        bool_findings = [f for f in findings if "True" in f.description or "False" in f.description]
        assert len(bool_findings) == 0


class TestNegativePythonMutableDefaults:
    """Immutable defaults should NOT be flagged."""

    def test_none_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x=None):\n    pass"))
        mut_findings = [f for f in findings if "Mutable" in f.title]
        assert len(mut_findings) == 0

    def test_int_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x=0):\n    pass"))
        mut_findings = [f for f in findings if "Mutable" in f.title]
        assert len(mut_findings) == 0

    def test_string_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx('def f(x="hello"):\n    pass'))
        mut_findings = [f for f in findings if "Mutable" in f.title]
        assert len(mut_findings) == 0

    def test_tuple_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x=(1, 2)):\n    pass"))
        mut_findings = [f for f in findings if "Mutable" in f.title]
        assert len(mut_findings) == 0

    def test_frozenset_default(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("def f(x=frozenset()):\n    pass"))
        mut_findings = [f for f in findings if "Mutable" in f.title]
        assert len(mut_findings) == 0


class TestNegativeJSStrictEquality:
    """=== / !== should NOT be flagged."""

    def test_strict_eq(self, agent: AstPatternMatchingAgent):
        code = 'if (x === 1) { console.log("hi"); }'
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        eq_findings = [f for f in findings if "===" in f.title]
        assert len(eq_findings) == 0

    def test_strict_neq(self, agent: AstPatternMatchingAgent):
        code = 'if (x !== null) { console.log("hi"); }'
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        eq_findings = [f for f in findings if "===" in f.title or "!==" in f.title]
        assert len(eq_findings) == 0


class TestNegativeJSLetConst:
    """let / const should NOT be flagged."""

    def test_let(self, agent: AstPatternMatchingAgent):
        code = "let x = 1;"
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        var_findings = [f for f in findings if "var" in f.title.lower()]
        assert len(var_findings) == 0

    def test_const(self, agent: AstPatternMatchingAgent):
        code = "const x = 1;"
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        var_findings = [f for f in findings if "var" in f.title.lower()]
        assert len(var_findings) == 0


class TestNegativeEmptySource:
    """Empty or whitespace-only source should produce no findings."""

    def test_empty(self, agent: AstPatternMatchingAgent):
        ctx = CodeContext(source_code="", language="python", file_path="empty.py")
        assert agent.analyze(ctx) == []

    def test_whitespace(self, agent: AstPatternMatchingAgent):
        ctx = CodeContext(source_code="   \n\n  ", language="python", file_path="ws.py")
        assert agent.analyze(ctx) == []


class TestNegativeNonEmptyCatch:
    """Catch blocks with meaningful content should NOT be flagged."""

    def test_python_except_with_logging(self, agent: AstPatternMatchingAgent):
        code = "try:\n    x = 1\nexcept Exception as e:\n    print(e)"
        findings = agent.analyze(_ctx(code, "python"))
        catch_findings = [f for f in findings if "catch" in f.title.lower() or "empty" in f.title.lower()]
        assert len(catch_findings) == 0

    def test_js_catch_with_body(self, agent: AstPatternMatchingAgent):
        code = "try { x = 1; } catch (e) { console.log(e); }"
        findings = agent.analyze(_ctx(code, "javascript", "test.js"))
        catch_findings = [f for f in findings if "catch" in f.title.lower() or "empty" in f.title.lower()]
        assert len(catch_findings) == 0


# ===================================================================
# Edge cases and explain()
# ===================================================================


class TestEdgeCases:
    def test_unsupported_language_returns_empty(self, agent: AstPatternMatchingAgent):
        """Go has no rules configured — should return empty list."""
        code = 'package main\nfunc main() {\n\tx := 1\n\t_ = x\n}'
        findings = agent.analyze(_ctx(code, "go", "main.go"))
        assert findings == []

    def test_confidence_is_090(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x == None:\n    pass"))
        assert all(f.confidence == 0.90 for f in findings)

    def test_explain_returns_string(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x == None:\n    pass"))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_suggest_fix_returns_none(self, agent: AstPatternMatchingAgent):
        findings = agent.analyze(_ctx("if x == None:\n    pass"))
        assert len(findings) >= 1
        assert agent.suggest_fix(findings[0]) is None

    def test_typescript_var_flagged(self, agent: AstPatternMatchingAgent):
        code = "var x: number = 1;"
        findings = agent.analyze(_ctx(code, "typescript", "test.ts"))
        var_findings = [f for f in findings if "var" in f.title.lower()]
        assert len(var_findings) >= 1

    def test_multiple_issues_in_one_file(self, agent: AstPatternMatchingAgent):
        code = "def f(x=[]):\n    if x == None:\n        pass"
        findings = agent.analyze(_ctx(code, "python"))
        # Should have at least a mutable-default AND a None-comparison finding
        titles = [f.title for f in findings]
        assert any("Mutable" in t for t in titles)
        assert any("is None" in t for t in titles)
