"""Tests for PythonTypeAnnotationAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_10_language_specific.python_type_annotation.agent import (
    PythonTypeAnnotationAgent,
)


@pytest.fixture
def agent() -> PythonTypeAnnotationAgent:
    return PythonTypeAnnotationAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ======================================================================
# Positive cases — the agent SHOULD produce findings
# ======================================================================


class TestPositiveCases:
    # --- 1. Missing return type annotation ---

    def test_missing_return_type(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def greet(name: str):\n    return f'hello {name}'\n"
        findings = agent.analyze(_ctx(src))
        titles = [f.title for f in findings]
        assert any("Missing return type" in t for t in titles)

    def test_missing_return_type_severity(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def greet(name: str):\n    return f'hello {name}'\n"
        findings = [f for f in agent.analyze(_ctx(src)) if "Missing return type" in f.title]
        assert findings[0].severity == "low"

    # --- 2. Missing parameter type annotation ---

    def test_missing_param_annotation(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def process(data) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        titles = [f.title for f in findings]
        assert any("Missing type annotation for parameter 'data'" in t for t in titles)

    def test_multiple_missing_params(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def add(a, b) -> int:\n    return a + b\n"
        findings = agent.analyze(_ctx(src))
        param_findings = [f for f in findings if "Missing type annotation for parameter" in f.title]
        assert len(param_findings) == 2

    def test_missing_param_severity(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def process(data) -> None:\n    pass\n"
        findings = [f for f in agent.analyze(_ctx(src)) if "Missing type annotation for parameter" in f.title]
        assert findings[0].severity == "low"

    # --- 3. Bare container annotations ---

    def test_bare_dict_param(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def process(data: dict) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        titles = [f.title for f in findings]
        assert any("Bare 'dict'" in t for t in titles)

    def test_bare_list_param(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def process(items: list) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        titles = [f.title for f in findings]
        assert any("Bare 'list'" in t for t in titles)

    def test_bare_tuple_param(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def process(items: tuple) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        titles = [f.title for f in findings]
        assert any("Bare 'tuple'" in t for t in titles)

    def test_bare_dict_return(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def get_config() -> dict:\n    return {}\n"
        findings = agent.analyze(_ctx(src))
        titles = [f.title for f in findings]
        assert any("Bare 'dict' return annotation" in t for t in titles)

    def test_bare_list_return(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def get_items() -> list:\n    return []\n"
        findings = agent.analyze(_ctx(src))
        titles = [f.title for f in findings]
        assert any("Bare 'list' return annotation" in t for t in titles)

    # --- 4. Optional misuse ---

    def test_optional_misuse_no_annotation(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def fetch(url: str, timeout=None) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        optional_findings = [f for f in findings if "defaults to None" in f.title]
        assert len(optional_findings) >= 1

    def test_optional_misuse_wrong_annotation(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def fetch(url: str, timeout: int = None) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        optional_findings = [f for f in findings if "defaults to None" in f.title]
        assert len(optional_findings) >= 1
        assert optional_findings[0].severity == "medium"

    def test_optional_misuse_str_default_none(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def greet(name: str = None) -> str:\n    return name or 'world'\n"
        findings = agent.analyze(_ctx(src))
        optional_findings = [f for f in findings if "defaults to None" in f.title]
        assert len(optional_findings) >= 1

    # --- 5. Overuse of Any ---

    def test_any_overuse(self, agent: PythonTypeAnnotationAgent) -> None:
        src = (
            "from typing import Any\n"
            "def a(x: Any) -> Any:\n    pass\n"
            "def b(y: Any) -> Any:\n    pass\n"
        )
        findings = agent.analyze(_ctx(src))
        any_findings = [f for f in findings if "Overuse of 'Any'" in f.title]
        assert len(any_findings) == 1

    def test_any_overuse_severity(self, agent: PythonTypeAnnotationAgent) -> None:
        src = (
            "from typing import Any\n"
            "def a(x: Any) -> Any:\n    pass\n"
            "def b(y: Any) -> Any:\n    pass\n"
        )
        any_findings = [f for f in agent.analyze(_ctx(src)) if "Overuse of 'Any'" in f.title]
        assert any_findings[0].severity == "low"

    # --- Combined ---

    def test_function_no_annotations_at_all(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def compute(x, y, z):\n    return x + y + z\n"
        findings = agent.analyze(_ctx(src))
        # Should have: 1 missing return type + 3 missing params = 4
        assert len(findings) >= 4


# ======================================================================
# Negative cases — the agent should NOT produce findings
# ======================================================================


class TestNegativeCases:
    def test_fully_annotated(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def add(a: int, b: int) -> int:\n    return a + b\n"
        assert agent.analyze(_ctx(src)) == []

    def test_fully_annotated_complex(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def process(data: dict[str, int], items: list[str]) -> tuple[int, str]:\n    pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_optional_annotation_correct(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "from typing import Optional\ndef fetch(url: str, timeout: Optional[int] = None) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        optional_findings = [f for f in findings if "defaults to None" in f.title]
        assert optional_findings == []

    def test_union_none_correct(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def fetch(url: str, timeout: int | None = None) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        optional_findings = [f for f in findings if "defaults to None" in f.title]
        assert optional_findings == []

    def test_parameterised_dict(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def get_config() -> dict[str, str]:\n    return {}\n"
        assert agent.analyze(_ctx(src)) == []

    def test_parameterised_list(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def get_items() -> list[int]:\n    return []\n"
        assert agent.analyze(_ctx(src)) == []

    def test_non_python_returns_empty(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "function greet(name) { return 'hello'; }\n"
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_empty_source(self, agent: PythonTypeAnnotationAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: PythonTypeAnnotationAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_no_functions(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "x = 1\ny = 'hello'\nprint(x, y)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_self_cls_ignored(self, agent: PythonTypeAnnotationAgent) -> None:
        src = (
            "class Foo:\n"
            "    def method(self, x: int) -> int:\n"
            "        return x\n"
            "    @classmethod\n"
            "    def create(cls, val: str) -> 'Foo':\n"
            "        return cls()\n"
        )
        findings = agent.analyze(_ctx(src))
        param_findings = [f for f in findings if "Missing type annotation for parameter" in f.title]
        assert param_findings == []

    def test_any_under_threshold(self, agent: PythonTypeAnnotationAgent) -> None:
        src = (
            "from typing import Any\n"
            "def process(data: Any) -> int:\n    return 1\n"
        )
        findings = agent.analyze(_ctx(src))
        any_findings = [f for f in findings if "Overuse of 'Any'" in f.title]
        assert any_findings == []


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    def test_metadata(self, agent: PythonTypeAnnotationAgent) -> None:
        m = agent.metadata()
        assert m.name == "python_type_annotation"
        assert m.version == "0.1.0"
        assert m.languages == ["python"]
        assert m.axis_type == "critical"
        assert m.methodology == "language_specific"
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def greet(name: str):\n    return f'hello {name}'\n"
        findings = agent.analyze(_ctx(src))
        explanation = agent.explain(findings[0])
        assert len(explanation) > 20
        assert "type annotations" in explanation.lower() or "Type annotations" in explanation

    def test_finding_fields(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def process(data) -> None:\n    pass\n"
        findings = agent.analyze(_ctx(src))
        param_f = [f for f in findings if "Missing type annotation for parameter" in f.title][0]
        assert param_f.agent_name == "python_type_annotation"
        assert param_f.category == "quality"
        assert param_f.confidence == 0.80
        assert param_f.file_path == "test.py"

    def test_confidence_value(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "def greet(name):\n    pass\n"
        for f in agent.analyze(_ctx(src)):
            assert f.confidence == 0.80

    def test_decorated_function(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "@decorator\ndef greet(name: str):\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert any("Missing return type" in f.title for f in findings)

    def test_async_function(self, agent: PythonTypeAnnotationAgent) -> None:
        src = "async def fetch(url):\n    pass\n"
        findings = agent.analyze(_ctx(src))
        assert any("Missing return type" in f.title for f in findings)
        assert any("Missing type annotation for parameter 'url'" in f.title for f in findings)
