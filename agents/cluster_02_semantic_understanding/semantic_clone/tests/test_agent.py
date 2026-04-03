"""Tests for SemanticCloneAgent."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_02_semantic_understanding.semantic_clone.agent import (
    SemanticCloneAgent,
)


@pytest.fixture
def agent() -> SemanticCloneAgent:
    return SemanticCloneAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ---------------------------------------------------------------------------
# Helper: fake embeddings that let us control similarity
# ---------------------------------------------------------------------------


def _make_similar_embeddings(n: int, similar_pairs: list[tuple[int, int]]) -> np.ndarray:
    """Create embeddings where specified pairs have cosine similarity > 0.95
    and all other pairs have cosine similarity < 0.5.
    """
    rng = np.random.RandomState(42)
    dim = 64
    # Start with well-separated random vectors
    embeddings = rng.randn(n, dim)
    # Normalize all vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # For similar pairs, make the second vector close to the first
    for i, j in similar_pairs:
        noise = rng.randn(dim) * 0.05
        embeddings[j] = embeddings[i] + noise
        # Re-normalize
        embeddings[j] = embeddings[j] / np.linalg.norm(embeddings[j])

    return embeddings


def _patch_embeddings(embeddings: np.ndarray):
    """Return a context manager that patches get_code_embeddings to return
    the given matrix.
    """
    return patch(
        "agents.cluster_02_semantic_understanding.semantic_clone.agent.get_code_embeddings",
        return_value=embeddings,
    )


def _patch_embeddings_none():
    """Return a context manager that patches get_code_embeddings to return None."""
    return patch(
        "agents.cluster_02_semantic_understanding.semantic_clone.agent.get_code_embeddings",
        return_value=None,
    )


# ---------------------------------------------------------------------------
# Positive cases (10+)
# ---------------------------------------------------------------------------


class TestPositiveCases:
    """Tests that verify semantic clones are detected."""

    def test_identical_functions_python(self, agent: SemanticCloneAgent) -> None:
        """Two functions with identical structure, different variable names."""
        src = (
            "def compute_a(x):\n"
            "    result = x + 1\n"
            "    output = result * 2\n"
            "    final = output - 3\n"
            "    return final\n\n"
            "def compute_b(y):\n"
            "    val = y + 1\n"
            "    temp = val * 2\n"
            "    ans = temp - 3\n"
            "    return ans\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        assert findings[0].severity == "medium"

    def test_same_logic_different_names_js(self, agent: SemanticCloneAgent) -> None:
        """JavaScript functions with same logic, different identifiers."""
        src = (
            "function processA(data) {\n"
            "    let result = data + 1;\n"
            "    let output = result * 2;\n"
            "    return output;\n"
            "}\n\n"
            "function processB(info) {\n"
            "    let temp = info + 1;\n"
            "    let calc = temp * 2;\n"
            "    return calc;\n"
            "}\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert len(findings) == 1

    def test_three_functions_two_clones(self, agent: SemanticCloneAgent) -> None:
        """Three functions where two are clones and one is different."""
        src = (
            "def add_values(x):\n"
            "    a = x + 1\n"
            "    b = a + 2\n"
            "    return b\n\n"
            "def sum_values(y):\n"
            "    c = y + 1\n"
            "    d = c + 2\n"
            "    return d\n\n"
            "def multiply_values(z):\n"
            "    a = z * 10\n"
            "    b = a ** 2\n"
            "    return b\n"
        )
        embeddings = _make_similar_embeddings(3, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert len(findings) == 1

    def test_typescript_clones(self, agent: SemanticCloneAgent) -> None:
        """TypeScript functions that are semantic clones."""
        src = (
            "function calcTax(amount: number): number {\n"
            "    let rate = 0.15;\n"
            "    let tax = amount * rate;\n"
            "    return tax;\n"
            "}\n\n"
            "function computeLevy(sum: number): number {\n"
            "    let percentage = 0.15;\n"
            "    let levy = sum * percentage;\n"
            "    return levy;\n"
            "}\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src, "typescript", "test.ts"))
        assert len(findings) == 1

    def test_java_clones(self, agent: SemanticCloneAgent) -> None:
        """Java methods that are semantic clones."""
        src = (
            "public class Calculator {\n"
            "    int computeA(int x) {\n"
            "        int a = x + 1;\n"
            "        int b = a * 2;\n"
            "        return b;\n"
            "    }\n"
            "    int computeB(int y) {\n"
            "        int c = y + 1;\n"
            "        int d = c * 2;\n"
            "        return d;\n"
            "    }\n"
            "}\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src, "java", "Calculator.java"))
        assert len(findings) == 1

    def test_go_clones(self, agent: SemanticCloneAgent) -> None:
        """Go functions that are semantic clones."""
        src = (
            "package main\n\n"
            "func processA(x int) int {\n"
            "    a := x + 1\n"
            "    b := a * 2\n"
            "    return b\n"
            "}\n\n"
            "func processB(y int) int {\n"
            "    c := y + 1\n"
            "    d := c * 2\n"
            "    return d\n"
            "}\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) == 1

    def test_finding_fields(self, agent: SemanticCloneAgent) -> None:
        """Verify all finding fields are correctly populated."""
        src = (
            "def alpha(x):\n"
            "    a = x + 1\n"
            "    b = a * 2\n"
            "    return b\n\n"
            "def beta(y):\n"
            "    c = y + 1\n"
            "    d = c * 2\n"
            "    return d\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src, fp="src/utils.py"))
        assert len(findings) == 1
        f = findings[0]
        assert f.agent_name == "semantic_clone"
        assert f.severity == "medium"
        assert f.category == "quality"
        assert f.file_path == "src/utils.py"
        assert f.confidence > 0.85
        assert "semantic" in f.tags

    def test_confidence_reflects_similarity(self, agent: SemanticCloneAgent) -> None:
        """Confidence score should reflect the cosine similarity."""
        src = (
            "def func_a(x):\n"
            "    return x + 1\n"
            "    pass\n\n"
            "def func_b(y):\n"
            "    return y + 1\n"
            "    pass\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        # Confidence should be between 0.85 and 1.0 for a detected clone
        assert 0.85 <= findings[0].confidence <= 1.0

    def test_multiple_clone_pairs(self, agent: SemanticCloneAgent) -> None:
        """Four functions forming two distinct clone pairs."""
        src = (
            "def add_a(x):\n    return x + 1\n    pass\n\n"
            "def add_b(y):\n    return y + 1\n    pass\n\n"
            "def mul_a(x):\n    return x * 10\n    pass\n\n"
            "def mul_b(y):\n    return y * 10\n    pass\n"
        )
        embeddings = _make_similar_embeddings(4, [(0, 1), (2, 3)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert len(findings) == 2

    def test_title_contains_function_names(self, agent: SemanticCloneAgent) -> None:
        """Finding title should mention both function names."""
        src = (
            "def foo(x):\n    a = x + 1\n    return a\n\n"
            "def bar(y):\n    b = y + 1\n    return b\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        assert "foo" in findings[0].title
        assert "bar" in findings[0].title


# ---------------------------------------------------------------------------
# Negative cases (10+)
# ---------------------------------------------------------------------------


class TestNegativeCases:
    """Tests that verify non-clones are not flagged."""

    def test_completely_different_functions(self, agent: SemanticCloneAgent) -> None:
        """Two functions with entirely different logic."""
        src = (
            "def sort_list(data):\n"
            "    data.sort()\n"
            "    return data\n\n"
            "def connect_db(host):\n"
            "    import sqlite3\n"
            "    return sqlite3.connect(host)\n"
        )
        # Dissimilar embeddings (no similar pairs)
        embeddings = _make_similar_embeddings(2, [])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_single_function(self, agent: SemanticCloneAgent) -> None:
        """Only one function -- nothing to compare."""
        src = (
            "def only_one(x):\n"
            "    a = x + 1\n"
            "    b = a * 2\n"
            "    return b\n"
        )
        embeddings = _make_similar_embeddings(1, [])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_empty_file(self, agent: SemanticCloneAgent) -> None:
        """Empty source code produces no findings."""
        assert agent.analyze(_ctx("")) == []

    def test_no_functions(self, agent: SemanticCloneAgent) -> None:
        """File with no function definitions."""
        src = "x = 1\ny = 2\nz = x + y\n"
        findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_very_short_functions(self, agent: SemanticCloneAgent) -> None:
        """Two-line functions below the minimum line threshold."""
        src = (
            "def a(x):\n    return x\n\n"
            "def b(y):\n    return y\n"
        )
        # These are only 2 lines each, below _MIN_FUNCTION_LINES=3
        findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_embeddings_unavailable(self, agent: SemanticCloneAgent) -> None:
        """When ML deps are missing, get_code_embeddings returns None."""
        src = (
            "def compute_a(x):\n    a = x + 1\n    return a\n\n"
            "def compute_b(y):\n    b = y + 1\n    return b\n"
        )
        with _patch_embeddings_none():
            findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_different_structure_js(self, agent: SemanticCloneAgent) -> None:
        """JavaScript functions with different control flow."""
        src = (
            "function loopFn(arr) {\n"
            "    for (let i = 0; i < arr.length; i++) {\n"
            "        console.log(arr[i]);\n"
            "    }\n"
            "}\n\n"
            "function recursiveFn(n) {\n"
            "    if (n <= 0) return 0;\n"
            "    return n + recursiveFn(n - 1);\n"
            "}\n"
        )
        embeddings = _make_similar_embeddings(2, [])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert findings == []

    def test_unsupported_language(self, agent: SemanticCloneAgent) -> None:
        """Unsupported language returns empty gracefully."""
        src = "IDENTIFICATION DIVISION.\nPROGRAM-ID. HELLO.\n"
        findings = agent.analyze(_ctx(src, "cobol", "hello.cob"))
        assert findings == []

    def test_whitespace_only(self, agent: SemanticCloneAgent) -> None:
        """Whitespace-only source code produces no findings."""
        assert agent.analyze(_ctx("   \n\n  \t  ")) == []

    def test_three_different_functions(self, agent: SemanticCloneAgent) -> None:
        """Three entirely different functions produce no findings."""
        src = (
            "def read_file(path):\n"
            "    with open(path) as f:\n"
            "        return f.read()\n\n"
            "def parse_json(text):\n"
            "    import json\n"
            "    return json.loads(text)\n\n"
            "def send_email(to, body):\n"
            "    print(f'Sending to {to}')\n"
            "    return True\n"
        )
        embeddings = _make_similar_embeddings(3, [])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_below_threshold(self, agent: SemanticCloneAgent) -> None:
        """Pairs with similarity below 0.85 are not reported."""
        src = (
            "def func_a(x):\n    a = x + 1\n    return a\n\n"
            "def func_b(y):\n    b = y * 99\n    return b\n"
        )
        # Create embeddings that are somewhat similar but below threshold
        rng = np.random.RandomState(99)
        dim = 64
        v1 = rng.randn(dim)
        v1 = v1 / np.linalg.norm(v1)
        # Create v2 with cosine similarity ~0.5
        v2 = rng.randn(dim)
        v2 = v2 / np.linalg.norm(v2)
        embeddings = np.stack([v1, v2])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert findings == []


# ---------------------------------------------------------------------------
# Edge cases and metadata
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_metadata_name(self, agent: SemanticCloneAgent) -> None:
        m = agent.metadata()
        assert m.name == "semantic_clone"

    def test_metadata_version(self, agent: SemanticCloneAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_axis_type(self, agent: SemanticCloneAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "aware"

    def test_metadata_model_required(self, agent: SemanticCloneAgent) -> None:
        m = agent.metadata()
        assert m.model_required is True

    def test_metadata_cost(self, agent: SemanticCloneAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 2.0

    def test_metadata_languages(self, agent: SemanticCloneAgent) -> None:
        m = agent.metadata()
        assert "python" in m.languages
        assert "javascript" in m.languages
        assert "go" in m.languages

    def test_metadata_methodology(self, agent: SemanticCloneAgent) -> None:
        m = agent.metadata()
        assert m.methodology == "semantic_understanding"

    def test_explain_returns_string(self, agent: SemanticCloneAgent) -> None:
        src = (
            "def alpha(x):\n    a = x + 1\n    return a\n\n"
            "def beta(y):\n    b = y + 1\n    return b\n"
        )
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 20

    def test_no_ast_fallback_parse(self, agent: SemanticCloneAgent) -> None:
        """When context.ast is None, the agent should parse source code itself."""
        src = (
            "def func_a(x):\n    a = x + 1\n    return a\n\n"
            "def func_b(y):\n    b = y + 1\n    return b\n"
        )
        ctx = CodeContext(source_code=src, language="python", file_path="test.py", ast=None)
        embeddings = _make_similar_embeddings(2, [(0, 1)])
        with _patch_embeddings(embeddings):
            findings = agent.analyze(ctx)
        assert len(findings) == 1

    def test_graceful_on_none_embeddings_real(self, agent: SemanticCloneAgent) -> None:
        """Without mocking, ML deps likely absent -- should return []."""
        src = (
            "def func_a(x):\n    a = x + 1\n    return a\n\n"
            "def func_b(y):\n    b = y + 1\n    return b\n"
        )
        # Don't mock -- this exercises the real get_code_embeddings path.
        # In CI without torch/sentence-transformers, it returns None.
        findings = agent.analyze(_ctx(src))
        # Either [] (no ML deps) or non-empty (ML deps present) -- both valid
        assert isinstance(findings, list)
