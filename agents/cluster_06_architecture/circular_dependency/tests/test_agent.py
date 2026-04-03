"""Tests for CircularDependencyAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_06_architecture.circular_dependency.agent import CircularDependencyAgent


@pytest.fixture
def agent() -> CircularDependencyAgent:
    return CircularDependencyAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ── Positive Cases ──────────────────────────────────────────────────────


class TestPositiveCases:
    def test_python_self_import(self, agent: CircularDependencyAgent) -> None:
        """A Python module that imports itself."""
        src = "from myapp import myapp\n"
        findings = agent.analyze(_ctx(src, "python", "myapp.py"))
        assert len(findings) >= 1

    def test_python_self_import_from(self, agent: CircularDependencyAgent) -> None:
        """from <own_module> import name."""
        src = "from utils import helper\n"
        findings = agent.analyze(_ctx(src, "python", "utils.py"))
        assert len(findings) >= 1

    def test_python_dotted_self_import(self, agent: CircularDependencyAgent) -> None:
        """from app.services import x inside app/services.py."""
        src = "from app.services import process\n"
        findings = agent.analyze(_ctx(src, "python", "app/services.py"))
        assert len(findings) >= 1

    def test_python_import_statement_self(self, agent: CircularDependencyAgent) -> None:
        """import <own_module> form."""
        src = "import models\n"
        findings = agent.analyze(_ctx(src, "python", "models.py"))
        assert len(findings) >= 1

    def test_js_self_import(self, agent: CircularDependencyAgent) -> None:
        """JS file importing itself via relative path."""
        src = "import { foo } from './utils';\n"
        findings = agent.analyze(_ctx(src, "javascript", "./utils.js"))
        assert len(findings) >= 1

    def test_ts_self_import(self, agent: CircularDependencyAgent) -> None:
        """TS file importing itself."""
        src = "import { Bar } from './service';\n"
        findings = agent.analyze(_ctx(src, "typescript", "./service.ts"))
        assert len(findings) >= 1

    def test_java_self_import(self, agent: CircularDependencyAgent) -> None:
        """Java class importing its own package path."""
        src = "import com.example.Utils;\n\npublic class Utils {}\n"
        findings = agent.analyze(_ctx(src, "java", "com/example/Utils.java"))
        assert len(findings) >= 1

    def test_finding_severity_high(self, agent: CircularDependencyAgent) -> None:
        """Findings should have severity 'high'."""
        src = "from utils import helper\n"
        findings = agent.analyze(_ctx(src, "python", "utils.py"))
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_finding_confidence(self, agent: CircularDependencyAgent) -> None:
        """Findings should have confidence 0.85."""
        src = "from utils import helper\n"
        findings = agent.analyze(_ctx(src, "python", "utils.py"))
        assert len(findings) >= 1
        assert findings[0].confidence == 0.85

    def test_finding_category_architecture(self, agent: CircularDependencyAgent) -> None:
        """Findings should have category 'architecture'."""
        src = "from utils import helper\n"
        findings = agent.analyze(_ctx(src, "python", "utils.py"))
        assert len(findings) >= 1
        assert findings[0].category == "architecture"

    def test_multiple_self_imports(self, agent: CircularDependencyAgent) -> None:
        """Multiple self-import lines should still produce finding(s)."""
        src = "from mymod import a\nfrom mymod import b\n"
        findings = agent.analyze(_ctx(src, "python", "mymod.py"))
        assert len(findings) >= 1


# ── Negative Cases ──────────────────────────────────────────────────────


class TestNegativeCases:
    def test_normal_import(self, agent: CircularDependencyAgent) -> None:
        """Importing a different module should not trigger."""
        src = "import os\nimport sys\n"
        assert agent.analyze(_ctx(src, "python", "main.py")) == []

    def test_from_import_different_module(self, agent: CircularDependencyAgent) -> None:
        """from <other_module> import name is fine."""
        src = "from collections import defaultdict\n"
        assert agent.analyze(_ctx(src, "python", "main.py")) == []

    def test_no_imports(self, agent: CircularDependencyAgent) -> None:
        """Code with no imports should produce no findings."""
        src = "x = 1\ny = 2\nprint(x + y)\n"
        assert agent.analyze(_ctx(src, "python", "main.py")) == []

    def test_empty_source(self, agent: CircularDependencyAgent) -> None:
        """Empty source should produce no findings."""
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: CircularDependencyAgent) -> None:
        """Whitespace-only source should produce no findings."""
        assert agent.analyze(_ctx("   \n\n   \n")) == []

    def test_js_import_different_module(self, agent: CircularDependencyAgent) -> None:
        """JS importing a different module is fine."""
        src = "import { useState } from 'react';\n"
        assert agent.analyze(_ctx(src, "javascript", "App.js")) == []

    def test_ts_import_different_module(self, agent: CircularDependencyAgent) -> None:
        """TS importing a different module is fine."""
        src = "import { Component } from '@angular/core';\n"
        assert agent.analyze(_ctx(src, "typescript", "app.ts")) == []

    def test_java_import_different_package(self, agent: CircularDependencyAgent) -> None:
        """Java importing a different package is fine."""
        src = "import java.util.List;\n\npublic class App {}\n"
        assert agent.analyze(_ctx(src, "java", "com/example/App.java")) == []

    def test_go_import_different_package(self, agent: CircularDependencyAgent) -> None:
        """Go importing a standard library package is fine."""
        src = 'package main\n\nimport "fmt"\n'
        assert agent.analyze(_ctx(src, "go", "main.go")) == []

    def test_python_comments_only(self, agent: CircularDependencyAgent) -> None:
        """Comments that look like imports should not trigger."""
        src = "# import utils\n# from utils import helper\nx = 1\n"
        assert agent.analyze(_ctx(src, "python", "utils.py")) == []

    def test_no_functions_no_imports(self, agent: CircularDependencyAgent) -> None:
        """Plain assignments with no imports."""
        src = "a = 1\nb = 2\nc = a + b\n"
        assert agent.analyze(_ctx(src, "python", "calc.py")) == []


# ── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unsupported_language(self, agent: CircularDependencyAgent) -> None:
        """Unsupported languages should return empty findings."""
        assert agent.analyze(_ctx("import foo", "cobol", "t.cob")) == []

    def test_metadata_name(self, agent: CircularDependencyAgent) -> None:
        m = agent.metadata()
        assert m.name == "circular_dependency"

    def test_metadata_version(self, agent: CircularDependencyAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages_agnostic(self, agent: CircularDependencyAgent) -> None:
        m = agent.metadata()
        assert m.languages == ["*"]

    def test_metadata_axis_type(self, agent: CircularDependencyAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "agnostic"

    def test_metadata_methodology(self, agent: CircularDependencyAgent) -> None:
        m = agent.metadata()
        assert m.methodology == "architecture"

    def test_metadata_model_not_required(self, agent: CircularDependencyAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False

    def test_metadata_zero_cost(self, agent: CircularDependencyAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: CircularDependencyAgent) -> None:
        src = "from utils import helper\n"
        findings = agent.analyze(_ctx(src, "python", "utils.py"))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_finding_file_path_preserved(self, agent: CircularDependencyAgent) -> None:
        """The finding should reference the original file path."""
        src = "from mymod import thing\n"
        findings = agent.analyze(_ctx(src, "python", "mymod.py"))
        assert len(findings) >= 1
        assert findings[0].file_path == "mymod.py"
