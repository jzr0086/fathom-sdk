"""Tests for CoChangeMinerAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_09_git_intelligence.co_change_miner.agent import CoChangeMinerAgent


@pytest.fixture
def agent() -> CoChangeMinerAgent:
    return CoChangeMinerAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ── Positive Cases (imports from 3+ distinct top-level packages) ──────


class TestPositiveCases:
    def test_python_three_top_level_packages(self, agent: CoChangeMinerAgent) -> None:
        """Importing from 3 distinct top-level packages triggers a finding."""
        src = (
            "from alpha.utils import helper\n"
            "from beta.core import engine\n"
            "from gamma.io import reader\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("3" in f.title for f in findings)

    def test_python_four_top_level_packages(self, agent: CoChangeMinerAgent) -> None:
        """Importing from 4 distinct top-level packages triggers a finding."""
        src = (
            "from alpha.models import User\n"
            "from beta.services import auth\n"
            "from gamma.utils import format_date\n"
            "from delta.config import settings\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("4" in f.title for f in findings)

    def test_python_five_top_level_packages(self, agent: CoChangeMinerAgent) -> None:
        """Importing from 5 distinct top-level packages triggers a finding."""
        src = (
            "from alpha import a\n"
            "from beta import b\n"
            "from gamma import c\n"
            "from delta import d\n"
            "from epsilon import e\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("5" in f.title for f in findings)

    def test_python_mixed_import_styles(self, agent: CoChangeMinerAgent) -> None:
        """Mix of 'import' and 'from...import' from 3+ packages triggers."""
        src = (
            "import alpha\n"
            "from beta.core import engine\n"
            "from gamma.io import reader\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_js_three_different_packages(self, agent: CoChangeMinerAgent) -> None:
        """JavaScript imports from 3 different packages triggers a finding."""
        src = (
            "import { useState } from 'react';\n"
            "import axios from 'axios';\n"
            "import { Router } from 'express';\n"
        )
        findings = agent.analyze(_ctx(src, "javascript", "app.js"))
        assert len(findings) >= 1

    def test_ts_three_different_packages(self, agent: CoChangeMinerAgent) -> None:
        """TypeScript imports from 3 different packages triggers a finding."""
        src = (
            "import { Component } from '@angular/core';\n"
            "import { Observable } from 'rxjs';\n"
            "import { HttpClient } from 'express';\n"
        )
        findings = agent.analyze(_ctx(src, "typescript", "service.ts"))
        assert len(findings) >= 1

    def test_java_three_different_packages(self, agent: CoChangeMinerAgent) -> None:
        """Java imports from 3 distinct top-level packages triggers."""
        src = (
            "import com.example.service.UserService;\n"
            "import org.springframework.web.bind.annotation.RestController;\n"
            "import javax.persistence.Entity;\n"
            "\npublic class App {}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "App.java"))
        assert len(findings) >= 1

    def test_finding_severity_is_low(self, agent: CoChangeMinerAgent) -> None:
        """Findings should have severity 'low'."""
        src = (
            "from alpha.utils import helper\n"
            "from beta.core import engine\n"
            "from gamma.io import reader\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "low"

    def test_finding_confidence_is_065(self, agent: CoChangeMinerAgent) -> None:
        """Findings should have confidence 0.65."""
        src = (
            "from alpha.utils import helper\n"
            "from beta.core import engine\n"
            "from gamma.io import reader\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].confidence == 0.65

    def test_finding_category_is_maintenance(self, agent: CoChangeMinerAgent) -> None:
        """Findings should have category 'maintenance'."""
        src = (
            "from alpha.utils import helper\n"
            "from beta.core import engine\n"
            "from gamma.io import reader\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].category == "maintenance"

    def test_finding_tags(self, agent: CoChangeMinerAgent) -> None:
        """Findings should include expected tags."""
        src = (
            "from alpha.utils import helper\n"
            "from beta.core import engine\n"
            "from gamma.io import reader\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert "coupling" in findings[0].tags

    def test_finding_file_path_preserved(self, agent: CoChangeMinerAgent) -> None:
        """The finding should reference the original file path."""
        src = (
            "from alpha import a\n"
            "from beta import b\n"
            "from gamma import c\n"
        )
        findings = agent.analyze(_ctx(src, fp="src/app/main.py"))
        assert len(findings) >= 1
        assert findings[0].file_path == "src/app/main.py"

    def test_python_deeply_nested_imports(self, agent: CoChangeMinerAgent) -> None:
        """Deeply nested imports still count top-level packages correctly."""
        src = (
            "from alpha.sub1.sub2.module import func\n"
            "from beta.sub3.sub4.module import other\n"
            "from gamma.sub5 import yet_another\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1


# ── Negative Cases (fewer than 3 distinct top-level packages) ─────────


class TestNegativeCases:
    def test_python_two_top_level_packages(self, agent: CoChangeMinerAgent) -> None:
        """Only 2 distinct top-level packages should not trigger."""
        src = (
            "from alpha.utils import helper\n"
            "from beta.core import engine\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_python_one_package(self, agent: CoChangeMinerAgent) -> None:
        """Single package imports should not trigger."""
        src = (
            "from alpha.utils import helper\n"
            "from alpha.core import engine\n"
            "from alpha.io import reader\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_python_single_import(self, agent: CoChangeMinerAgent) -> None:
        """A single import should not trigger."""
        src = "from alpha.utils import helper\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_imports(self, agent: CoChangeMinerAgent) -> None:
        """Code with no imports should produce no findings."""
        src = "x = 1\ny = 2\nprint(x + y)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: CoChangeMinerAgent) -> None:
        """Empty source should produce no findings."""
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: CoChangeMinerAgent) -> None:
        """Whitespace-only source should produce no findings."""
        assert agent.analyze(_ctx("   \n\n   \n")) == []

    def test_comments_only(self, agent: CoChangeMinerAgent) -> None:
        """Comments that look like imports should not trigger."""
        src = "# from alpha import a\n# from beta import b\n# from gamma import c\n"
        assert agent.analyze(_ctx(src)) == []

    def test_python_same_package_multiple_submodules(
        self, agent: CoChangeMinerAgent
    ) -> None:
        """Multiple submodule imports from the same top-level should not trigger."""
        src = (
            "from myapp.models import User\n"
            "from myapp.views import UserView\n"
            "from myapp.serializers import UserSerializer\n"
            "from myapp.urls import urlpatterns\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_js_two_packages(self, agent: CoChangeMinerAgent) -> None:
        """JavaScript with only 2 different packages should not trigger."""
        src = (
            "import { useState } from 'react';\n"
            "import { useEffect } from 'react';\n"
            "import axios from 'axios';\n"
        )
        findings = agent.analyze(_ctx(src, "javascript", "app.js"))
        assert findings == []

    def test_python_no_commit_history_graceful(
        self, agent: CoChangeMinerAgent
    ) -> None:
        """No commit history should not produce co-change findings."""
        src = "x = 1\n"
        ctx = _ctx(src)
        assert ctx.commit_history is None
        findings = agent.analyze(ctx)
        assert findings == []

    def test_unsupported_language(self, agent: CoChangeMinerAgent) -> None:
        """Unsupported languages with no AST should return empty."""
        src = "MOVE A TO B.\nMOVE C TO D.\n"
        assert agent.analyze(_ctx(src, "cobol", "main.cob")) == []

    def test_python_two_distinct_plus_stdlib(
        self, agent: CoChangeMinerAgent
    ) -> None:
        """Two distinct custom packages plus stdlib should not exceed threshold
        if the stdlib import resolves to a different top-level."""
        src = (
            "from alpha.core import engine\n"
            "from beta.io import reader\n"
        )
        assert agent.analyze(_ctx(src)) == []


# ── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_metadata_name(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert m.name == "co_change_miner"

    def test_metadata_version(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages_agnostic(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert m.languages == ["*"]

    def test_metadata_domains(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert "enterprise_engineering" in m.domains
        assert "web_development" in m.domains

    def test_metadata_methodology(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert m.methodology == "git_intelligence"

    def test_metadata_axis_type(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "agnostic"

    def test_metadata_model_not_required(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False

    def test_metadata_zero_cost(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 0.0

    def test_metadata_tags(self, agent: CoChangeMinerAgent) -> None:
        m = agent.metadata()
        assert "git" in m.tags
        assert "coupling" in m.tags
        assert "co-change" in m.tags

    def test_explain_returns_string(self, agent: CoChangeMinerAgent) -> None:
        src = (
            "from alpha import a\n"
            "from beta import b\n"
            "from gamma import c\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_description_lists_packages(self, agent: CoChangeMinerAgent) -> None:
        """The finding description should list the detected top-level packages."""
        src = (
            "from alpha.utils import helper\n"
            "from beta.core import engine\n"
            "from gamma.io import reader\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        desc = findings[0].description
        assert "alpha" in desc
        assert "beta" in desc
        assert "gamma" in desc

    def test_exactly_at_threshold(self, agent: CoChangeMinerAgent) -> None:
        """Exactly 3 distinct packages should trigger (threshold is 3)."""
        src = (
            "from pkg_a.mod import x\n"
            "from pkg_b.mod import y\n"
            "from pkg_c.mod import z\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_go_three_packages(self, agent: CoChangeMinerAgent) -> None:
        """Go imports from 3 different top-level paths triggers a finding."""
        src = (
            'package main\n\n'
            'import (\n'
            '    "fmt"\n'
            '    "net/http"\n'
            '    "os"\n'
            ')\n'
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1
