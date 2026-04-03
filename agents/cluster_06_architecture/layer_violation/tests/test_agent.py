"""Tests for LayerViolationAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_06_architecture.layer_violation.agent import LayerViolationAgent


@pytest.fixture
def agent() -> LayerViolationAgent:
    return LayerViolationAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ---- Positive cases (violations detected) ----


class TestPositiveCases:
    def test_repository_imports_controller(self, agent: LayerViolationAgent) -> None:
        """repository/ importing from controllers/ is an upward violation."""
        src = "from controllers.user_controller import UserController\n"
        findings = agent.analyze(_ctx(src, fp="src/repository/user_repo.py"))
        assert len(findings) >= 1
        assert findings[0].severity == "medium"

    def test_dal_imports_views(self, agent: LayerViolationAgent) -> None:
        """dal/ importing from views/ is an upward violation."""
        src = "from views.home import HomeView\n"
        findings = agent.analyze(_ctx(src, fp="app/dal/queries.py"))
        assert len(findings) >= 1

    def test_repository_imports_handlers(self, agent: LayerViolationAgent) -> None:
        """repositories/ importing from handlers/ is an upward violation."""
        src = "from handlers.auth import AuthHandler\n"
        findings = agent.analyze(_ctx(src, fp="src/repositories/auth_repo.py"))
        assert len(findings) >= 1

    def test_persistence_imports_routes(self, agent: LayerViolationAgent) -> None:
        """persistence/ importing from routes/ is an upward violation."""
        src = "from routes.api import api_router\n"
        findings = agent.analyze(_ctx(src, fp="src/persistence/db_manager.py"))
        assert len(findings) >= 1

    def test_models_imports_presentation(self, agent: LayerViolationAgent) -> None:
        """models/ importing from presentation/ is an upward violation."""
        src = "from presentation.formatters import format_user\n"
        findings = agent.analyze(_ctx(src, fp="app/models/user.py"))
        assert len(findings) >= 1

    def test_db_imports_controllers(self, agent: LayerViolationAgent) -> None:
        """db/ importing from controllers/ is an upward violation."""
        src = "from controllers.admin import AdminController\n"
        findings = agent.analyze(_ctx(src, fp="src/db/migrations.py"))
        assert len(findings) >= 1

    def test_service_imports_controllers(self, agent: LayerViolationAgent) -> None:
        """service/ importing from controllers/ is an upward violation."""
        src = "from controllers.order_controller import OrderController\n"
        findings = agent.analyze(_ctx(src, fp="src/service/order_service.py"))
        assert len(findings) >= 1

    def test_data_imports_views(self, agent: LayerViolationAgent) -> None:
        """data/ importing from views/ is an upward violation."""
        src = "from views.dashboard import DashboardView\n"
        findings = agent.analyze(_ctx(src, fp="src/data/analytics.py"))
        assert len(findings) >= 1

    def test_repository_imports_service(self, agent: LayerViolationAgent) -> None:
        """repository/ importing from service/ is an upward violation."""
        src = "from service.user_service import UserService\n"
        findings = agent.analyze(_ctx(src, fp="src/repository/user_repo.py"))
        assert len(findings) >= 1

    def test_multiple_violations(self, agent: LayerViolationAgent) -> None:
        """Multiple upward imports should each produce a finding."""
        src = (
            "from controllers.user import UserController\n"
            "from views.home import HomeView\n"
        )
        findings = agent.analyze(_ctx(src, fp="src/repository/user_repo.py"))
        assert len(findings) >= 2

    def test_finding_fields(self, agent: LayerViolationAgent) -> None:
        """Check all expected fields on a finding."""
        src = "from controllers.user_controller import UserController\n"
        findings = agent.analyze(_ctx(src, fp="src/repository/user_repo.py"))
        assert len(findings) == 1
        f = findings[0]
        assert f.agent_name == "layer_violation"
        assert f.severity == "medium"
        assert f.category == "architecture"
        assert f.confidence == 0.75
        assert "architecture" in f.tags
        assert "layers" in f.tags
        assert f.file_path == "src/repository/user_repo.py"

    def test_js_repository_imports_handlers(self, agent: LayerViolationAgent) -> None:
        """JavaScript: repository importing from handlers is a violation."""
        src = "import { AuthHandler } from '../handlers/auth';\n"
        findings = agent.analyze(_ctx(src, "javascript", fp="src/repository/user.js"))
        assert len(findings) >= 1

    def test_domain_imports_presentation(self, agent: LayerViolationAgent) -> None:
        """domain/ importing from presentation/ is an upward violation."""
        src = "from presentation.templates import render\n"
        findings = agent.analyze(_ctx(src, fp="src/domain/logic.py"))
        assert len(findings) >= 1


# ---- Negative cases (no violations) ----


class TestNegativeCases:
    def test_controller_imports_service(self, agent: LayerViolationAgent) -> None:
        """controllers/ importing from service/ is proper top-down flow."""
        src = "from service.user_service import UserService\n"
        assert agent.analyze(_ctx(src, fp="src/controllers/user_controller.py")) == []

    def test_controller_imports_repository(self, agent: LayerViolationAgent) -> None:
        """controllers/ importing from repository/ is proper top-down flow."""
        src = "from repository.user_repo import UserRepo\n"
        assert agent.analyze(_ctx(src, fp="src/controllers/user_controller.py")) == []

    def test_service_imports_repository(self, agent: LayerViolationAgent) -> None:
        """service/ importing from repository/ is proper top-down flow."""
        src = "from repository.user_repo import UserRepo\n"
        assert agent.analyze(_ctx(src, fp="src/service/user_service.py")) == []

    def test_views_imports_services(self, agent: LayerViolationAgent) -> None:
        """views/ importing from services/ is proper top-down flow."""
        src = "from services.auth import AuthService\n"
        assert agent.analyze(_ctx(src, fp="app/views/login.py")) == []

    def test_handlers_imports_dal(self, agent: LayerViolationAgent) -> None:
        """handlers/ importing from dal/ is proper top-down flow."""
        src = "from dal.user_queries import get_user\n"
        assert agent.analyze(_ctx(src, fp="src/handlers/user.py")) == []

    def test_no_imports(self, agent: LayerViolationAgent) -> None:
        """File with no imports produces no findings."""
        src = "x = 1\ny = 2\n"
        assert agent.analyze(_ctx(src, fp="src/repository/user_repo.py")) == []

    def test_unknown_layer_file(self, agent: LayerViolationAgent) -> None:
        """File not in a recognized layer produces no findings."""
        src = "from controllers.user import UserController\n"
        assert agent.analyze(_ctx(src, fp="src/utils/helpers.py")) == []

    def test_unknown_layer_import(self, agent: LayerViolationAgent) -> None:
        """Import from an unrecognized layer produces no findings."""
        src = "from utils.helpers import format_date\n"
        assert agent.analyze(_ctx(src, fp="src/repository/user_repo.py")) == []

    def test_empty_source(self, agent: LayerViolationAgent) -> None:
        """Empty source code produces no findings."""
        assert agent.analyze(_ctx("", fp="src/repository/user_repo.py")) == []

    def test_whitespace_only(self, agent: LayerViolationAgent) -> None:
        """Whitespace-only source code produces no findings."""
        assert agent.analyze(_ctx("   \n  \n", fp="src/repository/user_repo.py")) == []

    def test_same_layer_import(self, agent: LayerViolationAgent) -> None:
        """Importing within the same layer is not a violation."""
        src = "from repository.base import BaseRepo\n"
        assert agent.analyze(_ctx(src, fp="src/repository/user_repo.py")) == []

    def test_third_party_import(self, agent: LayerViolationAgent) -> None:
        """Imports from third-party packages are not layer violations."""
        src = "import sqlalchemy\nfrom flask import Flask\n"
        assert agent.analyze(_ctx(src, fp="src/repository/user_repo.py")) == []

    def test_routes_imports_services(self, agent: LayerViolationAgent) -> None:
        """routes/ importing from services/ is proper top-down flow."""
        src = "from services.order import OrderService\n"
        assert agent.analyze(_ctx(src, fp="src/routes/orders.py")) == []


# ---- Edge cases ----


class TestEdgeCases:
    def test_metadata(self, agent: LayerViolationAgent) -> None:
        m = agent.metadata()
        assert m.name == "layer_violation"
        assert m.version == "0.1.0"
        assert m.languages == ["*"]
        assert m.axis_type == "agnostic"
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: LayerViolationAgent) -> None:
        src = "from controllers.user_controller import UserController\n"
        findings = agent.analyze(_ctx(src, fp="src/repository/user_repo.py"))
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_unsupported_language(self, agent: LayerViolationAgent) -> None:
        """Unsupported language with no AST still returns empty."""
        src = "MOVE A TO B.\n"
        assert agent.analyze(_ctx(src, "cobol", fp="src/repository/main.cob")) == []

    def test_nested_layer_path(self, agent: LayerViolationAgent) -> None:
        """Deeply nested path still detects the layer correctly."""
        src = "from controllers.user import UserController\n"
        findings = agent.analyze(
            _ctx(src, fp="project/src/backend/repository/users/user_repo.py")
        )
        assert len(findings) >= 1

    def test_backslash_paths(self, agent: LayerViolationAgent) -> None:
        """Windows-style backslash paths are handled correctly."""
        src = "from controllers.admin import AdminController\n"
        findings = agent.analyze(
            _ctx(src, fp="src\\repository\\admin_repo.py")
        )
        assert len(findings) >= 1
