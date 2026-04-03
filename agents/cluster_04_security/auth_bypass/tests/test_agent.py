"""Tests for AuthBypassAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_04_security.auth_bypass.agent import AuthBypassAgent


@pytest.fixture
def agent() -> AuthBypassAgent:
    return AuthBypassAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


# ── Positive cases (should produce findings) ────────────────────────────


class TestPositiveCases:
    def test_flask_post_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "from flask import Flask\n"
            "app = Flask(__name__)\n"
            "\n"
            "@app.route('/users', methods=['POST'])\n"
            "def create_user():\n"
            "    pass\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_flask_delete_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/users/<id>', methods=['DELETE'])\n"
            "def delete_user(id):\n"
            "    pass\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_flask_put_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/items/<id>', methods=['PUT'])\n"
            "def update_item(id):\n"
            "    pass\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_flask_patch_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/items/<id>', methods=['PATCH'])\n"
            "def partial_update(id):\n"
            "    pass\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_express_post_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "const express = require('express');\n"
            "const router = express.Router();\n"
            "\n"
            "router.post('/users', (req, res) => {\n"
            "    res.send('created');\n"
            "});\n"
        )
        findings = agent.analyze(_ctx(src, "javascript", "routes.js"))
        assert len(findings) >= 1

    def test_express_delete_no_auth(self, agent: AuthBypassAgent) -> None:
        src = "router.delete('/items/:id', (req, res) => { res.send('ok'); });\n"
        assert len(agent.analyze(_ctx(src, "javascript", "routes.js"))) >= 1

    def test_express_put_no_auth(self, agent: AuthBypassAgent) -> None:
        src = "app.put('/items/:id', handler);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "routes.js"))) >= 1

    def test_express_patch_no_auth(self, agent: AuthBypassAgent) -> None:
        src = "app.patch('/items/:id', handler);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "routes.js"))) >= 1

    def test_django_path_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "from django.urls import path\n"
            "\n"
            "urlpatterns = [\n"
            "    path('users/', CreateUserView.as_view()),\n"
            "]\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_django_re_path_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "from django.urls import re_path\n"
            "\n"
            "urlpatterns = [\n"
            "    re_path(r'^users/$', create_user_view),\n"
            "]\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_flask_multiple_methods_includes_post(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/data', methods=['GET', 'POST'])\n"
            "def handle_data():\n"
            "    pass\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_confidence_value(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/users', methods=['POST'])\n"
            "def create_user():\n"
            "    pass\n"
        )
        findings = agent.analyze(_ctx(src))
        assert findings[0].confidence == 0.72

    def test_blueprint_route_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "@bp.route('/items', methods=['POST'])\n"
            "def create_item():\n"
            "    pass\n"
        )
        assert len(agent.analyze(_ctx(src))) >= 1


# ── Negative cases (should NOT produce findings) ────────────────────────


class TestNegativeCases:
    def test_flask_post_with_login_required(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/users', methods=['POST'])\n"
            "@login_required\n"
            "def create_user():\n"
            "    pass\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_flask_post_with_auth_required_above(self, agent: AuthBypassAgent) -> None:
        src = (
            "@auth_required\n"
            "@app.route('/users', methods=['POST'])\n"
            "def create_user():\n"
            "    pass\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_flask_post_with_jwt_required(self, agent: AuthBypassAgent) -> None:
        src = (
            "@jwt_required\n"
            "@app.route('/items', methods=['DELETE'])\n"
            "def delete_item():\n"
            "    pass\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_flask_get_only_no_auth(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/users', methods=['GET'])\n"
            "def list_users():\n"
            "    pass\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_express_post_with_auth_middleware(self, agent: AuthBypassAgent) -> None:
        src = "router.post('/users', authenticate, (req, res) => { res.send('ok'); });\n"
        assert agent.analyze(_ctx(src, "javascript", "routes.js")) == []

    def test_express_post_with_passport(self, agent: AuthBypassAgent) -> None:
        src = "router.post('/login', passport.authenticate('local'), handler);\n"
        assert agent.analyze(_ctx(src, "javascript", "routes.js")) == []

    def test_express_post_with_verifyToken(self, agent: AuthBypassAgent) -> None:
        src = "router.post('/data', verifyToken, controller.create);\n"
        assert agent.analyze(_ctx(src, "javascript", "routes.js")) == []

    def test_django_path_with_login_required_mixin(self, agent: AuthBypassAgent) -> None:
        src = (
            "class CreateUserView(LoginRequiredMixin, CreateView):\n"
            "    pass\n"
            "\n"
            "urlpatterns = [\n"
            "    path('users/', CreateUserView.as_view()),\n"
            "]\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_django_with_IsAuthenticated(self, agent: AuthBypassAgent) -> None:
        src = (
            "    permission_classes = [IsAuthenticated]\n"
            "\n"
            "urlpatterns = [\n"
            "    path('api/users/', UserViewSet.as_view()),\n"
            "]\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_non_route_code(self, agent: AuthBypassAgent) -> None:
        src = (
            "def calculate_sum(a, b):\n"
            "    return a + b\n"
            "\n"
            "result = calculate_sum(1, 2)\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: AuthBypassAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_comment_line(self, agent: AuthBypassAgent) -> None:
        src = "# @app.route('/users', methods=['POST'])\n"
        assert agent.analyze(_ctx(src)) == []

    def test_express_auth_middleware_above(self, agent: AuthBypassAgent) -> None:
        src = (
            "router.use(authMiddleware);\n"
            "\n"
            "router.post('/users', handler);\n"
        )
        assert agent.analyze(_ctx(src, "javascript", "routes.js")) == []


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_metadata(self, agent: AuthBypassAgent) -> None:
        m = agent.metadata()
        assert m.name == "auth_bypass"
        assert m.version == "0.1.0"
        assert m.languages == ["*"]
        assert "security" in m.tags
        assert "auth" in m.tags
        assert "bypass" in m.tags
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0
        assert m.axis_type == "agnostic"

    def test_explain(self, agent: AuthBypassAgent) -> None:
        finding = agent.analyze(
            _ctx("@app.route('/x', methods=['POST'])\ndef f():\n    pass\n")
        )[0]
        explanation = agent.explain(finding)
        assert "auth" in explanation.lower()

    def test_category_is_security(self, agent: AuthBypassAgent) -> None:
        src = (
            "@app.route('/users', methods=['POST'])\n"
            "def create_user():\n"
            "    pass\n"
        )
        findings = agent.analyze(_ctx(src))
        assert findings[0].category == "security"

    def test_whitespace_only_source(self, agent: AuthBypassAgent) -> None:
        assert agent.analyze(_ctx("   \n\n   \n")) == []

    def test_multiple_routes_mixed(self, agent: AuthBypassAgent) -> None:
        src = (
            "@login_required\n"
            "@app.route('/safe', methods=['POST'])\n"
            "def safe():\n"
            "    pass\n"
            "\n"
            "@app.route('/unsafe', methods=['DELETE'])\n"
            "def unsafe():\n"
            "    pass\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 1
        assert "unsafe" in findings[0].description.lower() or findings[0].line_start == 6
