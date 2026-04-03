"""Tests for HardcodedSecretsAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.hardcoded_secrets.agent import HardcodedSecretsAgent


@pytest.fixture
def agent() -> HardcodedSecretsAgent:
    return HardcodedSecretsAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_password_string(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('password = "s3cretP@ss!"\n'))) >= 1

    def test_api_key(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('API_KEY = "abc123def456ghi789"\n'))) >= 1

    def test_secret_key(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('SECRET_KEY = "myverysecretvalue"\n'))) >= 1

    def test_token(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('auth_token = "eyJhbGciOiJIUzI1NiIs"\n'))) >= 1

    def test_db_password(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('db_password = "prod_db_pass123"\n'))) >= 1

    def test_private_key(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('private_key = "MIIEvgIBADANBgkqhki"\n'))) >= 1

    def test_js_api_key(self, agent: HardcodedSecretsAgent) -> None:
        src = 'const apiKey = "sk-1234567890abcdef";\n'
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_severity_critical(self, agent: HardcodedSecretsAgent) -> None:
        f = agent.analyze(_ctx('password = "realpassword123"\n'))[0]
        assert f.severity == "critical"

    def test_access_key(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('access_key = "AKIAIOSFODNN7EXAMPLE"\n'))) >= 1

    def test_client_secret(self, agent: HardcodedSecretsAgent) -> None:
        assert len(agent.analyze(_ctx('client_secret = "a1b2c3d4e5f6g7h8"\n'))) >= 1


class TestNegativeCases:
    def test_env_var(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('password = os.environ["DB_PASS"]\n')) == []

    def test_placeholder(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('password = "changeme"\n')) == []

    def test_placeholder_xxx(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('password = "xxxx"\n')) == []

    def test_todo(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('api_key = "TODO"\n')) == []

    def test_empty_string(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('password = ""\n')) == []

    def test_short_value(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('password = "ab"\n')) == []

    def test_normal_variable(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('username = "john"\n')) == []

    def test_comment_line(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx('# password = "secret"\n')) == []

    def test_process_env(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx("api_key = process.env.API_KEY\n")) == []

    def test_no_assignments(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []


class TestEdgeCases:
    def test_empty(self, agent: HardcodedSecretsAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_metadata(self, agent: HardcodedSecretsAgent) -> None:
        m = agent.metadata()
        assert m.name == "hardcoded_secrets"
        assert "security" in m.tags
