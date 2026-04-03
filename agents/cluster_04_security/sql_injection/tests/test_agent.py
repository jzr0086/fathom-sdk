"""Tests for SqlInjectionAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.sql_injection.agent import SqlInjectionAgent


@pytest.fixture
def agent() -> SqlInjectionAgent:
    return SqlInjectionAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_python_fstring_execute(self, agent: SqlInjectionAgent) -> None:
        src = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_format_execute(self, agent: SqlInjectionAgent) -> None:
        src = 'cursor.execute("SELECT * FROM users WHERE id = {}".format(uid))\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_percent_format(self, agent: SqlInjectionAgent) -> None:
        src = 'cursor.execute("SELECT * FROM users WHERE id = %s" % uid)\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_concat(self, agent: SqlInjectionAgent) -> None:
        src = 'cursor.execute("SELECT * FROM users WHERE id = " + uid)\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_template_literal(self, agent: SqlInjectionAgent) -> None:
        src = "db.query(`SELECT * FROM users WHERE id = ${userId}`);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_concat(self, agent: SqlInjectionAgent) -> None:
        src = 'public class T {\n    void f() {\n        stmt.executeQuery("SELECT * FROM users WHERE id = " + id);\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_severity_critical(self, agent: SqlInjectionAgent) -> None:
        src = 'cursor.execute(f"SELECT * FROM users WHERE id = {uid}")\n'
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "critical"

    def test_raw_query(self, agent: SqlInjectionAgent) -> None:
        src = 'Model.raw(f"SELECT * FROM t WHERE id = {x}")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_delete_injection(self, agent: SqlInjectionAgent) -> None:
        src = 'cursor.execute(f"DELETE FROM users WHERE id = {uid}")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_update_injection(self, agent: SqlInjectionAgent) -> None:
        src = 'cursor.execute("UPDATE users SET name = " + name + " WHERE id = " + uid)\n'
        assert len(agent.analyze(_ctx(src))) >= 1


class TestNegativeCases:
    def test_parameterized_query(self, agent: SqlInjectionAgent) -> None:
        assert (
            agent.analyze(_ctx('cursor.execute("SELECT * FROM users WHERE id = %s", (uid,))\n'))
            == []
        )

    def test_placeholder_query(self, agent: SqlInjectionAgent) -> None:
        assert (
            agent.analyze(_ctx('cursor.execute("SELECT * FROM users WHERE id = ?", [uid])\n')) == []
        )

    def test_no_sql(self, agent: SqlInjectionAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_static_query(self, agent: SqlInjectionAgent) -> None:
        assert agent.analyze(_ctx('cursor.execute("SELECT count(*) FROM users")\n')) == []

    def test_orm_call(self, agent: SqlInjectionAgent) -> None:
        assert agent.analyze(_ctx("User.objects.filter(id=uid)\n")) == []

    def test_comment(self, agent: SqlInjectionAgent) -> None:
        assert (
            agent.analyze(_ctx('# cursor.execute(f"SELECT * FROM users WHERE id = {uid}")\n')) == []
        )

    def test_js_parameterized(self, agent: SqlInjectionAgent) -> None:
        src = 'db.query("SELECT * FROM users WHERE id = $1", [userId]);\n'
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_non_query_fstring(self, agent: SqlInjectionAgent) -> None:
        assert agent.analyze(_ctx('print(f"Hello {name}")\n')) == []

    def test_empty(self, agent: SqlInjectionAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_string_without_execute(self, agent: SqlInjectionAgent) -> None:
        assert agent.analyze(_ctx('sql = "SELECT * FROM users"\n')) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: SqlInjectionAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: SqlInjectionAgent) -> None:
        m = agent.metadata()
        assert m.name == "sql_injection"
