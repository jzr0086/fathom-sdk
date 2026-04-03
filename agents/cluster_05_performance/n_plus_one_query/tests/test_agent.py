"""Tests for NPlusOneQueryAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_05_performance.n_plus_one_query.agent import NPlusOneQueryAgent


@pytest.fixture
def agent() -> NPlusOneQueryAgent:
    return NPlusOneQueryAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_execute_in_for(self, agent: NPlusOneQueryAgent) -> None:
        src = "for item in items:\n    cursor.execute(query)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_fetchone_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "for x in xs:\n    row = cursor.fetchone()\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_find_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "for u in users:\n    profile = db.find(u.id)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_query_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "for (let u of users) {\n    const p = db.findOne({id: u.id});\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_execute_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "public class T {\n    void f() {\n        for (User u : users) {\n            stmt.executeQuery(q);\n        }\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_query_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = 'package main\n\nfunc f() {\n    for _, u := range users {\n        db.QueryRow("SELECT...")\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_severity_high(self, agent: NPlusOneQueryAgent) -> None:
        src = "for item in items:\n    cursor.execute(query)\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "high"

    def test_save_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "for obj in objects:\n    obj.save()\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_create_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "for data in items:\n    Model.create(data)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_while_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "while has_more:\n    cursor.execute(query)\n"
        assert len(agent.analyze(_ctx(src))) >= 1


class TestNegativeCases:
    def test_outside_loop(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("result = cursor.execute(query)\n")) == []

    def test_batch_operation(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("cursor.executemany(query, items)\n")) == []

    def test_no_db_calls(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("for x in items:\n    print(x)\n")) == []

    def test_no_loops(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("x = db.query(q)\n")) == []

    def test_empty(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_list_comprehension(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("results = [process(x) for x in items]\n")) == []

    def test_print_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("for x in items:\n    print(x)\n")) == []

    def test_math_in_loop(self, agent: NPlusOneQueryAgent) -> None:
        src = "for i in range(10):\n    total += i * 2\n"
        assert agent.analyze(_ctx(src)) == []

    def test_simple_code(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_function_def(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("def f():\n    return 1\n")) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: NPlusOneQueryAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: NPlusOneQueryAgent) -> None:
        m = agent.metadata()
        assert m.name == "n_plus_one_query"
