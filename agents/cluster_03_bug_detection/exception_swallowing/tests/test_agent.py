"""Tests for ExceptionSwallowingAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_03_bug_detection.exception_swallowing.agent import ExceptionSwallowingAgent


@pytest.fixture
def agent() -> ExceptionSwallowingAgent:
    return ExceptionSwallowingAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_python_bare_except_pass(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept:\n    pass\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_except_exception_pass(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept Exception:\n    pass\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_except_as_pass(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept Exception as e:\n    pass\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_empty_catch(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try {\n    x = 1;\n} catch (e) {\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_ts_empty_catch(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try {\n    x = 1;\n} catch (e) {\n}\n"
        assert len(agent.analyze(_ctx(src, "typescript", "t.ts"))) >= 1

    def test_java_empty_catch(self, agent: ExceptionSwallowingAgent) -> None:
        src = "public class T {\n    void f() {\n        try {\n            int x = 1;\n        } catch (Exception e) {\n        }\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_multiple_empty_catches(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept TypeError:\n    pass\nexcept ValueError:\n    pass\n"
        assert len(agent.analyze(_ctx(src))) >= 2

    def test_finding_severity(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept:\n    pass\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "medium"
        assert f.category == "bug"

    def test_finding_fields(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept:\n    pass\n"
        f = agent.analyze(_ctx(src, fp="src/mod.py"))[0]
        assert f.agent_name == "exception_swallowing"
        assert f.file_path == "src/mod.py"

    def test_nested_try_empty(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    try:\n        x = 1\n    except:\n        pass\nexcept:\n    pass\n"
        assert len(agent.analyze(_ctx(src))) >= 2


class TestNegativeCases:
    def test_except_with_logging(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept Exception as e:\n    logger.error(e)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_except_with_raise(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept Exception:\n    raise\n"
        assert agent.analyze(_ctx(src)) == []

    def test_except_with_return(self, agent: ExceptionSwallowingAgent) -> None:
        src = "def f():\n    try:\n        x = 1\n    except:\n        return None\n"
        assert agent.analyze(_ctx(src)) == []

    def test_js_catch_with_code(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try {\n    x = 1;\n} catch (e) {\n    console.error(e);\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_java_catch_with_code(self, agent: ExceptionSwallowingAgent) -> None:
        src = "public class T {\n    void f() {\n        try {\n            int x = 1;\n        } catch (Exception e) {\n            e.printStackTrace();\n        }\n    }\n}\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_no_try_blocks(self, agent: ExceptionSwallowingAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_except_with_assignment(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept:\n    x = None\n"
        assert agent.analyze(_ctx(src)) == []

    def test_try_finally(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nfinally:\n    cleanup()\n"
        assert agent.analyze(_ctx(src)) == []

    def test_except_with_print(self, agent: ExceptionSwallowingAgent) -> None:
        src = "try:\n    x = 1\nexcept:\n    print('error')\n"
        assert agent.analyze(_ctx(src)) == []

    def test_go_unsupported(self, agent: ExceptionSwallowingAgent) -> None:
        src = "package main\n\nfunc f() {\n    x := 1\n}\n"
        assert agent.analyze(_ctx(src, "go", "t.go")) == []


class TestEdgeCases:
    def test_empty(self, agent: ExceptionSwallowingAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_unsupported_lang(self, agent: ExceptionSwallowingAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: ExceptionSwallowingAgent) -> None:
        m = agent.metadata()
        assert m.name == "exception_swallowing"
        assert m.axis_type == "aware"
