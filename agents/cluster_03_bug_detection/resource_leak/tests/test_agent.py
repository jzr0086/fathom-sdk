"""Tests for ResourceLeakAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_03_bug_detection.resource_leak.agent import ResourceLeakAgent


@pytest.fixture
def agent() -> ResourceLeakAgent:
    return ResourceLeakAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_python_open_no_with(self, agent: ResourceLeakAgent) -> None:
        src = "f = open('file.txt')\ndata = f.read()\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_open_in_function(self, agent: ResourceLeakAgent) -> None:
        src = "def read_file():\n    f = open('test.txt')\n    return f.read()\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_create_read_stream(self, agent: ResourceLeakAgent) -> None:
        src = "const stream = fs.createReadStream('file');\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_file_input_stream(self, agent: ResourceLeakAgent) -> None:
        src = 'public class T {\n    void f() {\n        InputStream is = new FileInputStream("f");\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_open_no_defer(self, agent: ResourceLeakAgent) -> None:
        src = 'package main\n\nimport "os"\n\nfunc f() {\n    f, _ := os.Open("file")\n    data := f.Read(buf)\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_severity_medium(self, agent: ResourceLeakAgent) -> None:
        src = "f = open('file.txt')\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "medium"

    def test_python_socket(self, agent: ResourceLeakAgent) -> None:
        src = "s = socket.socket()\ns.connect(('host', 80))\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_connect(self, agent: ResourceLeakAgent) -> None:
        src = "conn = db.connect(dsn)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_multiple_opens(self, agent: ResourceLeakAgent) -> None:
        src = "f1 = open('a.txt')\nf2 = open('b.txt')\n"
        assert len(agent.analyze(_ctx(src))) >= 2

    def test_go_create_no_defer(self, agent: ResourceLeakAgent) -> None:
        src = 'package main\n\nimport "os"\n\nfunc f() {\n    f, _ := os.Create("file")\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1


class TestNegativeCases:
    def test_python_with_open(self, agent: ResourceLeakAgent) -> None:
        src = "with open('file.txt') as f:\n    data = f.read()\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_resource_calls(self, agent: ResourceLeakAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_empty(self, agent: ResourceLeakAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_python_with_nested(self, agent: ResourceLeakAgent) -> None:
        src = "with open('a') as a, open('b') as b:\n    pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_go_with_defer(self, agent: ResourceLeakAgent) -> None:
        src = 'package main\n\nimport "os"\n\nfunc f() {\n    f, _ := os.Open("file")\n    defer f.Close()\n}\n'
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_print_call(self, agent: ResourceLeakAgent) -> None:
        assert agent.analyze(_ctx("print('hello')\n")) == []

    def test_function_call(self, agent: ResourceLeakAgent) -> None:
        assert agent.analyze(_ctx("result = process(data)\n")) == []

    def test_list_operations(self, agent: ResourceLeakAgent) -> None:
        assert agent.analyze(_ctx("items = [1, 2, 3]\nitems.append(4)\n")) == []

    def test_string_open(self, agent: ResourceLeakAgent) -> None:
        # The word "open" in a string shouldn't trigger
        assert agent.analyze(_ctx("msg = 'please open the door'\n")) == []

    def test_comment(self, agent: ResourceLeakAgent) -> None:
        assert agent.analyze(_ctx("# f = open('file')\n")) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: ResourceLeakAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: ResourceLeakAgent) -> None:
        m = agent.metadata()
        assert m.name == "resource_leak"
