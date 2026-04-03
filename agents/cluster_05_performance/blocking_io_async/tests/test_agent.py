"""Tests for BlockingIoAsyncAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_05_performance.blocking_io_async.agent import BlockingIoAsyncAgent


@pytest.fixture
def agent() -> BlockingIoAsyncAgent:
    return BlockingIoAsyncAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_python_time_sleep(self, agent: BlockingIoAsyncAgent) -> None:
        src = "import time\nasync def f():\n    time.sleep(1)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_requests_get(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    resp = requests.get(url)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_requests_post(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    resp = requests.post(url, data=d)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_open(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    f = open('file.txt')\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_subprocess_run(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    subprocess.run(['ls'])\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_readfilesync(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async function f() {\n    const data = fs.readFileSync('file');\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_js_writefilesync(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async function f() {\n    fs.writeFileSync('file', data);\n}\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_ts_execsync(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async function f() {\n    child_process.execSync('cmd');\n}\n"
        assert len(agent.analyze(_ctx(src, "typescript", "t.ts"))) >= 1

    def test_severity_high(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    time.sleep(5)\n"
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "high"

    def test_multiple_blocking(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    time.sleep(1)\n    requests.get(url)\n"
        assert len(agent.analyze(_ctx(src))) >= 2


class TestNegativeCases:
    def test_sync_function(self, agent: BlockingIoAsyncAgent) -> None:
        src = "def f():\n    time.sleep(1)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_async_with_await(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    await asyncio.sleep(1)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_async_aiohttp(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    async with aiohttp.ClientSession() as s:\n        pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_async(self, agent: BlockingIoAsyncAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_js_async_read(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async function f() {\n    const data = await fs.readFile('file');\n}\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_go_not_supported(self, agent: BlockingIoAsyncAgent) -> None:
        src = "package main\n\nfunc f() {\n    time.Sleep(time.Second)\n}\n"
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_empty(self, agent: BlockingIoAsyncAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_sync_requests(self, agent: BlockingIoAsyncAgent) -> None:
        src = "def f():\n    resp = requests.get(url)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_class_method_sync(self, agent: BlockingIoAsyncAgent) -> None:
        src = "class Foo:\n    def bar(self):\n        open('f')\n"
        assert agent.analyze(_ctx(src)) == []

    def test_async_no_blocking(self, agent: BlockingIoAsyncAgent) -> None:
        src = "async def f():\n    x = 1 + 2\n    return x\n"
        assert agent.analyze(_ctx(src)) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: BlockingIoAsyncAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: BlockingIoAsyncAgent) -> None:
        m = agent.metadata()
        assert m.name == "blocking_io_async"
