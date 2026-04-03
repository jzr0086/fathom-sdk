"""Tests for JsAsyncAwaitAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_10_language_specific.js_async_await.agent import JsAsyncAwaitAgent


@pytest.fixture
def agent() -> JsAsyncAwaitAgent:
    return JsAsyncAwaitAgent()


def _ctx(source: str, language: str = "javascript", fp: str = "test.js") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# -----------------------------------------------------------------------
# Positive cases (should produce findings)
# -----------------------------------------------------------------------


class TestPositiveCases:
    def test_missing_await_on_fetch(self, agent: JsAsyncAwaitAgent) -> None:
        src = "async function loadData() {\n    const res = fetch('/api/data');\n}\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_missing_await_on_json(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function loadData() {\n"
            "    const res = await fetch('/api');\n"
            "    const data = res.json();\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("json" in f.title for f in findings)

    def test_await_in_for_loop(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function fetchAll(urls) {\n"
            "    for (let i = 0; i < urls.length; i++) {\n"
            "        const res = await fetch(urls[i]);\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("loop" in f.title.lower() or "Promise.all" in f.description for f in findings)

    def test_await_in_for_in_loop(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function fetchAll(urls) {\n"
            "    for (const url of urls) {\n"
            "        await fetch(url);\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "medium" for f in findings)

    def test_await_in_while_loop(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function poll() {\n"
            "    let done = false;\n"
            "    while (!done) {\n"
            "        const res = await fetch('/status');\n"
            "        done = true;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_then_without_catch(self, agent: JsAsyncAwaitAgent) -> None:
        src = "function doWork() {\n    fetch('/api').then(res => res.json());\n}\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("catch" in f.title.lower() for f in findings)

    def test_then_without_catch_severity(self, agent: JsAsyncAwaitAgent) -> None:
        src = "function doWork() {\n    fetch('/api').then(r => r.json());\n}\n"
        findings = agent.analyze(_ctx(src))
        then_findings = [f for f in findings if ".then()" in f.title]
        assert len(then_findings) >= 1
        assert then_findings[0].severity == "medium"

    def test_missing_await_typescript(self, agent: JsAsyncAwaitAgent) -> None:
        src = "async function loadData(): Promise<void> {\n    const res = fetch('/api');\n}\n"
        findings = agent.analyze(_ctx(src, "typescript", "test.ts"))
        assert len(findings) >= 1

    def test_missing_await_on_axios(self, agent: JsAsyncAwaitAgent) -> None:
        src = "async function getData() {\n    const res = axios.get('/api');\n}\n"
        findings = agent.analyze(_ctx(src))
        # 'get' is in _ASYNC_API_CALLS
        assert len(findings) >= 1

    def test_multiple_missing_awaits(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function process() {\n"
            "    const a = fetch('/api/a');\n"
            "    const b = fetch('/api/b');\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 2

    def test_finding_fields(self, agent: JsAsyncAwaitAgent) -> None:
        src = "async function f() {\n    fetch('/api');\n}\n"
        findings = agent.analyze(_ctx(src, fp="src/service.js"))
        assert len(findings) >= 1
        f = findings[0]
        assert f.agent_name == "js_async_await"
        assert f.file_path == "src/service.js"
        assert f.category == "bug"
        assert f.confidence == 0.78

    def test_await_in_do_while_loop(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function retry() {\n"
            "    do {\n"
            "        await fetch('/api');\n"
            "    } while (shouldRetry);\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1


# -----------------------------------------------------------------------
# Negative cases (should NOT produce findings)
# -----------------------------------------------------------------------


class TestNegativeCases:
    def test_properly_awaited_fetch(self, agent: JsAsyncAwaitAgent) -> None:
        src = "async function loadData() {\n    const res = await fetch('/api');\n}\n"
        findings = agent.analyze(_ctx(src))
        assert findings == []

    def test_promise_all_pattern(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function fetchAll(urls) {\n"
            "    const results = await Promise.all(urls.map(u => fetch(u)));\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        # fetch inside the map callback is not in an async function detected by find_functions
        # and the Promise.all is properly awaited
        assert findings == []

    def test_then_with_catch(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "function doWork() {\n"
            "    fetch('/api').then(res => res.json()).catch(err => console.error(err));\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        then_findings = [f for f in findings if ".then()" in f.title]
        assert then_findings == []

    def test_non_async_code(self, agent: JsAsyncAwaitAgent) -> None:
        src = "function add(a, b) {\n    return a + b;\n}\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: JsAsyncAwaitAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: JsAsyncAwaitAgent) -> None:
        assert agent.analyze(_ctx("   \n  \n")) == []

    def test_sync_fetch_outside_async(self, agent: JsAsyncAwaitAgent) -> None:
        src = "function doWork() {\n    const res = fetch('/api');\n}\n"
        findings = agent.analyze(_ctx(src))
        # Missing await detection only applies inside async functions
        missing_await = [f for f in findings if "Missing await" in f.title]
        assert missing_await == []

    def test_python_not_supported(self, agent: JsAsyncAwaitAgent) -> None:
        src = "async def f():\n    await asyncio.sleep(1)\n"
        assert agent.analyze(_ctx(src, "python", "test.py")) == []

    def test_go_not_supported(self, agent: JsAsyncAwaitAgent) -> None:
        src = "package main\n\nfunc f() {\n    x := 1\n}\n"
        assert agent.analyze(_ctx(src, "go", "test.go")) == []

    def test_properly_awaited_chain(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function loadData() {\n"
            "    const data = await fetch('/api');\n"
            "    const parsed = await data.json();\n"
            "    return parsed;\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_regular_sync_loop(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "function sum(nums) {\n"
            "    let total = 0;\n"
            "    for (let i = 0; i < nums.length; i++) {\n"
            "        total += nums[i];\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_await_outside_loop(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function loadData() {\n"
            "    const res = await fetch('/api');\n"
            "    for (const item of res.items) {\n"
            "        console.log(item);\n"
            "    }\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src)) == []


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_unsupported_lang(self, agent: JsAsyncAwaitAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: JsAsyncAwaitAgent) -> None:
        m = agent.metadata()
        assert m.name == "js_async_await"
        assert m.version == "0.1.0"
        assert m.axis_type == "critical"
        assert m.methodology == "language_specific"
        assert "javascript" in m.languages
        assert "typescript" in m.languages
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain(self, agent: JsAsyncAwaitAgent) -> None:
        src = "async function f() {\n    fetch('/api');\n}\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert len(explanation) > 0
        assert "async" in explanation.lower() or "await" in explanation.lower()

    def test_typescript_language(self, agent: JsAsyncAwaitAgent) -> None:
        src = (
            "async function fetchData(): Promise<Response> {\n"
            "    const result = fetch('/api');\n"
            "    return result;\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "typescript", "test.ts"))
        assert len(findings) >= 1

    def test_arrow_function_async(self, agent: JsAsyncAwaitAgent) -> None:
        src = "const loadData = async () => {\n    const res = fetch('/api');\n};\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
