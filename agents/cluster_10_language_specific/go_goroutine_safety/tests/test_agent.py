"""Tests for GoGoroutineSafetyAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_10_language_specific.go_goroutine_safety.agent import (
    GoGoroutineSafetyAgent,
)


@pytest.fixture
def agent() -> GoGoroutineSafetyAgent:
    return GoGoroutineSafetyAgent()


def _ctx(source: str, language: str = "go", fp: str = "test.go") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ── Positive cases (10+) ─────────────────────────────────────────────────


class TestPositiveCases:
    def test_lock_without_defer_unlock(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "import \"sync\"\n\n"
            "var mu sync.Mutex\n\n"
            "func f() {\n"
            "    mu.Lock()\n"
            "    doWork()\n"
            "    mu.Unlock()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("Lock without defer Unlock" in f.title for f in findings)

    def test_lock_without_defer_severity_medium(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    mu.Lock()\n"
            "    process()\n"
            "    mu.Unlock()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        lock_findings = [f for f in findings if "Lock without defer" in f.title]
        assert len(lock_findings) >= 1
        assert all(f.severity == "medium" for f in lock_findings)

    def test_wg_add_inside_goroutine(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "import \"sync\"\n\n"
            "func f() {\n"
            "    var wg sync.WaitGroup\n"
            "    for i := 0; i < 10; i++ {\n"
            "        go func() {\n"
            "            wg.Add(1)\n"
            "            defer wg.Done()\n"
            "            work()\n"
            "        }()\n"
            "    }\n"
            "    wg.Wait()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert any("WaitGroup.Add called inside goroutine" in f.title for f in findings)

    def test_wg_add_inside_goroutine_severity_high(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    var wg sync.WaitGroup\n"
            "    go func() {\n"
            "        wg.Add(1)\n"
            "        defer wg.Done()\n"
            "    }()\n"
            "    wg.Wait()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        wg_findings = [f for f in findings if "WaitGroup" in f.title]
        assert len(wg_findings) >= 1
        assert all(f.severity == "high" for f in wg_findings)

    def test_goroutine_without_waitgroup(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    go func() {\n"
            "        doWork()\n"
            "    }()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert any("without WaitGroup or context" in f.title for f in findings)

    def test_goroutine_leak_severity_high(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    go func() {\n"
            "        for {\n"
            "            process()\n"
            "        }\n"
            "    }()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        leak_findings = [f for f in findings if "without WaitGroup" in f.title]
        assert len(leak_findings) >= 1
        assert all(f.severity == "high" for f in leak_findings)

    def test_multiple_goroutines_without_wg(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    go doA()\n"
            "    go doB()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        leak_findings = [f for f in findings if "without WaitGroup" in f.title]
        assert len(leak_findings) >= 2

    def test_channel_deadlock_same_goroutine(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    ch := make(chan int)\n"
            "    ch <- 42\n"
            "    val := <-ch\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert any("deadlock" in f.title.lower() for f in findings)

    def test_lock_no_unlock_at_all(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    mu.Lock()\n"
            "    doWork()\n"
            "    return\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert any("Lock without defer Unlock" in f.title for f in findings)

    def test_goroutine_no_function_call(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    go handle()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert any("without WaitGroup or context" in f.title for f in findings)

    def test_category_is_bug(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    mu.Lock()\n"
            "    process()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert all(f.category == "bug" for f in findings)

    def test_confidence_078(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    go func() { work() }()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert all(f.confidence == 0.78 for f in findings)


# ── Negative cases (10+) ─────────────────────────────────────────────────


class TestNegativeCases:
    def test_proper_defer_unlock(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    mu.Lock()\n"
            "    defer mu.Unlock()\n"
            "    doWork()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        lock_findings = [f for f in findings if "Lock without defer" in f.title]
        assert len(lock_findings) == 0

    def test_wg_add_before_go(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "import \"sync\"\n\n"
            "func f() {\n"
            "    var wg sync.WaitGroup\n"
            "    for i := 0; i < 10; i++ {\n"
            "        wg.Add(1)\n"
            "        go func() {\n"
            "            defer wg.Done()\n"
            "            work()\n"
            "        }()\n"
            "    }\n"
            "    wg.Wait()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        wg_findings = [f for f in findings if "WaitGroup.Add" in f.title]
        assert len(wg_findings) == 0

    def test_goroutine_with_context(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "import \"context\"\n\n"
            "func f(ctx context.Context) {\n"
            "    go func() {\n"
            "        select {\n"
            "        case <-ctx.Done():\n"
            "            return\n"
            "        }\n"
            "    }()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        leak_findings = [f for f in findings if "without WaitGroup" in f.title]
        assert len(leak_findings) == 0

    def test_goroutine_with_waitgroup(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "import \"sync\"\n\n"
            "func f() {\n"
            "    var wg sync.WaitGroup\n"
            "    wg.Add(1)\n"
            "    go func() {\n"
            "        defer wg.Done()\n"
            "        work()\n"
            "    }()\n"
            "    wg.Wait()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        leak_findings = [f for f in findings if "without WaitGroup" in f.title]
        assert len(leak_findings) == 0

    def test_non_go_language(self, agent: GoGoroutineSafetyAgent) -> None:
        src = "import threading\nt = threading.Thread(target=worker)\nt.start()\n"
        assert agent.analyze(_ctx(src, "python", "t.py")) == []

    def test_empty_source(self, agent: GoGoroutineSafetyAgent) -> None:
        assert agent.analyze(_ctx("", "go", "t.go")) == []

    def test_whitespace_only(self, agent: GoGoroutineSafetyAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ", "go", "t.go")) == []

    def test_no_goroutines_no_locks(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "func main() {\n"
            "    x := 1\n"
            "    y := x + 2\n"
            "    fmt.Println(y)\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_buffered_channel_no_deadlock(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    ch := make(chan int, 1)\n"
            "    ch <- 42\n"
            "    val := <-ch\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        deadlock_findings = [f for f in findings if "deadlock" in f.title.lower()]
        assert len(deadlock_findings) == 0

    def test_channel_with_goroutine(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "func f(ctx context.Context) {\n"
            "    ch := make(chan int)\n"
            "    go func() {\n"
            "        ch <- 42\n"
            "    }()\n"
            "    val := <-ch\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        deadlock_findings = [f for f in findings if "deadlock" in f.title.lower()]
        assert len(deadlock_findings) == 0

    def test_javascript_ignored(self, agent: GoGoroutineSafetyAgent) -> None:
        src = "async function f() { await fetch('/api'); }\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_java_ignored(self, agent: GoGoroutineSafetyAgent) -> None:
        src = "public class T { void f() { new Thread(r).start(); } }\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []


# ── Edge cases & metadata ────────────────────────────────────────────────


class TestEdgeCases:
    def test_metadata_name(self, agent: GoGoroutineSafetyAgent) -> None:
        m = agent.metadata()
        assert m.name == "go_goroutine_safety"

    def test_metadata_version(self, agent: GoGoroutineSafetyAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages(self, agent: GoGoroutineSafetyAgent) -> None:
        m = agent.metadata()
        assert m.languages == ["go"]

    def test_metadata_domains(self, agent: GoGoroutineSafetyAgent) -> None:
        m = agent.metadata()
        assert set(m.domains) == {"systems_programming", "distributed_systems"}

    def test_metadata_methodology(self, agent: GoGoroutineSafetyAgent) -> None:
        m = agent.metadata()
        assert m.methodology == "language_specific"

    def test_metadata_axis_type_critical(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        m = agent.metadata()
        assert m.axis_type == "critical"

    def test_metadata_model_not_required(
        self, agent: GoGoroutineSafetyAgent
    ) -> None:
        m = agent.metadata()
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_metadata_tags(self, agent: GoGoroutineSafetyAgent) -> None:
        m = agent.metadata()
        assert "go" in m.tags
        assert "goroutine" in m.tags
        assert "concurrency" in m.tags

    def test_explain_returns_string(self, agent: GoGoroutineSafetyAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    go func() { work() }()\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_unsupported_language(self, agent: GoGoroutineSafetyAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []
