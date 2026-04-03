"""Tests for JavaConcurrencyAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_10_language_specific.java_concurrency.agent import (
    JavaConcurrencyAgent,
)


@pytest.fixture
def agent() -> JavaConcurrencyAgent:
    return JavaConcurrencyAgent()


def _ctx(source: str, language: str = "java", fp: str = "Test.java") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp, ast=None)


# =========================================================================
# Positive cases (10+)
# =========================================================================


class TestPositiveCases:
    def test_synchronized_on_string_literal(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Cache {\n"
            '    public void update() {\n'
            '        synchronized("LOCK") {\n'
            "            // critical section\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "critical" for f in findings)
        assert any("string literal" in f.title.lower() for f in findings)

    def test_synchronized_on_string_variable(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Cache {\n"
            '    String lockKey = "myLock";\n'
            "    public void update() {\n"
            "        synchronized(lockKey) {\n"
            "            // critical section\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "critical" for f in findings)

    def test_double_checked_locking_no_volatile(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Singleton {\n"
            "    private static Singleton instance;\n"
            "    public static Singleton getInstance() {\n"
            "        if (instance == null) {\n"
            "            synchronized(Singleton.class) {\n"
            "                if (instance == null) {\n"
            "                    instance = new Singleton();\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        return instance;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "high" for f in findings)
        assert any("double-checked" in f.title.lower() for f in findings)

    def test_volatile_increment(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Counter {\n"
            "    volatile int count = 0;\n"
            "    public void increment() {\n"
            "        count++;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "high" for f in findings)
        assert any("volatile" in f.title.lower() or "atomic" in f.description.lower() for f in findings)

    def test_volatile_plus_equals(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Counter {\n"
            "    volatile long total = 0;\n"
            "    public void add(long n) {\n"
            "        total += n;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("volatile" in f.title.lower() for f in findings)

    def test_synchronized_method_on_public_class(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Service {\n"
            "    public synchronized void process() {\n"
            "        // work\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "low" for f in findings)
        assert any("synchronized method" in f.title.lower() for f in findings)

    def test_lock_without_try_finally(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Worker {\n"
            "    private final ReentrantLock lock = new ReentrantLock();\n"
            "    public void doWork() {\n"
            "        lock.lock();\n"
            "        process();\n"
            "        lock.unlock();\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "medium" for f in findings)
        assert any("try-finally" in f.title.lower() or "try-finally" in f.description.lower() for f in findings)

    def test_prefix_volatile_increment(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Counter {\n"
            "    volatile int value = 0;\n"
            "    public void inc() {\n"
            "        ++value;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_multiple_findings(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Bad {\n"
            '    String lockStr = "LOCK";\n'
            "    volatile int count = 0;\n"
            "    public synchronized void work() {\n"
            "        count++;\n"
            "        synchronized(lockStr) {\n"
            "            // nested\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        # Should have at least: volatile++, synchronized method, sync on string
        assert len(findings) >= 3

    def test_finding_fields(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Counter {\n"
            "    volatile int count = 0;\n"
            "    public void inc() {\n"
            "        count++;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        f = findings[0]
        assert f.agent_name == "java_concurrency"
        assert f.category == "bug"
        assert f.confidence == 0.82
        assert f.file_path == "Test.java"

    def test_synchronized_on_empty_string_literal(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class X {\n"
            '    void f() { synchronized("") { doWork(); } }\n'
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "critical" for f in findings)


# =========================================================================
# Negative cases (10+)
# =========================================================================


class TestNegativeCases:
    def test_proper_lock_with_try_finally(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Worker {\n"
            "    private final ReentrantLock lock = new ReentrantLock();\n"
            "    public void doWork() {\n"
            "        lock.lock();\n"
            "        try {\n"
            "            process();\n"
            "        } finally {\n"
            "            lock.unlock();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert not any("try-finally" in f.title.lower() for f in findings)

    def test_atomic_integer_usage(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Counter {\n"
            "    private final AtomicInteger count = new AtomicInteger(0);\n"
            "    public void increment() {\n"
            "        count.incrementAndGet();\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 0

    def test_private_lock_object(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Cache {\n"
            "    private final Object lock = new Object();\n"
            "    public void update() {\n"
            "        synchronized(lock) {\n"
            "            // critical section\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        # Should not flag synchronized on a non-String Object
        assert not any("string" in f.title.lower() for f in findings)

    def test_non_java_language(self, agent: JavaConcurrencyAgent) -> None:
        src = "def foo():\n    pass\n"
        assert agent.analyze(_ctx(src, language="python", fp="test.py")) == []

    def test_empty_source(self, agent: JavaConcurrencyAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: JavaConcurrencyAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_go_language(self, agent: JavaConcurrencyAgent) -> None:
        src = "package main\nfunc main() {}\n"
        assert agent.analyze(_ctx(src, language="go", fp="main.go")) == []

    def test_javascript_language(self, agent: JavaConcurrencyAgent) -> None:
        src = "async function work() { await fetch('/api'); }\n"
        assert agent.analyze(_ctx(src, language="javascript", fp="app.js")) == []

    def test_double_checked_with_volatile(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Singleton {\n"
            "    private static volatile Singleton instance;\n"
            "    public static Singleton getInstance() {\n"
            "        if (instance == null) {\n"
            "            synchronized(Singleton.class) {\n"
            "                if (instance == null) {\n"
            "                    instance = new Singleton();\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        return instance;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert not any("double-checked" in f.title.lower() for f in findings)

    def test_no_concurrency_code(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Simple {\n"
            "    public int add(int a, int b) {\n"
            "        return a + b;\n"
            "    }\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_synchronized_on_class_literal(self, agent: JavaConcurrencyAgent) -> None:
        # Synchronized on Foo.class is fine — not a string
        src = (
            "public class Foo {\n"
            "    public void work() {\n"
            "        synchronized(Foo.class) {\n"
            "            // ok\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert not any("string" in f.title.lower() for f in findings)

    def test_non_volatile_field_increment(self, agent: JavaConcurrencyAgent) -> None:
        # Regular (non-volatile) field increment is not flagged by volatile++ detector
        src = (
            "public class Counter {\n"
            "    int count = 0;\n"
            "    public void inc() {\n"
            "        count++;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        assert not any("volatile" in f.title.lower() for f in findings)


# =========================================================================
# Edge cases & metadata
# =========================================================================


class TestEdgeCases:
    def test_metadata_name(self, agent: JavaConcurrencyAgent) -> None:
        m = agent.metadata()
        assert m.name == "java_concurrency"

    def test_metadata_version(self, agent: JavaConcurrencyAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages(self, agent: JavaConcurrencyAgent) -> None:
        m = agent.metadata()
        assert m.languages == ["java"]

    def test_metadata_axis_type(self, agent: JavaConcurrencyAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "critical"

    def test_metadata_model_not_required(self, agent: JavaConcurrencyAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False

    def test_metadata_zero_cost(self, agent: JavaConcurrencyAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_nonempty(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Counter {\n"
            "    volatile int count = 0;\n"
            "    public void inc() {\n"
            "        count++;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src))
        explanation = agent.explain(findings[0])
        assert len(explanation) > 20

    def test_case_insensitive_language(self, agent: JavaConcurrencyAgent) -> None:
        src = (
            "public class Counter {\n"
            "    volatile int count = 0;\n"
            "    public void inc() {\n"
            "        count++;\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, language="Java"))
        assert len(findings) >= 1

    def test_tags_present(self, agent: JavaConcurrencyAgent) -> None:
        m = agent.metadata()
        assert "java" in m.tags
        assert "concurrency" in m.tags
        assert "threading" in m.tags
