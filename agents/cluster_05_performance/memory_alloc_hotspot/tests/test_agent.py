"""Tests for MemoryAllocHotspotAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_05_performance.memory_alloc_hotspot.agent import (
    MemoryAllocHotspotAgent,
)


@pytest.fixture
def agent() -> MemoryAllocHotspotAgent:
    return MemoryAllocHotspotAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# =========================================================================
# Positive cases (10+)
# =========================================================================


class TestPositiveCases:
    # -- Go ---------------------------------------------------------------

    def test_go_append_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    var result []int\n"
            "    for i := 0; i < 100; i++ {\n"
            "        result = append(result, i)\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1
        assert any("append" in f.title for f in findings)

    def test_go_make_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    for i := 0; i < 10; i++ {\n"
            "        s := make([]byte, 1024)\n"
            "        _ = s\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1
        assert any("make" in f.title for f in findings)

    def test_go_new_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    for i := 0; i < 10; i++ {\n"
            "        p := new(int)\n"
            "        _ = p\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1
        assert any("new" in f.title for f in findings)

    # -- Java -------------------------------------------------------------

    def test_java_new_arraylist_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        for (int i = 0; i < 100; i++) {\n"
            "            List<String> list = new ArrayList<>();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_java_new_hashmap_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        for (int i = 0; i < n; i++) {\n"
            "            Map<String, Integer> m = new HashMap<>();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_java_new_stringbuilder_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        for (int i = 0; i < n; i++) {\n"
            "            StringBuilder sb = new StringBuilder();\n"
            "            sb.append(i);\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_java_new_object_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        for (int i = 0; i < 100; i++) {\n"
            "            Object o = new Object();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    # -- Python -----------------------------------------------------------

    def test_python_object_creation_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "class Config:\n"
            "    pass\n\n"
            "for item in items:\n"
            "    cfg = Config()\n"
            "    process(cfg)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("Config" in f.title for f in findings)

    def test_python_regex_compile_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "import re\n"
            "for line in lines:\n"
            "    pattern = Pattern(line)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_python_dataclass_creation_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "for row in rows:\n"
            "    record = Record(row['name'], row['age'])\n"
            "    results.append(record)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    # -- Cross-language ---------------------------------------------------

    def test_severity_medium(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    for i := 0; i < 10; i++ {\n"
            "        s := make([]byte, 1024)\n"
            "        _ = s\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert findings[0].severity == "medium"

    def test_confidence_075(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        for (int i = 0; i < 100; i++) {\n"
            "            List<String> list = new ArrayList<>();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert findings[0].confidence == 0.75


# =========================================================================
# Negative cases (10+)
# =========================================================================


class TestNegativeCases:
    # -- Pre-allocated / outside loop -------------------------------------

    def test_go_prealloc_slice(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    result := make([]int, 0, 100)\n"
            "    for i := 0; i < 100; i++ {\n"
            "        result = append(result, i)\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        # append should not be flagged because pre-allocation exists
        append_findings = [f for f in findings if "append" in f.title]
        assert len(append_findings) == 0

    def test_go_alloc_outside_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    s := make([]int, 10)\n"
            "    for i := 0; i < 10; i++ {\n"
            "        s[i] = i\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        # make is outside the loop — no finding expected
        make_findings = [f for f in findings if "make" in f.title]
        assert len(make_findings) == 0

    def test_java_alloc_outside_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        List<String> list = new ArrayList<>();\n"
            "        for (int i = 0; i < 100; i++) {\n"
            "            list.add(String.valueOf(i));\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) == 0

    def test_python_list_comprehension(self, agent: MemoryAllocHotspotAgent) -> None:
        src = "result = [x * 2 for x in items]\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 0

    def test_python_alloc_outside_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "cfg = Config()\n"
            "for item in items:\n"
            "    process(item, cfg)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 0

    def test_python_builtin_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "for x in items:\n"
            "    n = int(x)\n"
            "    s = str(n)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 0

    def test_empty_source(self, agent: MemoryAllocHotspotAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_no_loops(self, agent: MemoryAllocHotspotAgent) -> None:
        src = "x = 1\ny = Config()\n"
        assert agent.analyze(_ctx(src)) == []

    def test_unsupported_language(self, agent: MemoryAllocHotspotAgent) -> None:
        src = "for (let i = 0; i < 10; i++) { new Map(); }\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_python_exception_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "for item in items:\n"
            "    if not item:\n"
            "        raise ValueError('bad')\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 0

    def test_python_lowercase_call_in_loop(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "for item in items:\n"
            "    result = process(item)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) == 0


# =========================================================================
# Edge cases & metadata
# =========================================================================


class TestEdgeCases:
    def test_metadata_name(self, agent: MemoryAllocHotspotAgent) -> None:
        m = agent.metadata()
        assert m.name == "memory_alloc_hotspot"

    def test_metadata_version(self, agent: MemoryAllocHotspotAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_axis_type(self, agent: MemoryAllocHotspotAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "critical"

    def test_metadata_languages(self, agent: MemoryAllocHotspotAgent) -> None:
        m = agent.metadata()
        assert set(m.languages) == {"python", "java", "go"}

    def test_metadata_no_model_required(self, agent: MemoryAllocHotspotAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False

    def test_metadata_cost_zero(self, agent: MemoryAllocHotspotAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "package main\n\n"
            "func f() {\n"
            "    for i := 0; i < 10; i++ {\n"
            "        s := make([]byte, 1024)\n"
            "        _ = s\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_unsupported_lang_returns_empty(self, agent: MemoryAllocHotspotAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_category_is_performance(self, agent: MemoryAllocHotspotAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        for (int i = 0; i < 10; i++) {\n"
            "            Object o = new Object();\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert findings[0].category == "performance"
