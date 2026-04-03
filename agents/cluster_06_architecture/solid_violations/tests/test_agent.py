"""Tests for SolidViolationsAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_06_architecture.solid_violations.agent import SolidViolationsAgent


@pytest.fixture
def agent() -> SolidViolationsAgent:
    return SolidViolationsAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ---------------------------------------------------------------------------
# Helpers — generate classes with N methods
# ---------------------------------------------------------------------------


def _python_class(name: str, method_count: int, *, init_params: int = 0) -> str:
    """Generate a Python class with *method_count* methods."""
    lines = [f"class {name}:"]
    if init_params > 0:
        params = ", ".join(f"p{i}" for i in range(init_params))
        lines.append(f"    def __init__(self, {params}):")
        lines.append("        pass")
    for i in range(method_count):
        lines.append(f"    def method_{i}(self):")
        lines.append(f"        return {i}")
    return "\n".join(lines) + "\n"


def _java_class(name: str, method_count: int, *, ctor_params: int = 0) -> str:
    """Generate a Java class with *method_count* methods."""
    lines = [f"public class {name} {{"]
    if ctor_params > 0:
        params = ", ".join(f"int p{i}" for i in range(ctor_params))
        lines.append(f"    public {name}({params}) {{}}")
    for i in range(method_count):
        lines.append(f"    public int method_{i}() {{ return {i}; }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _js_class(name: str, method_count: int, *, ctor_params: int = 0) -> str:
    """Generate a JavaScript class with *method_count* methods."""
    lines = [f"class {name} {{"]
    if ctor_params > 0:
        params = ", ".join(f"p{i}" for i in range(ctor_params))
        lines.append(f"    constructor({params}) {{}}")
    for i in range(method_count):
        lines.append(f"    method_{i}() {{ return {i}; }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _ts_class(name: str, method_count: int, *, ctor_params: int = 0) -> str:
    """Generate a TypeScript class with *method_count* methods."""
    lines = [f"class {name} {{"]
    if ctor_params > 0:
        params = ", ".join(f"p{i}: number" for i in range(ctor_params))
        lines.append(f"    constructor({params}) {{}}")
    for i in range(method_count):
        lines.append(f"    method_{i}(): number {{ return {i}; }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# =========================================================================
# Positive cases (10+)
# =========================================================================


class TestPositiveCases:
    # --- SRP violations ---

    def test_python_god_class_srp(self, agent: SolidViolationsAgent) -> None:
        """A Python class with 12 methods should trigger SRP."""
        ctx = _ctx(_python_class("GodClass", 12))
        findings = agent.analyze(ctx)
        srp = [f for f in findings if "srp" in f.tags]
        assert len(srp) == 1
        assert srp[0].severity == "high"
        assert srp[0].confidence == 0.75

    def test_java_god_class_srp(self, agent: SolidViolationsAgent) -> None:
        """A Java class with 15 methods should trigger SRP."""
        ctx = _ctx(_java_class("GodService", 15), "java", "GodService.java")
        srp = [f for f in agent.analyze(ctx) if "srp" in f.tags]
        assert len(srp) == 1
        assert srp[0].severity == "high"

    def test_js_god_class_srp(self, agent: SolidViolationsAgent) -> None:
        """A JavaScript class with 11 methods should trigger SRP."""
        ctx = _ctx(_js_class("BigController", 11), "javascript", "big.js")
        srp = [f for f in agent.analyze(ctx) if "srp" in f.tags]
        assert len(srp) == 1

    def test_ts_god_class_srp(self, agent: SolidViolationsAgent) -> None:
        """A TypeScript class with 11 methods should trigger SRP."""
        ctx = _ctx(_ts_class("HugeService", 11), "typescript", "huge.ts")
        srp = [f for f in agent.analyze(ctx) if "srp" in f.tags]
        assert len(srp) == 1

    # --- ISP violations ---

    def test_python_fat_interface_isp(self, agent: SolidViolationsAgent) -> None:
        """A Python class with 9 methods triggers ISP (threshold >7)."""
        ctx = _ctx(_python_class("FatInterface", 9))
        isp = [f for f in agent.analyze(ctx) if "isp" in f.tags]
        assert len(isp) == 1
        assert isp[0].severity == "medium"
        assert isp[0].confidence == 0.70

    def test_java_fat_interface_isp(self, agent: SolidViolationsAgent) -> None:
        """A Java class with 8 methods triggers ISP."""
        ctx = _ctx(_java_class("BigRepo", 8), "java", "BigRepo.java")
        isp = [f for f in agent.analyze(ctx) if "isp" in f.tags]
        assert len(isp) == 1

    # --- DIP violations ---

    def test_python_dip_many_ctor_params(self, agent: SolidViolationsAgent) -> None:
        """Python __init__ with 7 params (including self) triggers DIP."""
        ctx = _ctx(_python_class("Tightly", 2, init_params=6))
        dip = [f for f in agent.analyze(ctx) if "dip" in f.tags]
        assert len(dip) == 1
        assert dip[0].severity == "medium"
        assert dip[0].confidence == 0.70

    def test_js_dip_many_ctor_params(self, agent: SolidViolationsAgent) -> None:
        """JS constructor with 7 params triggers DIP."""
        ctx = _ctx(_js_class("Service", 2, ctor_params=7), "javascript", "svc.js")
        dip = [f for f in agent.analyze(ctx) if "dip" in f.tags]
        assert len(dip) == 1

    def test_ts_dip_many_ctor_params(self, agent: SolidViolationsAgent) -> None:
        """TS constructor with 6 params triggers DIP."""
        ctx = _ctx(_ts_class("Handler", 2, ctor_params=6), "typescript", "h.ts")
        dip = [f for f in agent.analyze(ctx) if "dip" in f.tags]
        assert len(dip) == 1

    # --- Combined violations ---

    def test_srp_and_isp_combined(self, agent: SolidViolationsAgent) -> None:
        """A class with 12 methods triggers both SRP (>10) and ISP (>7)."""
        ctx = _ctx(_python_class("Bloated", 12))
        findings = agent.analyze(ctx)
        srp = [f for f in findings if "srp" in f.tags]
        isp = [f for f in findings if "isp" in f.tags]
        assert len(srp) == 1
        assert len(isp) == 1

    def test_all_three_violations(self, agent: SolidViolationsAgent) -> None:
        """A class with 12 methods and a fat constructor triggers SRP, ISP, and DIP."""
        ctx = _ctx(_python_class("Monster", 12, init_params=7))
        findings = agent.analyze(ctx)
        tags_found = {tag for f in findings for tag in f.tags}
        assert "srp" in tags_found
        assert "isp" in tags_found
        assert "dip" in tags_found

    def test_finding_fields(self, agent: SolidViolationsAgent) -> None:
        """Verify core finding fields for an SRP violation."""
        ctx = _ctx(_python_class("Big", 12), fp="src/big.py")
        srp = [f for f in agent.analyze(ctx) if "srp" in f.tags]
        assert len(srp) == 1
        f = srp[0]
        assert f.agent_name == "solid_violations"
        assert f.category == "architecture"
        assert f.file_path == "src/big.py"
        assert f.line_start >= 1
        assert f.line_end >= f.line_start


# =========================================================================
# Negative cases (10+)
# =========================================================================


class TestNegativeCases:
    def test_small_class_no_findings(self, agent: SolidViolationsAgent) -> None:
        """A class with 3 methods should produce no findings."""
        ctx = _ctx(_python_class("Tiny", 3))
        assert agent.analyze(ctx) == []

    def test_exactly_10_methods_no_srp(self, agent: SolidViolationsAgent) -> None:
        """Exactly 10 methods should NOT trigger SRP (threshold is >10)."""
        ctx = _ctx(_python_class("Borderline", 10))
        srp = [f for f in agent.analyze(ctx) if "srp" in f.tags]
        assert len(srp) == 0

    def test_exactly_7_methods_no_isp(self, agent: SolidViolationsAgent) -> None:
        """Exactly 7 methods should NOT trigger ISP (threshold is >7)."""
        ctx = _ctx(_python_class("Border", 7))
        isp = [f for f in agent.analyze(ctx) if "isp" in f.tags]
        assert len(isp) == 0

    def test_5_total_params_no_dip(self, agent: SolidViolationsAgent) -> None:
        """4 explicit params + self = 5 total params; should NOT trigger DIP (>5)."""
        ctx = _ctx(_python_class("OkClass", 2, init_params=4))
        dip = [f for f in agent.analyze(ctx) if "dip" in f.tags]
        assert len(dip) == 0

    def test_empty_source(self, agent: SolidViolationsAgent) -> None:
        """Empty source code produces no findings."""
        assert agent.analyze(_ctx("")) == []

    def test_no_classes(self, agent: SolidViolationsAgent) -> None:
        """Source with only functions (no classes) produces no findings."""
        src = "def f():\n    return 1\n\ndef g():\n    return 2\n"
        assert agent.analyze(_ctx(src)) == []

    def test_single_method_class(self, agent: SolidViolationsAgent) -> None:
        """A class with a single method is clean."""
        src = "class Simple:\n    def run(self):\n        return True\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_class(self, agent: SolidViolationsAgent) -> None:
        """An empty class with a pass body produces no findings."""
        src = "class Empty:\n    pass\n"
        assert agent.analyze(_ctx(src)) == []

    def test_java_small_class(self, agent: SolidViolationsAgent) -> None:
        """A small Java class produces no findings."""
        ctx = _ctx(_java_class("SmallService", 3), "java", "Small.java")
        assert agent.analyze(ctx) == []

    def test_js_small_class(self, agent: SolidViolationsAgent) -> None:
        """A small JavaScript class produces no findings."""
        ctx = _ctx(_js_class("SmallCtrl", 3), "javascript", "small.js")
        assert agent.analyze(ctx) == []

    def test_unsupported_language(self, agent: SolidViolationsAgent) -> None:
        """Unsupported language returns no findings."""
        assert agent.analyze(_ctx("class Foo {}", "cobol", "t.cob")) == []

    def test_multiple_small_classes(self, agent: SolidViolationsAgent) -> None:
        """Multiple small classes produce no findings."""
        src = (
            _python_class("A", 2)
            + "\n"
            + _python_class("B", 3)
            + "\n"
            + _python_class("C", 1)
        )
        assert agent.analyze(_ctx(src)) == []


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_metadata(self, agent: SolidViolationsAgent) -> None:
        m = agent.metadata()
        assert m.name == "solid_violations"
        assert m.version == "0.1.0"
        assert m.languages == ["*"]
        assert m.axis_type == "agnostic"
        assert m.methodology == "architecture"
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0
        assert "solid" in m.tags

    def test_explain_srp(self, agent: SolidViolationsAgent) -> None:
        ctx = _ctx(_python_class("Big", 12))
        srp = [f for f in agent.analyze(ctx) if "srp" in f.tags][0]
        explanation = agent.explain(srp)
        assert "Single Responsibility" in explanation
        assert len(explanation) > 20

    def test_explain_isp(self, agent: SolidViolationsAgent) -> None:
        ctx = _ctx(_python_class("Fat", 9))
        isp = [f for f in agent.analyze(ctx) if "isp" in f.tags][0]
        explanation = agent.explain(isp)
        assert "Interface Segregation" in explanation

    def test_explain_dip(self, agent: SolidViolationsAgent) -> None:
        ctx = _ctx(_python_class("Coupled", 2, init_params=7))
        dip = [f for f in agent.analyze(ctx) if "dip" in f.tags][0]
        explanation = agent.explain(dip)
        assert "Dependency Inversion" in explanation

    def test_boundary_11_methods_triggers_srp(self, agent: SolidViolationsAgent) -> None:
        """11 methods is exactly one above the threshold; should trigger SRP."""
        ctx = _ctx(_python_class("Boundary", 11))
        srp = [f for f in agent.analyze(ctx) if "srp" in f.tags]
        assert len(srp) == 1

    def test_boundary_8_methods_triggers_isp(self, agent: SolidViolationsAgent) -> None:
        """8 methods is exactly one above the ISP threshold."""
        ctx = _ctx(_python_class("Boundary", 8))
        isp = [f for f in agent.analyze(ctx) if "isp" in f.tags]
        assert len(isp) == 1

    def test_two_classes_one_big_one_small(self, agent: SolidViolationsAgent) -> None:
        """Only the large class should produce findings."""
        src = _python_class("Big", 12) + "\n" + _python_class("Small", 2)
        ctx = _ctx(src)
        findings = agent.analyze(ctx)
        names_in_findings = {f.title for f in findings}
        assert any("Big" in t for t in names_in_findings)
        assert all("Small" not in t for t in names_in_findings)
