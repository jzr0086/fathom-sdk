"""Tests for IntegerOverflowAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext

from agents.cluster_03_bug_detection.integer_overflow.agent import IntegerOverflowAgent


@pytest.fixture
def agent() -> IntegerOverflowAgent:
    return IntegerOverflowAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    return CodeContext(source_code=source, language=language, file_path=fp)


# -----------------------------------------------------------------------
# Positive cases (should detect overflow risk)
# -----------------------------------------------------------------------


class TestPositiveCases:
    def test_java_parseint_multiply(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = Integer.parseInt(input) * 1024;\n"
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_java_parseint_add(self, agent: IntegerOverflowAgent) -> None:
        src = "int total = Integer.parseInt(s) + offset;\n"
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_go_atoi_in_arithmetic(self, agent: IntegerOverflowAgent) -> None:
        src = 'package main\n\nimport "strconv"\n\nfunc f(s string) int {\n    n, _ := strconv.Atoi(s)\n    return n * 1000\n}\n'
        findings = agent.analyze(_ctx(src, "go", "t.go"))
        assert len(findings) >= 1

    def test_js_parseint_multiply(self, agent: IntegerOverflowAgent) -> None:
        src = "let total = parseInt(qty) * price;\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 1

    def test_python_user_input_arithmetic(self, agent: IntegerOverflowAgent) -> None:
        src = "amount = int(request.args.get('qty')) * price\n"
        findings = agent.analyze(_ctx(src, "python", "t.py"))
        assert len(findings) >= 1

    def test_python_input_arithmetic(self, agent: IntegerOverflowAgent) -> None:
        src = "result = int(input('Enter value: ')) * multiplier\n"
        findings = agent.analyze(_ctx(src, "python", "t.py"))
        assert len(findings) >= 1

    def test_java_max_value_addition(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = Integer.MAX_VALUE + 1;\n"
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_java_max_value_multiply(self, agent: IntegerOverflowAgent) -> None:
        src = "long y = Long.MAX_VALUE * 2;\n"
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_go_maxint_addition(self, agent: IntegerOverflowAgent) -> None:
        src = "x := math.MaxInt64 + 1\n"
        findings = agent.analyze(_ctx(src, "go", "t.go"))
        assert len(findings) >= 1

    def test_java_triple_multiply(self, agent: IntegerOverflowAgent) -> None:
        src = "int result = width * height * depth;\n"
        findings = agent.analyze(_ctx(src, "java", "T.java"))
        assert len(findings) >= 1

    def test_go_triple_multiply(self, agent: IntegerOverflowAgent) -> None:
        src = "result := rows * cols * layers\n"
        findings = agent.analyze(_ctx(src, "go", "t.go"))
        assert len(findings) >= 1

    def test_js_request_parseint_arith(self, agent: IntegerOverflowAgent) -> None:
        src = "let total = parseInt(req.params.qty) * unitPrice;\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 1

    def test_js_number_request_arith(self, agent: IntegerOverflowAgent) -> None:
        src = "let size = Number(req.body.count) * blockSize;\n"
        findings = agent.analyze(_ctx(src, "javascript", "t.js"))
        assert len(findings) >= 1

    def test_severity_medium(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = Integer.parseInt(input) * 1024;\n"
        f = agent.analyze(_ctx(src, "java", "T.java"))[0]
        assert f.severity == "medium"
        assert f.category == "bug"

    def test_confidence_072(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = Integer.parseInt(input) * 1024;\n"
        f = agent.analyze(_ctx(src, "java", "T.java"))[0]
        assert f.confidence == pytest.approx(0.72)


# -----------------------------------------------------------------------
# Negative cases (should NOT detect overflow risk)
# -----------------------------------------------------------------------


class TestNegativeCases:
    def test_java_biginteger(self, agent: IntegerOverflowAgent) -> None:
        src = "BigInteger total = BigInteger.valueOf(x).multiply(BigInteger.valueOf(y));\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_java_math_multiply_exact(self, agent: IntegerOverflowAgent) -> None:
        src = "int result = Math.multiplyExact(a, b);\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_python_arbitrary_precision(self, agent: IntegerOverflowAgent) -> None:
        # Python int has arbitrary precision; plain int arithmetic is safe
        src = "x = 10\ny = x * 999999999999\n"
        assert agent.analyze(_ctx(src, "python", "t.py")) == []

    def test_bounded_arithmetic(self, agent: IntegerOverflowAgent) -> None:
        src = "if (x < MAX_VALUE) {\n    int result = x * factor;\n}\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_simple_constants(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = 2 * 3 * 4;\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_empty(self, agent: IntegerOverflowAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_comment_line(self, agent: IntegerOverflowAgent) -> None:
        src = "// int x = Integer.parseInt(input) * 1024;\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_go_big_int(self, agent: IntegerOverflowAgent) -> None:
        src = "x := new(big.Int).Mul(a, b)  // math/big\n"
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_no_arithmetic(self, agent: IntegerOverflowAgent) -> None:
        src = "String name = scanner.nextLine();\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_print_only(self, agent: IntegerOverflowAgent) -> None:
        src = "System.out.println(42);\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_python_simple_int(self, agent: IntegerOverflowAgent) -> None:
        # Plain Python int() is arbitrary precision — no overflow risk
        src = "x = int('42')\nprint(x)\n"
        assert agent.analyze(_ctx(src, "python", "t.py")) == []

    def test_unsupported_language(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = 42 * 100;\n"
        assert agent.analyze(_ctx(src, "cobol", "t.cob")) == []

    def test_java_constant_only_triple(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = 10 * 20 * 30;\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_metadata_name(self, agent: IntegerOverflowAgent) -> None:
        m = agent.metadata()
        assert m.name == "integer_overflow"

    def test_metadata_version(self, agent: IntegerOverflowAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_tags(self, agent: IntegerOverflowAgent) -> None:
        m = agent.metadata()
        assert "bug" in m.tags
        assert "overflow" in m.tags
        assert "integer" in m.tags

    def test_metadata_axis_type(self, agent: IntegerOverflowAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "aware"

    def test_metadata_model_not_required(self, agent: IntegerOverflowAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False

    def test_metadata_zero_cost(self, agent: IntegerOverflowAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = Integer.parseInt(input) * 1024;\n"
        f = agent.analyze(_ctx(src, "java", "T.java"))[0]
        explanation = agent.explain(f)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_finding_file_path(self, agent: IntegerOverflowAgent) -> None:
        src = "int x = Integer.parseInt(input) * 1024;\n"
        f = agent.analyze(_ctx(src, "java", "Calc.java"))[0]
        assert f.file_path == "Calc.java"

    def test_whitespace_only(self, agent: IntegerOverflowAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []
