"""Assertion Quality Agent — detects assertion anti-patterns in test code."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, find_functions
from fathom_sdk.context.ast_parser import parse

# Patterns for assertTrue with comparison operators (should use assertEqual etc.)
_ASSERT_TRUE_COMPARISON = re.compile(
    r"\bassertTrue\s*\(\s*[^,)]+\s*(==|!=)\s*[^,)]+\s*\)"
)

# Patterns for meaningless assertions
_MEANINGLESS_ASSERT_PATTERNS = [
    re.compile(r"\bassert\s+True\b"),
    re.compile(r"\bassertTrue\s*\(\s*True\s*\)"),
]

# Assertion function/method names used across languages
_ASSERTION_NAMES: set[str] = {
    # Python unittest
    "assertTrue",
    "assertFalse",
    "assertEqual",
    "assertNotEqual",
    "assertIs",
    "assertIsNot",
    "assertIsNone",
    "assertIsNotNone",
    "assertIn",
    "assertNotIn",
    "assertRaises",
    "assertAlmostEqual",
    "assertGreater",
    "assertGreaterEqual",
    "assertLess",
    "assertLessEqual",
    "assertRegex",
    "assertCountEqual",
    # pytest
    "assert",
    # JavaScript/TypeScript (jest, mocha, chai)
    "expect",
    "assert",
    "deepStrictEqual",
    "strictEqual",
    "notStrictEqual",
    "throws",
    "doesNotThrow",
    "ok",
    "fail",
    # Java (JUnit)
    "assertEquals",
    "assertNotEquals",
    "assertNull",
    "assertNotNull",
    "assertSame",
    "assertNotSame",
    "assertArrayEquals",
    "assertThrows",
    # Go (testing)
    "Equal",
    "NotEqual",
    "Nil",
    "NotNil",
    "True",
    "False",
    "Error",
    "NoError",
    "Contains",
    "Fail",
    "FailNow",
    "Fatal",
    "Fatalf",
    "Errorf",
}


def _is_test_function(name: str) -> bool:
    """Check if a function name matches test function naming conventions."""
    return name.startswith("test_") or name.startswith("Test")


def _function_has_assertion(func_info, calls: list, source_lines: list[str]) -> bool:
    """Check whether a function body contains any assertion calls or assert statements."""
    # Check AST-based calls within this function
    for call in calls:
        if call.enclosing_function == func_info.name and call.callee_name in _ASSERTION_NAMES:
            return True

    # Check for bare `assert` statements (Python keyword, not a function call)
    for line_num in range(func_info.start_line, min(func_info.end_line + 1, len(source_lines) + 1)):
        line = source_lines[line_num - 1].strip()
        if re.match(r"\bassert\s+", line):
            return True

    # Check for Go t.Error/t.Fatal style calls
    for call in calls:
        if call.enclosing_function == func_info.name and call.callee_name in (
            "Error",
            "Errorf",
            "Fatal",
            "Fatalf",
            "Fail",
            "FailNow",
        ):
            if "t." in call.full_text or "assert." in call.full_text:
                return True

    return False


class AssertionQualityAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="assertion_quality",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["quality_engineering", "web_development"],
            methodology="testing",
            axis_type="aware",
            tags=["testing", "assertions", "quality"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []
        findings: list[Finding] = []
        source_lines = context.source_code.splitlines()
        functions = find_functions(root, context.language)
        calls = find_calls(root, context.language)

        # Anti-pattern 1: assertTrue with comparison operators
        self._check_assert_true_comparison(context, source_lines, findings)

        # Anti-pattern 2: Test functions with zero assertions
        self._check_no_assertions(
            context, functions, calls, source_lines, findings
        )

        # Anti-pattern 3: Meaningless assertions (assert True, assertTrue(True))
        self._check_meaningless_assertions(context, source_lines, findings)

        return findings

    def _check_assert_true_comparison(
        self,
        context: CodeContext,
        source_lines: list[str],
        findings: list[Finding],
    ) -> None:
        for i, line in enumerate(source_lines, start=1):
            if _ASSERT_TRUE_COMPARISON.search(line):
                if "==" in line:
                    suggestion = "assertEqual/assertEquals"
                else:
                    suggestion = "assertNotEqual/assertNotEquals"
                findings.append(
                    Finding(
                        agent_name="assertion_quality",
                        severity="low",
                        category="testing",
                        title=f"assertTrue with comparison operator on line {i}",
                        description=(
                            f"Use {suggestion} instead of assertTrue with a "
                            f"comparison operator. Specific assertion methods "
                            f"provide better failure messages."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.85,
                        tags=["testing", "assertions", "quality"],
                    )
                )

    def _check_no_assertions(
        self,
        context: CodeContext,
        functions: list,
        calls: list,
        source_lines: list[str],
        findings: list[Finding],
    ) -> None:
        for func in functions:
            if not _is_test_function(func.name):
                continue
            if not _function_has_assertion(func, calls, source_lines):
                findings.append(
                    Finding(
                        agent_name="assertion_quality",
                        severity="medium",
                        category="testing",
                        title=f"Test function '{func.name}' has no assertions",
                        description=(
                            f"Test function '{func.name}' at line {func.start_line} "
                            f"does not contain any assertion calls. A test without "
                            f"assertions verifies nothing and will always pass."
                        ),
                        file_path=context.file_path,
                        line_start=func.start_line,
                        line_end=func.end_line,
                        confidence=0.85,
                        tags=["testing", "assertions", "quality"],
                    )
                )

    def _check_meaningless_assertions(
        self,
        context: CodeContext,
        source_lines: list[str],
        findings: list[Finding],
    ) -> None:
        for i, line in enumerate(source_lines, start=1):
            for pattern in _MEANINGLESS_ASSERT_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        Finding(
                            agent_name="assertion_quality",
                            severity="low",
                            category="testing",
                            title=f"Meaningless assertion on line {i}",
                            description=(
                                f"The assertion on line {i} always passes and "
                                f"does not verify any behaviour. Replace it with "
                                f"a meaningful assertion that checks actual values."
                            ),
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=0.85,
                            tags=["testing", "assertions", "quality"],
                        )
                    )
                    break  # one finding per line

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Weak or missing assertions reduce test effectiveness. "
            f"Use specific assertion methods that provide clear failure messages "
            f"and actually verify the expected behaviour."
        )
