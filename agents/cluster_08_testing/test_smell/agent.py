"""Test Smell Detector — identifies common anti-patterns in test code."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, find_functions, get_node_text, walk
from fathom_sdk.context.ast_parser import parse

# Sleep call patterns across languages
_SLEEP_PATTERNS = re.compile(
    r"\b(?:time\.sleep|sleep|setTimeout|Thread\.sleep)\s*\("
)

# Conditional statement node types per language
_CONDITIONAL_TYPES: dict[str, set[str]] = {
    "python": {"if_statement"},
    "javascript": {"if_statement"},
    "typescript": {"if_statement"},
    "java": {"if_statement"},
    "go": {"if_statement"},
}

# Assertion names used to count assert calls
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
    # JavaScript/TypeScript (jest, mocha, chai)
    "expect",
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

_EAGER_TEST_THRESHOLD = 5
_LONG_TEST_LOC_THRESHOLD = 30


def _is_test_function(name: str) -> bool:
    """Check if a function name matches test function naming conventions."""
    return name.startswith("test_") or name.startswith("Test")


def _count_assertions_in_function(
    func_info, calls: list, source_lines: list[str]
) -> int:
    """Count the number of assertion calls within a test function."""
    count = 0

    # Count AST-based assertion calls within this function
    for call in calls:
        if call.enclosing_function == func_info.name and call.callee_name in _ASSERTION_NAMES:
            count += 1

    # Count bare `assert` statements (Python keyword)
    for line_num in range(
        func_info.start_line,
        min(func_info.end_line + 1, len(source_lines) + 1),
    ):
        line = source_lines[line_num - 1].strip()
        if re.match(r"\bassert\s+", line):
            count += 1

    return count


def _function_has_conditional(func_node, language: str) -> bool:
    """Check if a function body contains if/else conditional statements."""
    cond_types = _CONDITIONAL_TYPES.get(language.lower(), {"if_statement"})
    for node in walk(func_node):
        if node == func_node:
            continue
        if node.type in cond_types:
            return True
    return False


def _function_has_sleep(func_info, source_lines: list[str]) -> bool:
    """Check if a function body contains sleep calls via source scanning."""
    for line_num in range(
        func_info.start_line,
        min(func_info.end_line + 1, len(source_lines) + 1),
    ):
        line = source_lines[line_num - 1]
        if _SLEEP_PATTERNS.search(line):
            return True
    return False


class TestSmellAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="test_smell",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["quality_engineering", "web_development"],
            methodology="testing",
            axis_type="aware",
            tags=["testing", "smell", "quality"],
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

        for func in functions:
            if not _is_test_function(func.name):
                continue

            # Smell 1: Eager test — too many assertions
            self._check_eager_test(
                context, func, calls, source_lines, findings
            )

            # Smell 2: Conditional logic in tests
            self._check_conditional_logic(context, func, findings)

            # Smell 3: Sleep in test
            self._check_sleep_in_test(context, func, source_lines, findings)

            # Smell 4: Long test
            self._check_long_test(context, func, findings)

        return findings

    def _check_eager_test(
        self,
        context: CodeContext,
        func,
        calls: list,
        source_lines: list[str],
        findings: list[Finding],
    ) -> None:
        count = _count_assertions_in_function(func, calls, source_lines)
        if count > _EAGER_TEST_THRESHOLD:
            findings.append(
                Finding(
                    agent_name="test_smell",
                    severity="low",
                    category="testing",
                    title=f"Eager test: '{func.name}' has {count} assertions",
                    description=(
                        f"Test function '{func.name}' at line {func.start_line} "
                        f"contains {count} assertions (threshold: "
                        f"{_EAGER_TEST_THRESHOLD}). Consider splitting into "
                        f"smaller, focused test functions that each verify a "
                        f"single behaviour."
                    ),
                    file_path=context.file_path,
                    line_start=func.start_line,
                    line_end=func.end_line,
                    confidence=0.82,
                    tags=["testing", "smell", "quality"],
                )
            )

    def _check_conditional_logic(
        self,
        context: CodeContext,
        func,
        findings: list[Finding],
    ) -> None:
        if _function_has_conditional(func.node, context.language):
            findings.append(
                Finding(
                    agent_name="test_smell",
                    severity="medium",
                    category="testing",
                    title=f"Conditional logic in test '{func.name}'",
                    description=(
                        f"Test function '{func.name}' at line {func.start_line} "
                        f"contains conditional (if/else) logic. Tests should be "
                        f"deterministic with a predictable execution path. Use "
                        f"parameterized tests or separate test functions instead."
                    ),
                    file_path=context.file_path,
                    line_start=func.start_line,
                    line_end=func.end_line,
                    confidence=0.82,
                    tags=["testing", "smell", "quality"],
                )
            )

    def _check_sleep_in_test(
        self,
        context: CodeContext,
        func,
        source_lines: list[str],
        findings: list[Finding],
    ) -> None:
        if _function_has_sleep(func, source_lines):
            findings.append(
                Finding(
                    agent_name="test_smell",
                    severity="medium",
                    category="testing",
                    title=f"Sleep call in test '{func.name}'",
                    description=(
                        f"Test function '{func.name}' at line {func.start_line} "
                        f"contains a sleep/delay call. Sleeping in tests makes "
                        f"them slow and flaky. Use polling, mocks, or event-based "
                        f"synchronisation instead."
                    ),
                    file_path=context.file_path,
                    line_start=func.start_line,
                    line_end=func.end_line,
                    confidence=0.82,
                    tags=["testing", "smell", "quality"],
                )
            )

    def _check_long_test(
        self,
        context: CodeContext,
        func,
        findings: list[Finding],
    ) -> None:
        if func.line_count > _LONG_TEST_LOC_THRESHOLD:
            findings.append(
                Finding(
                    agent_name="test_smell",
                    severity="low",
                    category="testing",
                    title=f"Long test: '{func.name}' ({func.line_count} lines)",
                    description=(
                        f"Test function '{func.name}' at line {func.start_line} "
                        f"is {func.line_count} lines long (threshold: "
                        f"{_LONG_TEST_LOC_THRESHOLD}). Long tests are harder to "
                        f"understand and maintain. Extract setup logic into "
                        f"fixtures or helper functions."
                    ),
                    file_path=context.file_path,
                    line_start=func.start_line,
                    line_end=func.end_line,
                    confidence=0.82,
                    tags=["testing", "smell", "quality"],
                )
            )

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Test smells make test suites harder to maintain, "
            f"slower to run, and more fragile. Refactor tests to be focused, "
            f"deterministic, and fast."
        )
