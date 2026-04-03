"""Test Coverage Gap Detector -- flags complex functions lacking test coverage."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, walk
from fathom_sdk.context.ast_parser import parse

# Decision node types that contribute to function complexity, per language.
# A function with more than _COMPLEXITY_THRESHOLD of these is considered
# "complex enough to warrant dedicated tests".
_DECISION_TYPES: dict[str, set[str]] = {
    "python": {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "try_statement",
    },
    "javascript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "try_statement",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "try_statement",
    },
    "java": {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "try_statement",
    },
    "go": {
        "if_statement",
        "for_statement",
    },
}

_COMPLEXITY_THRESHOLD = 5


def _count_decision_points(func_node, decision_types: set[str]) -> int:
    """Count the number of decision-point nodes inside a function body."""
    count = 0
    for node in walk(func_node):
        if node is func_node:
            continue
        if node.type in decision_types:
            count += 1
    return count


def _is_test_function(name: str) -> bool:
    """Return True if the function name looks like a test function."""
    lower = name.lower()
    return (
        lower.startswith("test_")
        or lower.startswith("test")
        and (len(name) == 4 or not name[4:5].islower())
    )


def _is_test_file(file_path: str) -> bool:
    """Return True if the file path indicates a test file."""
    lower = file_path.lower()
    return "test" in lower


class TestCoverageGapAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="test_coverage_gap",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["quality_engineering", "web_development"],
            methodology="testing",
            axis_type="aware",
            tags=["testing", "coverage", "quality"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []

        # Skip test files -- they are tests themselves, not production code.
        if _is_test_file(context.file_path):
            return []

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        lang = context.language.lower()
        decision_types = _DECISION_TYPES.get(lang, set())
        if not decision_types:
            return []

        functions = find_functions(root, context.language)
        if not functions:
            return []

        # Collect the names of all test functions in the file (if any).
        test_func_names: set[str] = set()
        all_func_names: set[str] = set()
        for func in functions:
            all_func_names.add(func.name)
            if _is_test_function(func.name):
                test_func_names.add(func.name)

        # Build a set of function names that appear to be referenced by
        # test functions (simple heuristic: the production function name
        # is a substring of a test function name).
        tested_names: set[str] = set()
        for prod_name in all_func_names:
            if _is_test_function(prod_name):
                continue
            for test_name in test_func_names:
                if prod_name.lower() in test_name.lower():
                    tested_names.add(prod_name)
                    break

        findings: list[Finding] = []
        for func in functions:
            # Skip test functions themselves.
            if _is_test_function(func.name):
                continue

            decision_count = _count_decision_points(func.node, decision_types)
            if decision_count <= _COMPLEXITY_THRESHOLD:
                continue

            # If the function appears in the tested set, skip it.
            if func.name in tested_names:
                continue

            findings.append(
                Finding(
                    agent_name="test_coverage_gap",
                    severity="low",
                    category="testing",
                    title=(
                        f"Complex function '{func.name}' has no corresponding test "
                        f"({decision_count} decision points)"
                    ),
                    description=(
                        f"Function '{func.name}' (lines {func.start_line}-"
                        f"{func.end_line}) contains {decision_count} decision "
                        f"points (if/for/while/try) but no test function in this "
                        f"file references it. Consider adding dedicated tests to "
                        f"cover its branching logic."
                    ),
                    file_path=context.file_path,
                    line_start=func.start_line,
                    line_end=func.end_line,
                    confidence=0.68,
                    tags=["testing", "coverage", "quality"],
                )
            )

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Functions with many decision points have more "
            f"execution paths and are more likely to harbour bugs. Without "
            f"dedicated tests, regressions in these paths may go undetected."
        )
