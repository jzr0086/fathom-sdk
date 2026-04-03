"""Cognitive Complexity — nesting-weighted complexity scoring."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text, walk
from fathom_sdk.context.ast_parser import parse

# Node types that increment cognitive complexity AND increase nesting
_NESTING_TYPES: dict[str, set[str]] = {
    "python": {"if_statement", "for_statement", "while_statement", "try_statement"},
    "javascript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "try_statement",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "switch_statement",
        "try_statement",
    },
    "java": {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "switch_expression",
        "try_statement",
    },
    "go": {"if_statement", "for_statement", "switch_statement", "select_statement"},
}

# Node types that increment but do NOT increase nesting (else, elif, catch)
_INCREMENT_ONLY: dict[str, set[str]] = {
    "python": {"elif_clause", "else_clause", "except_clause"},
    "javascript": {"else_clause", "catch_clause"},
    "typescript": {"else_clause", "catch_clause"},
    "java": {"else_clause", "catch_clause"},
    "go": set(),
}

# Boolean sequences: count once per sequence
_BOOLEAN_TYPES: dict[str, set[str]] = {
    "python": {"boolean_operator"},
    "javascript": {"binary_expression"},
    "typescript": {"binary_expression"},
    "java": {"binary_expression"},
    "go": {"binary_expression"},
}

_THRESHOLDS = [(25, "high"), (15, "medium"), (10, "low")]


def _compute_cognitive(func_node, language: str) -> int:
    """Compute cognitive complexity for a function node."""
    lang = language.lower()
    nesting_types = _NESTING_TYPES.get(lang, set())
    increment_only = _INCREMENT_ONLY.get(lang, set())
    boolean_types = _BOOLEAN_TYPES.get(lang, set())

    score = 0

    def _visit(node, nesting: int) -> None:
        nonlocal score

        if node.type in nesting_types:
            score += 1 + nesting
            for child in node.children:
                if child.is_named:
                    _visit(child, nesting + 1)
            return

        if node.type in increment_only:
            score += 1
            for child in node.children:
                if child.is_named:
                    _visit(child, nesting)
            return

        if node.type in boolean_types:
            # For non-Python: only count && and || operators
            if lang != "python":
                op = node.child_by_field_name("operator")
                if op is not None and get_node_text(op) in ("&&", "||"):
                    # Only count if parent is not also a boolean expression
                    # (sequence counts as 1)
                    parent = node.parent
                    if parent is None or parent.type not in boolean_types:
                        score += 1
            else:
                # Python boolean_operator: only count if parent isn't one too
                parent = node.parent
                if parent is None or parent.type != "boolean_operator":
                    score += 1

        for child in node.children:
            if child.is_named:
                _visit(child, nesting)

    # Start visiting from function body children
    for child in func_node.children:
        if child.is_named:
            _visit(child, 0)
    return score


class CognitiveComplexityAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="cognitive_complexity",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="code_quality",
            axis_type="aware",
            tags=["quality", "complexity", "cognitive"],
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

        lang = context.language.lower()
        if lang not in _NESTING_TYPES:
            return []

        findings: list[Finding] = []
        for func in find_functions(root, context.language):
            score = _compute_cognitive(func.node, context.language)
            for threshold, severity in _THRESHOLDS:
                if score > threshold:
                    findings.append(
                        Finding(
                            agent_name="cognitive_complexity",
                            severity=severity,
                            category="quality",
                            title=f"High cognitive complexity in '{func.name}' ({score})",
                            description=(
                                f"Function '{func.name}' has cognitive complexity {score} "
                                f"(threshold: {threshold}). Deeply nested logic is hard to "
                                f"understand and maintain."
                            ),
                            file_path=context.file_path,
                            line_start=func.start_line,
                            line_end=func.end_line,
                            confidence=0.95,
                            tags=["quality", "complexity", "cognitive"],
                        )
                    )
                    break
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Cognitive complexity penalizes deeply nested logic. "
            f"Flatten with early returns, extract nested conditions into helper "
            f"functions, or use guard clauses."
        )
