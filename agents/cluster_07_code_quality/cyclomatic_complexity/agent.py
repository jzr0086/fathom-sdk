"""Cyclomatic Complexity — counts decision points per function."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text, walk
from fathom_sdk.context.ast_parser import parse

_DECISION_TYPES: dict[str, set[str]] = {
    "python": {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "except_clause",
        "boolean_operator",
        "conditional_expression",
    },
    "javascript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "switch_case",
        "ternary_expression",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "switch_case",
        "ternary_expression",
    },
    "java": {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "switch_label",
        "ternary_expression",
    },
    "go": {
        "if_statement",
        "for_statement",
        "expression_case",
        "type_case",
        "communication_case",
    },
}

_LOGICAL_OP_TYPES: dict[str, set[str]] = {
    "python": set(),  # boolean_operator already counted above
    "javascript": {"binary_expression"},
    "typescript": {"binary_expression"},
    "java": {"binary_expression"},
    "go": {"binary_expression"},
}

_THRESHOLDS = [(20, "high"), (10, "medium"), (7, "low")]


class CyclomaticComplexityAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="cyclomatic_complexity",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="code_quality",
            axis_type="aware",
            tags=["quality", "complexity", "cyclomatic"],
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
        decision_types = _DECISION_TYPES.get(lang, set())
        logical_types = _LOGICAL_OP_TYPES.get(lang, set())
        if not decision_types:
            return []

        findings: list[Finding] = []
        for func in find_functions(root, context.language):
            complexity = 1  # base complexity
            for node in walk(func.node):
                if node.type in decision_types:
                    # Skip default cases in switch
                    if node.type in ("switch_case", "switch_label"):
                        text = get_node_text(node)
                        if text.strip().startswith("default"):
                            continue
                    complexity += 1
                elif node.type in logical_types:
                    # Count && and || operators
                    op = node.child_by_field_name("operator")
                    if op is not None:
                        op_text = get_node_text(op)
                        if op_text in ("&&", "||"):
                            complexity += 1

            for threshold, severity in _THRESHOLDS:
                if complexity > threshold:
                    findings.append(
                        Finding(
                            agent_name="cyclomatic_complexity",
                            severity=severity,
                            category="quality",
                            title=f"High cyclomatic complexity in '{func.name}' ({complexity})",
                            description=(
                                f"Function '{func.name}' has cyclomatic complexity {complexity} "
                                f"(threshold: {threshold}). High complexity increases bug risk "
                                f"and makes testing difficult."
                            ),
                            file_path=context.file_path,
                            line_start=func.start_line,
                            line_end=func.end_line,
                            confidence=0.95,
                            tags=["quality", "complexity", "cyclomatic"],
                        )
                    )
                    break
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Cyclomatic complexity measures the number of independent "
            f"paths through a function. Reduce it by extracting conditions into helper "
            f"methods or using early returns."
        )
