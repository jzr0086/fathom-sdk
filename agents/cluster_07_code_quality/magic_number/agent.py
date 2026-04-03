"""Magic Number Detector — flags unexplained numeric literals in code."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_numeric_literals, get_node_text
from fathom_sdk.context.ast_parser import parse

# Values that are universally understood and not considered magic.
_ALLOWED_VALUES: set[float] = {0, 1, -1}

# Pattern for UPPER_CASE constant names (e.g. MAX_SIZE, TIMEOUT_SECONDS).
_CONSTANT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _parse_numeric(text: str) -> float | None:
    """Try to convert a literal text to a float for comparison."""
    text = text.strip().replace("_", "")
    # Handle Go/Java suffixes like 0L, 0.0f, etc.
    text = text.rstrip("lLfFdD")
    try:
        return float(text)
    except ValueError:
        return None


def _is_constant_assignment(node, language: str) -> bool:
    """Return True if the numeric literal sits in an UPPER_CASE constant assignment.

    Walks up the tree to find an assignment whose target is an UPPER_CASE name.
    """
    lang = language.lower()
    current = node.parent
    while current is not None:
        # Python: assignment  ->  left = UPPER_CASE
        if current.type in ("assignment", "augmented_assignment", "assignment_expression"):
            left = current.child_by_field_name("left")
            if left is not None and _CONSTANT_NAME_RE.match(get_node_text(left)):
                return True
        # JS/TS/Java: variable_declarator  ->  name = UPPER_CASE
        if current.type == "variable_declarator":
            name_n = current.child_by_field_name("name")
            if name_n is not None and _CONSTANT_NAME_RE.match(get_node_text(name_n)):
                return True
        # Go: short_var_declaration / assignment_statement
        if current.type in ("short_var_declaration", "assignment_statement"):
            left = current.child_by_field_name("left")
            if left is not None and _CONSTANT_NAME_RE.match(get_node_text(left).strip()):
                return True
        # Go: const_spec  (const MAX_SIZE = 100)
        if current.type == "const_spec":
            name_n = current.child_by_field_name("name")
            if name_n is not None and _CONSTANT_NAME_RE.match(get_node_text(name_n)):
                return True
        # Don't look past function boundaries
        if current.type in (
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
            "arrow_function",
        ):
            break
        current = current.parent
    return False


def _is_unary_minus(node) -> bool:
    """Check if the node is inside a unary minus expression (e.g. -1)."""
    parent = node.parent
    if parent is not None and parent.type == "unary_expression":
        op = parent.child_by_field_name("operator")
        if op is not None and get_node_text(op) == "-":
            return True
        # Some grammars put '-' as the first child token
        if parent.child_count >= 2:
            first = parent.children[0]
            if get_node_text(first) == "-":
                return True
    return False


def _effective_value(node, text: str) -> float | None:
    """Return the effective numeric value, accounting for unary negation."""
    val = _parse_numeric(text)
    if val is not None and _is_unary_minus(node):
        val = -val
    return val


class MagicNumberAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="magic_number",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="code_quality",
            axis_type="aware",
            tags=["quality", "magic-number", "readability"],
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
        for lit in find_numeric_literals(root, context.language):
            value = _effective_value(lit.node, lit.value_text)

            # Exclude common sentinel / base values: 0, 1, -1
            if value is not None and value in _ALLOWED_VALUES:
                continue

            # Exclude numbers in UPPER_CASE constant assignments
            if _is_constant_assignment(lit.node, context.language):
                continue

            findings.append(
                Finding(
                    agent_name="magic_number",
                    severity="low",
                    category="quality",
                    title=f"Magic number {lit.value_text} at line {lit.start_line}",
                    description=(
                        f"Numeric literal {lit.value_text} appears without an "
                        f"explanatory name. Extract it into a named constant to "
                        f"improve readability and maintainability."
                    ),
                    file_path=context.file_path,
                    line_start=lit.start_line,
                    line_end=lit.start_line,
                    confidence=0.75,
                    tags=["quality", "magic-number", "readability"],
                )
            )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Magic numbers make code harder to understand "
            f"and maintain. Replace them with well-named constants that convey "
            f"the value's purpose."
        )
