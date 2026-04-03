"""AST Pattern Matching — configurable rule engine for detecting anti-patterns.

Walks the tree-sitter AST and flags common language-specific anti-patterns:

* **Python**: ``== None`` instead of ``is None``, ``== True``/``== False``
  instead of truthiness checks, mutable default arguments.
* **JavaScript / TypeScript**: ``==``/``!=`` instead of ``===``/``!==``
  (type coercion), ``var`` declarations that should use ``let``/``const``.
* **General**: empty except/catch blocks (pattern-based overlap with the
  dedicated exception_swallowing agent).
"""

from __future__ import annotations

import re
from typing import Optional

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import (
    find_assignments,
    find_functions,
    get_node_text,
    walk,
)
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MUTABLE_DEFAULT_RE = re.compile(
    r"^(\[\s*\]|\{\s*\}|set\(\s*\)|dict\(\s*\)|list\(\s*\))$"
)


def _is_mutable_default(value_text: str) -> bool:
    """Return True if *value_text* looks like a mutable default value."""
    return bool(_MUTABLE_DEFAULT_RE.match(value_text.strip()))


# ---------------------------------------------------------------------------
# Per-language rule checkers
# ---------------------------------------------------------------------------


def _check_python_none_comparison(
    root, language: str, file_path: str
) -> list[Finding]:
    """Flag ``== None`` / ``!= None`` — should use ``is None`` / ``is not None``."""
    findings: list[Finding] = []
    for node in walk(root):
        if node.type != "comparison_operator":
            continue
        text = get_node_text(node)
        # Match patterns like `x == None` or `x != None`
        if re.search(r"[!=]=\s*None\b", text) or re.search(
            r"\bNone\s*[!=]=", text
        ):
            findings.append(
                Finding(
                    agent_name="ast_pattern_matching",
                    severity="medium",
                    category="correctness",
                    title="Use 'is None' / 'is not None' instead of '==' / '!='",
                    description=(
                        f"Comparison `{text.strip()}` uses equality operator with None. "
                        "PEP 8 requires identity checks ('is' / 'is not') for singletons."
                    ),
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    confidence=0.90,
                    tags=["static-analysis", "pattern", "python", "pep8"],
                )
            )
    return findings


def _check_python_bool_comparison(
    root, language: str, file_path: str
) -> list[Finding]:
    """Flag ``== True`` / ``== False`` — should use truthiness checks."""
    findings: list[Finding] = []
    for node in walk(root):
        if node.type != "comparison_operator":
            continue
        text = get_node_text(node)
        if re.search(r"==\s*(True|False)\b", text) or re.search(
            r"\b(True|False)\s*==", text
        ):
            findings.append(
                Finding(
                    agent_name="ast_pattern_matching",
                    severity="low",
                    category="style",
                    title="Avoid explicit comparison to True/False",
                    description=(
                        f"Comparison `{text.strip()}` explicitly compares to a boolean. "
                        "Use the value directly or negate with 'not' instead."
                    ),
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    confidence=0.90,
                    tags=["static-analysis", "pattern", "python", "pep8"],
                )
            )
    return findings


def _check_python_mutable_defaults(
    root, language: str, file_path: str
) -> list[Finding]:
    """Flag mutable default arguments such as ``def f(x=[])`` or ``def f(x={})``."""
    findings: list[Finding] = []
    for func in find_functions(root, language):
        params_node = func.node.child_by_field_name("parameters")
        if params_node is None:
            # For decorated_definition, dig into inner function
            for child in func.node.children:
                if child.type == "function_definition":
                    params_node = child.child_by_field_name("parameters")
                    break
        if params_node is None:
            continue
        for param in walk(params_node):
            if param.type == "default_parameter":
                value_node = param.child_by_field_name("value")
                if value_node is not None:
                    val_text = get_node_text(value_node)
                    if _is_mutable_default(val_text):
                        findings.append(
                            Finding(
                                agent_name="ast_pattern_matching",
                                severity="medium",
                                category="correctness",
                                title=f"Mutable default argument in '{func.name}'",
                                description=(
                                    f"Parameter has mutable default `{val_text.strip()}`. "
                                    "Mutable defaults are shared across calls and can "
                                    "cause subtle bugs. Use None and assign inside the "
                                    "function body instead."
                                ),
                                file_path=file_path,
                                line_start=param.start_point[0] + 1,
                                line_end=param.end_point[0] + 1,
                                confidence=0.90,
                                tags=[
                                    "static-analysis",
                                    "pattern",
                                    "python",
                                    "mutable-default",
                                ],
                            )
                        )
    return findings


def _check_js_loose_equality(
    root, language: str, file_path: str
) -> list[Finding]:
    """Flag ``==`` / ``!=`` (loose equality) in JS/TS — should use ``===`` / ``!==``."""
    findings: list[Finding] = []
    for node in walk(root):
        if node.type != "binary_expression":
            continue
        operator_node = node.child_by_field_name("operator")
        if operator_node is None:
            # Fall back: look for unnamed child that is the operator token
            for child in node.children:
                if child.type in ("==", "!="):
                    operator_node = child
                    break
        if operator_node is None:
            continue
        op_text = get_node_text(operator_node)
        if op_text in ("==", "!="):
            text = get_node_text(node)
            findings.append(
                Finding(
                    agent_name="ast_pattern_matching",
                    severity="medium",
                    category="correctness",
                    title=f"Use '===' / '!==' instead of '{op_text}'",
                    description=(
                        f"Expression `{text.strip()}` uses loose equality ('{op_text}'). "
                        "Loose equality performs type coercion and can produce unexpected "
                        "results. Use strict equality instead."
                    ),
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    confidence=0.90,
                    tags=["static-analysis", "pattern", "javascript", "equality"],
                )
            )
    return findings


def _check_js_var_declarations(
    root, language: str, file_path: str
) -> list[Finding]:
    """Flag ``var`` declarations — should use ``let`` or ``const``."""
    findings: list[Finding] = []
    for node in walk(root):
        if node.type != "variable_declaration":
            continue
        text = get_node_text(node)
        if text.lstrip().startswith("var "):
            findings.append(
                Finding(
                    agent_name="ast_pattern_matching",
                    severity="low",
                    category="style",
                    title="Replace 'var' with 'let' or 'const'",
                    description=(
                        "'var' has function scope and is hoisted, which can lead to "
                        "confusing behaviour. Use 'let' for variables that are "
                        "reassigned and 'const' for those that are not."
                    ),
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    confidence=0.90,
                    tags=["static-analysis", "pattern", "javascript", "var"],
                )
            )
    return findings


def _check_empty_catch_blocks(
    root, language: str, file_path: str
) -> list[Finding]:
    """Flag empty except/catch blocks (language-agnostic, pattern-based)."""
    findings: list[Finding] = []
    catch_types = {
        "python": "except_clause",
        "javascript": "catch_clause",
        "typescript": "catch_clause",
        "java": "catch_clause",
    }
    target_type = catch_types.get(language.lower())
    if target_type is None:
        return findings

    for node in walk(root):
        if node.type != target_type:
            continue

        # Find the block/body child inside the catch clause
        body = node.child_by_field_name("body")
        if body is None:
            # Python except_clause: the block child is the last named child
            for child in node.named_children:
                if child.type in ("block", "statement_block"):
                    body = child
                    break
        if body is None:
            continue

        # Check whether the body contains only comments or pass statements
        meaningful_children = [
            c
            for c in body.named_children
            if c.type not in ("comment", "pass_statement")
        ]
        if len(meaningful_children) == 0:
            findings.append(
                Finding(
                    agent_name="ast_pattern_matching",
                    severity="medium",
                    category="correctness",
                    title="Empty except/catch block",
                    description=(
                        "This except/catch block silently swallows exceptions. "
                        "At minimum, log the error or add a comment explaining "
                        "why it is intentionally ignored."
                    ),
                    file_path=file_path,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    confidence=0.90,
                    tags=[
                        "static-analysis",
                        "pattern",
                        "anti-pattern",
                        "empty-catch",
                    ],
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AstPatternMatchingAgent(BaseReviewAgent):
    """Configurable rule engine that matches AST anti-patterns.

    Walks the tree-sitter AST once and applies language-specific pattern
    rules.  All checks are deterministic (no ML model required) and have
    high precision because they operate on the parsed syntax tree, not
    raw text.
    """

    # Maps language -> list of checker functions
    _LANGUAGE_RULES: dict[str, list] = {
        "python": [
            _check_python_none_comparison,
            _check_python_bool_comparison,
            _check_python_mutable_defaults,
            _check_empty_catch_blocks,
        ],
        "javascript": [
            _check_js_loose_equality,
            _check_js_var_declarations,
            _check_empty_catch_blocks,
        ],
        "typescript": [
            _check_js_loose_equality,
            _check_js_var_declarations,
            _check_empty_catch_blocks,
        ],
        "java": [
            _check_empty_catch_blocks,
        ],
        "go": [],
    }

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="ast_pattern_matching",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="static_analysis",
            axis_type="aware",
            tags=["static-analysis", "pattern", "anti-pattern"],
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

        lang = context.language.lower()
        rules = self._LANGUAGE_RULES.get(lang, [])
        if not rules:
            return []

        findings: list[Finding] = []
        for rule_fn in rules:
            findings.extend(rule_fn(root, lang, context.file_path))
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. {finding.description} "
            "Fixing AST anti-patterns improves code correctness and readability."
        )

    def suggest_fix(self, finding: Finding) -> Optional["CodeFix"]:
        """Return None — fixes are described in the finding description."""
        return None
