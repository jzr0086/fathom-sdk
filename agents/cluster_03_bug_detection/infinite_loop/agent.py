"""Infinite Loop / Recursion Detector — flags loops and recursive functions
that are likely to run forever.

Detects:
1. ``while True`` / ``while(true)`` / ``for(;;)`` without break/return/raise/throw
2. Recursive functions with no visible base case (no conditional guard)
3. ``while`` loops where the loop variable is never modified in the body
"""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import (
    find_calls,
    find_functions,
    find_loops,
    get_node_text,
    walk,
)
from fathom_sdk.context.ast_parser import parse

# Node types that represent exit-like statements inside a loop body
_BREAK_TYPES: set[str] = {"break_statement"}
_RETURN_TYPES: set[str] = {"return_statement"}
_RAISE_TYPES: set[str] = {"raise_statement", "throw_statement"}

# Node types that represent conditional branching
_CONDITIONAL_TYPES: set[str] = {
    "if_statement",
    "if_expression",           # Python ternary / Go if
    "ternary_expression",      # JS/TS/Java ? :
    "conditional_expression",  # Python conditional expr
    "switch_statement",
}


def _contains_node_type(node, types: set[str]) -> bool:
    """Return True if any descendant of *node* has a type in *types*."""
    for child in walk(node):
        if child.type in types:
            return True
    return False


def _has_exit_path(loop_node) -> bool:
    """Check whether a loop body contains a break, return, or raise/throw."""
    # Get the loop body — different field names by language/grammar
    body = loop_node.child_by_field_name("body")
    if body is None:
        # JS/TS/Java: the body is often a statement_block child
        for child in loop_node.children:
            if child.type in ("block", "statement_block"):
                body = child
                break
    if body is None:
        # Fallback: search the entire loop node (minus the condition)
        body = loop_node

    for child in walk(body):
        if child.type in _BREAK_TYPES | _RETURN_TYPES | _RAISE_TYPES:
            return True
    return False


def _is_infinite_literal_loop(loop_node, language: str) -> bool:
    """Detect ``while True``, ``while(true)``, and ``for(;;)`` patterns."""
    text = get_node_text(loop_node.node).strip()
    lang = language.lower()

    # Python: while True:
    if lang == "python":
        if loop_node.loop_type == "while_statement":
            cond = loop_node.node.child_by_field_name("condition")
            if cond is not None and get_node_text(cond).strip().lower() == "true":
                return True
        return False

    # JS/TS/Java/Go: while(true) or for(;;)
    if loop_node.loop_type == "while_statement":
        cond = loop_node.node.child_by_field_name("condition")
        if cond is not None:
            cond_text = get_node_text(cond).strip().lower()
            if cond_text in ("true", "(true)"):
                return True
            # parenthesized_expression wrapping true
            if cond.type == "parenthesized_expression":
                inner = get_node_text(cond).strip("() ").lower()
                if inner == "true":
                    return True

    if loop_node.loop_type == "for_statement":
        # for(;;) — no condition, no init, no update
        # In tree-sitter, for(;;) may have an empty_statement as condition
        if lang in ("javascript", "typescript", "java"):
            cond = loop_node.node.child_by_field_name("condition")
            if cond is None:
                return True
            cond_text = get_node_text(cond).strip()
            if cond_text == "" or cond_text == ";" or cond.type == "empty_statement":
                return True
        # Go: for {} (bare for statement with no condition)
        if lang == "go":
            cond = loop_node.node.child_by_field_name("condition")
            init = loop_node.node.child_by_field_name("initializer")
            update = loop_node.node.child_by_field_name("update")
            if cond is None and init is None and update is None:
                # Check it has a body (block) but no range clause
                has_range = any(
                    c.type == "range_clause" for c in loop_node.node.children
                )
                if not has_range:
                    return True

    return False


def _detect_infinite_loops(root, context: CodeContext) -> list[Finding]:
    """Find while(true)/for(;;) loops without break/return/raise."""
    findings: list[Finding] = []
    loops = find_loops(root, context.language)

    for loop in loops:
        if not _is_infinite_literal_loop(loop, context.language):
            continue
        if _has_exit_path(loop.node):
            continue
        findings.append(
            Finding(
                agent_name="infinite_loop",
                severity="high",
                category="bug",
                title="Infinite loop without exit condition",
                description=(
                    f"Loop at line {loop.start_line} runs indefinitely "
                    f"(no break, return, or raise/throw found in the body)."
                ),
                file_path=context.file_path,
                line_start=loop.start_line,
                line_end=loop.end_line,
                confidence=0.78,
                tags=["bug", "infinite-loop"],
            )
        )
    return findings


def _detect_unguarded_recursion(root, context: CodeContext) -> list[Finding]:
    """Find recursive functions that lack a conditional base case."""
    findings: list[Finding] = []
    functions = find_functions(root, context.language)
    calls = find_calls(root, context.language)

    # Build a map: function_name -> list of calls to that function within itself
    for func in functions:
        func_name = func.name
        # Find calls within this function that call the function itself
        recursive_calls = [
            c
            for c in calls
            if c.callee_name == func_name and c.enclosing_function == func_name
        ]
        if not recursive_calls:
            continue

        # Check if there is a conditional guard (if/switch) anywhere in the function body
        has_conditional = _contains_node_type(func.node, _CONDITIONAL_TYPES)
        if has_conditional:
            continue

        findings.append(
            Finding(
                agent_name="infinite_loop",
                severity="medium",
                category="bug",
                title=f"Recursive function '{func_name}' has no visible base case",
                description=(
                    f"Function '{func_name}' at line {func.start_line} calls "
                    f"itself but contains no conditional (if/switch) to stop "
                    f"recursion, which will likely cause a stack overflow."
                ),
                file_path=context.file_path,
                line_start=func.start_line,
                line_end=func.end_line,
                confidence=0.78,
                tags=["bug", "infinite-loop", "recursion"],
            )
        )
    return findings


def _detect_unmodified_loop_var(root, context: CodeContext) -> list[Finding]:
    """Find while loops where the condition variable is never modified in the body."""
    findings: list[Finding] = []
    loops = find_loops(root, context.language)

    for loop in loops:
        if loop.loop_type != "while_statement":
            continue

        cond = loop.node.child_by_field_name("condition")
        if cond is None:
            continue

        cond_text = get_node_text(cond).strip().strip("()")
        # Skip literal booleans — already handled by infinite-loop detection
        if cond_text.lower() in ("true", "false"):
            continue

        # Extract simple identifiers used in the condition
        cond_identifiers: set[str] = set()
        for child in walk(cond):
            if child.type == "identifier":
                cond_identifiers.add(get_node_text(child))

        if not cond_identifiers:
            continue

        # Get the loop body
        body = loop.node.child_by_field_name("body")
        if body is None:
            for child in loop.node.children:
                if child.type in ("block", "statement_block"):
                    body = child
                    break
        if body is None:
            continue

        body_text = get_node_text(body)

        # Check if any condition variable is modified in the body
        # Look for assignments, augmented assignments, increment/decrement
        any_modified = False
        for var in cond_identifiers:
            # Simple heuristic: var appears on left side of assignment or
            # in augmented assignment / update expression
            for child in walk(body):
                if child.type in (
                    "assignment",
                    "augmented_assignment",
                    "assignment_expression",
                    "assignment_statement",
                    "short_var_declaration",
                ):
                    left = child.child_by_field_name("left")
                    if left is not None and var in get_node_text(left):
                        any_modified = True
                        break
                # variable_declarator with same name
                if child.type == "variable_declarator":
                    name_n = child.child_by_field_name("name")
                    if name_n is not None and get_node_text(name_n) == var:
                        any_modified = True
                        break
                # update_expression (i++, i--)
                if child.type == "update_expression":
                    if var in get_node_text(child):
                        any_modified = True
                        break
            if any_modified:
                break

        # Also check for break/return/raise which would prevent infinite looping
        if not any_modified and not _has_exit_path(loop.node):
            findings.append(
                Finding(
                    agent_name="infinite_loop",
                    severity="medium",
                    category="bug",
                    title="Loop variable never modified in body",
                    description=(
                        f"while loop at line {loop.start_line} tests "
                        f"{cond_identifiers} but none of these variables are "
                        f"modified inside the loop body, which may cause "
                        f"infinite looping."
                    ),
                    file_path=context.file_path,
                    line_start=loop.start_line,
                    line_end=loop.end_line,
                    confidence=0.78,
                    tags=["bug", "infinite-loop", "loop-variable"],
                )
            )
    return findings


class InfiniteLoopAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="infinite_loop",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "systems_programming"],
            methodology="bug_detection",
            axis_type="aware",
            tags=["bug", "infinite-loop", "recursion"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        if lang not in ("python", "javascript", "typescript", "java", "go"):
            return []

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []
        findings.extend(_detect_infinite_loops(root, context))
        findings.extend(_detect_unguarded_recursion(root, context))
        findings.extend(_detect_unmodified_loop_var(root, context))
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Infinite loops and unbounded recursion cause "
            f"programs to hang or crash with stack overflows. Ensure every "
            f"loop has a reachable exit condition and every recursive function "
            f"has a base case."
        )
