"""JS/TS Async/Await Correctness — detects common async/await mistakes."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import (
    find_calls,
    find_functions,
    find_loops,
    get_node_text,
    walk,
)
from fathom_sdk.context.ast_parser import parse

# Functions that are inherently async (return a Promise)
_ASYNC_API_CALLS: set[str] = {
    "fetch",
    "axios",
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "request",
    "query",
    "findOne",
    "findMany",
    "save",
    "create",
    "update",
    "remove",
    "sendMessage",
    "readFile",
    "writeFile",
    "connect",
    "listen",
    "json",
}


def _is_inside_await(node) -> bool:
    """Check whether the given node is wrapped in an await_expression."""
    current = node.parent
    while current is not None:
        if current.type == "await_expression":
            return True
        # Stop walking past function boundaries
        if current.type in (
            "function_declaration",
            "method_definition",
            "arrow_function",
        ):
            break
        current = current.parent
    return False


def _is_inside_loop(node, loop_types: set[str]) -> tuple[bool, int]:
    """Check whether the node is inside a loop; return (True, line) or (False, 0)."""
    current = node.parent
    while current is not None:
        if current.type in loop_types:
            return True, current.start_point[0] + 1
        # Stop walking past function boundaries
        if current.type in (
            "function_declaration",
            "method_definition",
            "arrow_function",
        ):
            break
        current = current.parent
    return False, 0


def _has_then_chain(node) -> bool:
    """Check if a call_expression is followed by .then() in the parent chain."""
    current = node.parent
    while current is not None:
        if current.type == "member_expression":
            prop = current.child_by_field_name("property")
            if prop is not None and get_node_text(prop) == "then":
                return True
        # If the parent is a call_expression whose function is a member_expression
        # accessing .then on our result, that counts
        if current.type == "call_expression" and current != node:
            func = current.child_by_field_name("function")
            if func is not None and func.type == "member_expression":
                prop = func.child_by_field_name("property")
                if prop is not None and get_node_text(prop) == "then":
                    return True
        # Stop at statement level
        if current.type in (
            "expression_statement",
            "variable_declaration",
            "lexical_declaration",
            "return_statement",
        ):
            break
        current = current.parent
    return False


def _has_catch_chain(node) -> bool:
    """Check whether a .then() call is followed by .catch() in the same chain."""
    # Walk up from the .then() call_expression to see if there is a .catch()
    # higher in the chain, OR walk the full text of the containing statement
    # for a simpler regex-based check.
    current = node.parent
    while current is not None:
        if current.type == "call_expression" and current != node:
            func = current.child_by_field_name("function")
            if func is not None and func.type == "member_expression":
                prop = func.child_by_field_name("property")
                if prop is not None and get_node_text(prop) == "catch":
                    return True
        if current.type in (
            "expression_statement",
            "variable_declaration",
            "lexical_declaration",
            "return_statement",
        ):
            break
        current = current.parent
    return False


class JsAsyncAwaitAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="js_async_await",
            version="0.1.0",
            languages=["javascript", "typescript"],
            domains=["web_development", "api_integration"],
            methodology="language_specific",
            axis_type="critical",
            tags=["javascript", "typescript", "async", "await"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        if lang not in ("javascript", "typescript"):
            return []

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []

        # Collect async function names for scoping
        async_funcs = {
            f.name for f in find_functions(root, context.language) if f.is_async
        }

        # Loop types for this language
        loop_types = {
            "for_statement",
            "for_in_statement",
            "while_statement",
            "do_statement",
        }

        # Track which lines have already been reported to avoid duplicates
        reported_lines: set[int] = set()

        # ---------------------------------------------------------------
        # Detection 1 & 3: Missing await / floating promise
        # Walk all call_expression nodes inside async functions
        # ---------------------------------------------------------------
        for call_info in find_calls(root, context.language):
            if call_info.enclosing_function not in async_funcs:
                continue

            call_node = call_info.node
            callee = call_info.callee_name

            # Only check known async API calls
            if callee not in _ASYNC_API_CALLS:
                continue

            # Skip if already awaited
            if _is_inside_await(call_node):
                continue

            # Skip if chained with .then()
            if _has_then_chain(call_node):
                continue

            line = call_info.start_line
            if line in reported_lines:
                continue
            reported_lines.add(line)

            findings.append(
                Finding(
                    agent_name="js_async_await",
                    severity="high",
                    category="bug",
                    title=f"Missing await on async call '{callee}'",
                    description=(
                        f"The async call '{callee}' at line {line} inside "
                        f"async function '{call_info.enclosing_function}' is not "
                        f"awaited. This may cause unhandled promise behaviour or "
                        f"logic errors."
                    ),
                    file_path=context.file_path,
                    line_start=line,
                    line_end=call_info.end_line,
                    confidence=0.78,
                    tags=["javascript", "async", "await", "bug"],
                )
            )

        # ---------------------------------------------------------------
        # Detection 2: Await in loop
        # Walk all await_expression nodes and check if inside a loop
        # ---------------------------------------------------------------
        for node in walk(root):
            if node.type != "await_expression":
                continue

            # Check that the await is inside an async function
            enc_func = None
            current = node.parent
            while current is not None:
                if current.type in (
                    "function_declaration",
                    "method_definition",
                    "arrow_function",
                ):
                    # Check if this function is async
                    text = get_node_text(current).lstrip()
                    if text.startswith("async "):
                        name_node = current.child_by_field_name("name")
                        if name_node is not None:
                            enc_func = get_node_text(name_node)
                        elif (
                            current.type == "arrow_function"
                            and current.parent is not None
                            and current.parent.type == "variable_declarator"
                        ):
                            vname = current.parent.child_by_field_name("name")
                            if vname is not None:
                                enc_func = get_node_text(vname)
                    break
                current = current.parent

            if enc_func is None:
                continue

            in_loop, loop_line = _is_inside_loop(node, loop_types)
            if not in_loop:
                continue

            line = node.start_point[0] + 1
            if line in reported_lines:
                continue
            reported_lines.add(line)

            await_text = get_node_text(node).split("\n")[0][:60]
            findings.append(
                Finding(
                    agent_name="js_async_await",
                    severity="medium",
                    category="bug",
                    title="Await inside loop — consider Promise.all",
                    description=(
                        f"'await' at line {line} ('{await_text}') is inside a "
                        f"loop starting at line {loop_line}. Sequential awaits "
                        f"in loops are slow; consider using Promise.all() for "
                        f"parallel execution."
                    ),
                    file_path=context.file_path,
                    line_start=line,
                    line_end=node.end_point[0] + 1,
                    confidence=0.78,
                    tags=["javascript", "async", "performance", "bug"],
                )
            )

        # ---------------------------------------------------------------
        # Detection 4: Unhandled rejection — .then() without .catch()
        # ---------------------------------------------------------------
        for node in walk(root):
            if node.type != "call_expression":
                continue

            func = node.child_by_field_name("function")
            if func is None or func.type != "member_expression":
                continue

            prop = func.child_by_field_name("property")
            if prop is None or get_node_text(prop) != "then":
                continue

            # This is a .then() call — check for .catch() in the chain
            if _has_catch_chain(node):
                continue

            # Also check if the full statement text contains .catch
            # (covers cases where .catch is after the .then in the same expr)
            stmt_node = node
            current = node.parent
            while current is not None:
                if current.type in (
                    "expression_statement",
                    "variable_declaration",
                    "lexical_declaration",
                    "return_statement",
                ):
                    stmt_node = current
                    break
                current = current.parent

            stmt_text = get_node_text(stmt_node)
            if ".catch(" in stmt_text:
                continue

            line = node.start_point[0] + 1
            if line in reported_lines:
                continue
            reported_lines.add(line)

            findings.append(
                Finding(
                    agent_name="js_async_await",
                    severity="medium",
                    category="bug",
                    title="Unhandled rejection: .then() without .catch()",
                    description=(
                        f"Promise chain at line {line} uses .then() but has no "
                        f".catch() handler. Unhandled promise rejections can "
                        f"crash Node.js processes or silently swallow errors."
                    ),
                    file_path=context.file_path,
                    line_start=line,
                    line_end=node.end_point[0] + 1,
                    confidence=0.78,
                    tags=["javascript", "async", "error-handling", "bug"],
                )
            )

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. "
            f"Async/await mistakes are a common source of subtle bugs in "
            f"JavaScript and TypeScript. Missing awaits lead to floating "
            f"promises, sequential awaits in loops cause performance issues, "
            f"and unhandled rejections can crash processes."
        )
