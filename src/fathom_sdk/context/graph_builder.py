"""Graph construction utilities — call graphs and control-flow graphs.

Working skeleton implementations that walk a tree-sitter AST and produce
NetworkX directed graphs suitable for downstream agents.
"""

from __future__ import annotations

from typing import Iterator

import networkx as nx
import tree_sitter

# ---------------------------------------------------------------------------
# Node-type mappings per language
# ---------------------------------------------------------------------------

_FUNCTION_DEF_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
}

_CALL_TYPES: dict[str, set[str]] = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "java": {"method_invocation"},
    "go": {"call_expression"},
}

_BRANCH_TYPES: dict[str, set[str]] = {
    "python": {
        "if_statement", "elif_clause", "else_clause",
        "for_statement", "while_statement",
        "try_statement", "except_clause",
        "with_statement",
    },
    "javascript": {
        "if_statement", "else_clause",
        "for_statement", "for_in_statement", "while_statement", "do_statement",
        "switch_statement", "try_statement", "catch_clause",
    },
    "typescript": {
        "if_statement", "else_clause",
        "for_statement", "for_in_statement", "while_statement", "do_statement",
        "switch_statement", "try_statement", "catch_clause",
    },
    "java": {
        "if_statement", "else_clause",
        "for_statement", "enhanced_for_statement", "while_statement", "do_statement",
        "switch_expression", "try_statement", "catch_clause",
    },
    "go": {
        "if_statement", "for_statement",
        "switch_statement", "select_statement",
    },
}

_DEFAULT_FUNCTION_DEFS: set[str] = {
    "function_definition", "function_declaration", "method_definition",
}
_DEFAULT_CALL_TYPES: set[str] = {"call_expression", "call"}
_DEFAULT_BRANCH_TYPES: set[str] = {
    "if_statement", "else_clause", "for_statement",
    "while_statement", "try_statement",
}


def _get_types(
    mapping: dict[str, set[str]], language: str, default: set[str],
) -> set[str]:
    return mapping.get(language.lower(), default)


def _walk(node: tree_sitter.Node) -> Iterator[tree_sitter.Node]:
    """Depth-first iterator over all nodes in the subtree."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _node_name(node: tree_sitter.Node, language: str) -> str | None:
    """Try to extract the name of a function/method definition node."""
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return name_node.text.decode("utf-8") if name_node.text else None

    if node.type == "decorated_definition":
        for child in node.children:
            if child.type == "function_definition":
                return _node_name(child, language)

    return None


def _call_target_name(node: tree_sitter.Node) -> str | None:
    """Best-effort extraction of the callee name from a call node."""
    func = node.child_by_field_name("function")
    if func is not None:
        if func.type == "identifier":
            return func.text.decode("utf-8") if func.text else None
        if func.type == "attribute":
            attr = func.child_by_field_name("attribute")
            if attr is not None:
                return attr.text.decode("utf-8") if attr.text else None
        text = func.text.decode("utf-8") if func.text else None
        if text:
            return text.rsplit(".", 1)[-1]

    name = node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8") if name.text else None

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_call_graph(root_node: tree_sitter.Node, language: str) -> nx.DiGraph:
    """Build a directed call graph from the AST root node.

    Nodes represent function definitions; edges represent caller->callee
    relationships.  A synthetic ``<module>`` node represents top-level code.
    """
    graph = nx.DiGraph()
    func_def_types = _get_types(_FUNCTION_DEF_TYPES, language, _DEFAULT_FUNCTION_DEFS)
    call_types = _get_types(_CALL_TYPES, language, _DEFAULT_CALL_TYPES)

    # Collect function definitions.
    func_nodes: list[tuple[str, tree_sitter.Node]] = []
    for node in _walk(root_node):
        if node.type in func_def_types:
            name = _node_name(node, language)
            if name:
                func_nodes.append((name, node))
                graph.add_node(
                    name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    type="function",
                )

    graph.add_node(
        "<module>",
        start_line=1,
        end_line=root_node.end_point[0] + 1,
        type="module",
    )

    def _enclosing_function(call_node: tree_sitter.Node) -> str:
        current = call_node.parent
        while current is not None:
            if current.type in func_def_types:
                name = _node_name(current, language)
                if name:
                    return name
            current = current.parent
        return "<module>"

    for node in _walk(root_node):
        if node.type in call_types:
            callee = _call_target_name(node)
            if callee is None:
                continue
            caller = _enclosing_function(node)
            if callee not in graph:
                graph.add_node(callee, type="external")
            graph.add_edge(caller, callee, line=node.start_point[0] + 1)

    return graph


def build_control_flow_graph(
    root_node: tree_sitter.Node, language: str,
) -> nx.DiGraph:
    """Build a basic control-flow graph from the AST root node.

    Nodes are statements; edges represent sequential and branching control flow.
    """
    graph = nx.DiGraph()
    branch_types = _get_types(_BRANCH_TYPES, language, _DEFAULT_BRANCH_TYPES)

    _counter = 0

    def _next_id() -> int:
        nonlocal _counter
        _counter += 1
        return _counter

    def _add_node(ts_node: tree_sitter.Node) -> int:
        nid = _next_id()
        text_preview = (ts_node.text.decode("utf-8", errors="replace") if ts_node.text else "")[:60]
        graph.add_node(
            nid,
            label=f"L{ts_node.start_point[0] + 1}: {ts_node.type} | {text_preview}",
            node_type=ts_node.type,
            start_line=ts_node.start_point[0] + 1,
            end_line=ts_node.end_point[0] + 1,
        )
        return nid

    entry_id = _next_id()
    graph.add_node(entry_id, label="ENTRY", node_type="entry", start_line=0, end_line=0)
    exit_id = _next_id()
    graph.add_node(exit_id, label="EXIT", node_type="exit", start_line=0, end_line=0)

    def _process_block(children: list[tree_sitter.Node], predecessor: int) -> int:
        current = predecessor
        for child in children:
            if child.is_named:
                nid = _process_node(child, current)
                current = nid
        return current

    def _process_node(ts_node: tree_sitter.Node, predecessor: int) -> int:
        nid = _add_node(ts_node)
        graph.add_edge(predecessor, nid)

        if ts_node.type in branch_types:
            merge_id = _next_id()
            graph.add_node(
                merge_id,
                label=f"merge({ts_node.type})",
                node_type="merge",
                start_line=ts_node.end_point[0] + 1,
                end_line=ts_node.end_point[0] + 1,
            )

            named_children = [c for c in ts_node.children if c.is_named]
            if named_children:
                for child in named_children:
                    branch_exit = _process_block([child], nid)
                    graph.add_edge(branch_exit, merge_id)
            else:
                graph.add_edge(nid, merge_id)

            return merge_id

        return nid

    top_level = [c for c in root_node.children if c.is_named]
    last = _process_block(top_level, entry_id)
    graph.add_edge(last, exit_id)

    return graph
