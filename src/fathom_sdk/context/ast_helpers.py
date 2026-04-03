"""Common AST traversal helpers used by multiple agents.

Provides language-aware extraction of functions, loops, try/catch blocks,
string literals, calls, and assignments from tree-sitter ASTs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import tree_sitter

# ---------------------------------------------------------------------------
# Node-type mappings per language
# ---------------------------------------------------------------------------

_FUNCTION_DEF_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "java": {"method_declaration", "constructor_declaration"},
    "go": {"function_declaration", "method_declaration"},
}

_LOOP_TYPES: dict[str, set[str]] = {
    "python": {"for_statement", "while_statement"},
    "javascript": {"for_statement", "for_in_statement", "while_statement", "do_statement"},
    "typescript": {"for_statement", "for_in_statement", "while_statement", "do_statement"},
    "java": {"for_statement", "enhanced_for_statement", "while_statement", "do_statement"},
    "go": {"for_statement"},
}

_TRY_TYPES: dict[str, set[str]] = {
    "python": {"try_statement"},
    "javascript": {"try_statement"},
    "typescript": {"try_statement"},
    "java": {"try_statement"},
}

_CATCH_TYPES: dict[str, set[str]] = {
    "python": {"except_clause"},
    "javascript": {"catch_clause"},
    "typescript": {"catch_clause"},
    "java": {"catch_clause"},
}

_STRING_TYPES: dict[str, set[str]] = {
    "python": {"string", "concatenated_string"},
    "javascript": {"string", "template_string"},
    "typescript": {"string", "template_string"},
    "java": {"string_literal"},
    "go": {"interpreted_string_literal", "raw_string_literal"},
}

_CALL_TYPES: dict[str, set[str]] = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "java": {"method_invocation", "object_creation_expression"},
    "go": {"call_expression"},
}

_ASSIGNMENT_TYPES: dict[str, set[str]] = {
    "python": {"assignment", "augmented_assignment"},
    "javascript": {"assignment_expression", "variable_declarator"},
    "typescript": {"assignment_expression", "variable_declarator"},
    "java": {"assignment_expression", "variable_declarator"},
    "go": {"assignment_statement", "short_var_declaration"},
}

_IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
    "java": {"import_declaration"},
    "go": {"import_declaration", "import_spec"},
}

_CLASS_DEF_TYPES: dict[str, set[str]] = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "java": {"class_declaration", "interface_declaration"},
    "go": {"type_declaration"},
}

_NUMERIC_TYPES: dict[str, set[str]] = {
    "python": {"integer", "float"},
    "javascript": {"number"},
    "typescript": {"number"},
    "java": {"decimal_integer_literal", "decimal_floating_point_literal"},
    "go": {"int_literal", "float_literal"},
}

_RETURN_TYPES: dict[str, set[str]] = {
    "python": {"return_statement"},
    "javascript": {"return_statement"},
    "typescript": {"return_statement"},
    "java": {"return_statement"},
    "go": {"return_statement"},
}

_DEFAULT_FUNC_TYPES: set[str] = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
}
_DEFAULT_LOOP_TYPES: set[str] = {"for_statement", "while_statement"}
_DEFAULT_CALL_TYPES: set[str] = {"call_expression", "call"}

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FunctionInfo:
    """Extracted function/method definition."""

    name: str
    node: tree_sitter.Node
    start_line: int  # 1-based
    end_line: int  # 1-based
    line_count: int
    param_count: int
    is_async: bool


@dataclass(slots=True)
class LoopInfo:
    """Extracted loop with nesting depth."""

    node: tree_sitter.Node
    loop_type: str
    start_line: int
    end_line: int
    nesting_depth: int  # 1 = top-level loop, 2 = nested inside one loop, etc.


@dataclass(slots=True)
class TryCatchInfo:
    """Extracted try/catch block."""

    try_node: tree_sitter.Node
    catch_nodes: list[tree_sitter.Node]
    start_line: int
    end_line: int


@dataclass(slots=True)
class StringLiteralInfo:
    """Extracted string literal."""

    node: tree_sitter.Node
    value: str
    start_line: int
    end_line: int


@dataclass(slots=True)
class CallInfo:
    """Extracted function/method call."""

    node: tree_sitter.Node
    callee_name: str
    full_text: str
    start_line: int
    end_line: int
    enclosing_function: str | None


@dataclass(slots=True)
class AssignmentInfo:
    """Extracted variable assignment."""

    node: tree_sitter.Node
    target: str
    value_text: str
    start_line: int
    end_line: int


@dataclass(slots=True)
class ImportInfo:
    """Extracted import statement."""

    node: tree_sitter.Node
    module: str
    names: list[str]  # imported names (empty for bare imports)
    start_line: int
    end_line: int


@dataclass(slots=True)
class ClassInfo:
    """Extracted class/interface definition."""

    name: str
    node: tree_sitter.Node
    start_line: int
    end_line: int
    method_count: int
    parent_classes: list[str]


@dataclass(slots=True)
class NumericLiteralInfo:
    """Extracted numeric literal."""

    node: tree_sitter.Node
    value_text: str
    start_line: int
    enclosing_function: str | None


@dataclass(slots=True)
class ReturnInfo:
    """Extracted return statement."""

    node: tree_sitter.Node
    value_text: str
    start_line: int
    enclosing_function: str | None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def get_node_text(node: tree_sitter.Node) -> str:
    """Safely decode a tree-sitter node's text."""
    if node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def walk(node: tree_sitter.Node) -> Iterator[tree_sitter.Node]:
    """Depth-first iterator over all nodes in a subtree."""
    yield node
    for child in node.children:
        yield from walk(child)


def _get_types(mapping: dict[str, set[str]], language: str, default: set[str]) -> set[str]:
    return mapping.get(language.lower(), default)


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------


def _func_name(node: tree_sitter.Node, language: str) -> str | None:
    """Extract function name from a definition node."""
    # Handle Python decorated_definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type == "function_definition":
                return _func_name(child, language)
        return None

    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return get_node_text(name_node)

    # Arrow functions: try the parent variable declarator for the name
    if node.type == "arrow_function" and node.parent is not None:
        if node.parent.type == "variable_declarator":
            name_n = node.parent.child_by_field_name("name")
            if name_n is not None:
                return get_node_text(name_n)
    return None


def _count_params(node: tree_sitter.Node, language: str) -> int:
    """Count function parameters."""
    params = node.child_by_field_name("parameters")
    if params is None:
        # Java uses formal_parameters
        for child in node.children:
            if child.type in ("formal_parameters", "parameter_list"):
                params = child
                break
    if params is None:
        return 0
    return sum(1 for c in params.named_children if c.type != "comment")


def find_functions(root: tree_sitter.Node, language: str) -> list[FunctionInfo]:
    """Extract all function/method definitions from the AST."""
    func_types = _get_types(_FUNCTION_DEF_TYPES, language, _DEFAULT_FUNC_TYPES)
    results: list[FunctionInfo] = []

    # For Python, also check for decorated_definition wrapping function_definition
    extra_types = set()
    if language.lower() == "python":
        extra_types.add("decorated_definition")

    for node in walk(root):
        if node.type not in func_types and node.type not in extra_types:
            continue

        # Skip decorated_definition if the inner function_definition will be caught
        if node.type == "decorated_definition":
            # Only process if the inner function_definition would not be caught independently
            inner = None
            for child in node.children:
                if child.type == "function_definition":
                    inner = child
                    break
            if inner is None:
                continue
            # Use the decorated_definition as the node but inner for name/params
            name = _func_name(node, language)
            if not name:
                continue
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            param_count = _count_params(inner, language)
            is_async = get_node_text(inner).lstrip().startswith("async")
            results.append(
                FunctionInfo(
                    name=name,
                    node=node,
                    start_line=start_line,
                    end_line=end_line,
                    line_count=end_line - start_line + 1,
                    param_count=param_count,
                    is_async=is_async,
                )
            )
            continue

        # Skip if this is inside a decorated_definition (already handled above)
        if (
            language.lower() == "python"
            and node.parent is not None
            and node.parent.type == "decorated_definition"
        ):
            continue

        name = _func_name(node, language)
        if not name:
            continue

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        results.append(
            FunctionInfo(
                name=name,
                node=node,
                start_line=start_line,
                end_line=end_line,
                line_count=end_line - start_line + 1,
                param_count=_count_params(node, language),
                is_async=is_async_function(node, language),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Loop extraction
# ---------------------------------------------------------------------------


def find_loops(root: tree_sitter.Node, language: str) -> list[LoopInfo]:
    """Extract all loops with nesting depth information."""
    loop_types = _get_types(_LOOP_TYPES, language, _DEFAULT_LOOP_TYPES)
    results: list[LoopInfo] = []

    for node in walk(root):
        if node.type not in loop_types:
            continue
        depth = _loop_nesting_depth(node, loop_types)
        results.append(
            LoopInfo(
                node=node,
                loop_type=node.type,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                nesting_depth=depth,
            )
        )
    return results


def _loop_nesting_depth(node: tree_sitter.Node, loop_types: set[str]) -> int:
    """Count how many ancestor loops enclose this loop (1 = top-level)."""
    depth = 1
    current = node.parent
    while current is not None:
        if current.type in loop_types:
            depth += 1
        current = current.parent
    return depth


# ---------------------------------------------------------------------------
# Try/catch extraction
# ---------------------------------------------------------------------------


def find_try_catch_blocks(root: tree_sitter.Node, language: str) -> list[TryCatchInfo]:
    """Extract try/catch/except blocks."""
    try_types = _get_types(_TRY_TYPES, language, set())
    catch_types = _get_types(_CATCH_TYPES, language, set())
    if not try_types:
        return []

    results: list[TryCatchInfo] = []
    for node in walk(root):
        if node.type not in try_types:
            continue
        catches = [c for c in node.named_children if c.type in catch_types]
        results.append(
            TryCatchInfo(
                try_node=node,
                catch_nodes=catches,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
    return results


# ---------------------------------------------------------------------------
# String literal extraction
# ---------------------------------------------------------------------------


def find_string_literals(root: tree_sitter.Node, language: str) -> list[StringLiteralInfo]:
    """Extract string literal nodes and their values."""
    string_types = _get_types(_STRING_TYPES, language, set())
    if not string_types:
        return []

    results: list[StringLiteralInfo] = []
    for node in walk(root):
        if node.type not in string_types:
            continue
        results.append(
            StringLiteralInfo(
                node=node,
                value=get_node_text(node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Call extraction
# ---------------------------------------------------------------------------


def _call_target_name(node: tree_sitter.Node) -> str | None:
    """Best-effort callee name extraction from a call node."""
    func = node.child_by_field_name("function")
    if func is not None:
        if func.type == "identifier":
            return get_node_text(func)
        if func.type in ("attribute", "member_expression"):
            attr = func.child_by_field_name("attribute") or func.child_by_field_name("property")
            if attr is not None:
                return get_node_text(attr)
            text = get_node_text(func)
            if text:
                return text.rsplit(".", 1)[-1]
        return get_node_text(func).rsplit(".", 1)[-1] if get_node_text(func) else None

    # Java method_invocation: name field
    name = node.child_by_field_name("name")
    if name is not None:
        return get_node_text(name)

    # Java object_creation_expression: new Type(args) — extract the type name
    if node.type == "object_creation_expression":
        type_node = node.child_by_field_name("type")
        if type_node is not None:
            return get_node_text(type_node)

    return None


def _enclosing_function_name(
    node: tree_sitter.Node, func_types: set[str], language: str
) -> str | None:
    """Walk up the tree to find the enclosing function name."""
    current = node.parent
    while current is not None:
        if current.type in func_types or (
            language.lower() == "python" and current.type == "decorated_definition"
        ):
            return _func_name(current, language)
        current = current.parent
    return None


def find_calls(root: tree_sitter.Node, language: str) -> list[CallInfo]:
    """Extract all function/method calls."""
    call_types = _get_types(_CALL_TYPES, language, _DEFAULT_CALL_TYPES)
    func_types = _get_types(_FUNCTION_DEF_TYPES, language, _DEFAULT_FUNC_TYPES)
    results: list[CallInfo] = []

    for node in walk(root):
        if node.type not in call_types:
            continue
        callee = _call_target_name(node)
        if callee is None:
            continue
        results.append(
            CallInfo(
                node=node,
                callee_name=callee,
                full_text=get_node_text(node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                enclosing_function=_enclosing_function_name(node, func_types, language),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Assignment extraction
# ---------------------------------------------------------------------------


def find_assignments(root: tree_sitter.Node, language: str) -> list[AssignmentInfo]:
    """Extract variable assignments."""
    assign_types = _get_types(_ASSIGNMENT_TYPES, language, set())
    if not assign_types:
        return []

    results: list[AssignmentInfo] = []
    for node in walk(root):
        if node.type not in assign_types:
            continue

        target: str | None = None
        value_text = ""

        if node.type in ("assignment", "augmented_assignment", "assignment_expression"):
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None:
                target = get_node_text(left)
            if right is not None:
                value_text = get_node_text(right)
        elif node.type == "variable_declarator":
            name_n = node.child_by_field_name("name")
            value_n = node.child_by_field_name("value")
            if name_n is not None:
                target = get_node_text(name_n)
            if value_n is not None:
                value_text = get_node_text(value_n)
        elif node.type in ("assignment_statement", "short_var_declaration"):
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None:
                target = get_node_text(left)
            if right is not None:
                value_text = get_node_text(right)

        if target:
            results.append(
                AssignmentInfo(
                    node=node,
                    target=target,
                    value_text=value_text,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            )
    return results


# ---------------------------------------------------------------------------
# Async detection
# ---------------------------------------------------------------------------


def is_async_function(node: tree_sitter.Node, language: str) -> bool:
    """Check whether a function/method node is async."""
    text = get_node_text(node).lstrip()
    if text.startswith("async "):
        return True
    # Python: async is a child keyword
    if language.lower() == "python" and node.parent is not None:
        if node.parent.type == "decorated_definition":
            for child in node.parent.children:
                if get_node_text(child).strip() == "async":
                    return True
    return False


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def find_imports(root: tree_sitter.Node, language: str) -> list[ImportInfo]:
    """Extract all import statements from the AST."""
    import_types = _get_types(_IMPORT_TYPES, language, set())
    if not import_types:
        return []

    lang = language.lower()
    results: list[ImportInfo] = []

    for node in walk(root):
        if node.type not in import_types:
            continue

        text = get_node_text(node)
        module = ""
        names: list[str] = []

        if lang == "python":
            if node.type == "import_from_statement":
                mod_node = node.child_by_field_name("module_name")
                if mod_node is not None:
                    module = get_node_text(mod_node)
                else:
                    # fallback: parse from text
                    parts = text.split("from")
                    if len(parts) > 1:
                        module = parts[1].split("import")[0].strip()
                for child in node.named_children:
                    if child.type in ("dotted_name", "aliased_import"):
                        names.append(get_node_text(child).split(" as ")[0].strip())
            else:
                # import X / import X.Y
                for child in node.named_children:
                    if child.type in ("dotted_name", "aliased_import"):
                        module = get_node_text(child).split(" as ")[0].strip()
        elif lang in ("javascript", "typescript"):
            # import { x } from 'module' / import x from 'module'
            source_node = node.child_by_field_name("source")
            if source_node is not None:
                module = get_node_text(source_node).strip("\"'`")
            for child in node.named_children:
                if child.type == "import_specifier":
                    names.append(get_node_text(child).split(" as ")[0].strip())
                elif child.type == "identifier":
                    names.append(get_node_text(child))
        elif lang == "java":
            # import package.name;
            module = text.replace("import", "").replace("static", "").strip().rstrip(";").strip()
        elif lang == "go":
            if node.type == "import_spec":
                path_node = node.child_by_field_name("path")
                if path_node is not None:
                    module = get_node_text(path_node).strip('"')
            else:
                # import_declaration may contain multiple specs
                for child in node.named_children:
                    if child.type == "import_spec_list":
                        for spec in child.named_children:
                            if spec.type == "import_spec":
                                path_node = spec.child_by_field_name("path")
                                if path_node is not None:
                                    module = get_node_text(path_node).strip('"')

        results.append(
            ImportInfo(
                node=node,
                module=module,
                names=names,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def find_class_definitions(root: tree_sitter.Node, language: str) -> list[ClassInfo]:
    """Extract class/interface definitions from the AST."""
    class_types = _get_types(_CLASS_DEF_TYPES, language, set())
    if not class_types:
        return []

    lang = language.lower()
    func_types = _get_types(_FUNCTION_DEF_TYPES, language, _DEFAULT_FUNC_TYPES)
    results: list[ClassInfo] = []

    for node in walk(root):
        if node.type not in class_types:
            continue

        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue
        name = get_node_text(name_node)

        # Count methods
        method_count = 0
        for child in walk(node):
            if child != node and child.type in func_types:
                method_count += 1

        # Extract parent classes
        parents: list[str] = []
        if lang == "python":
            args_node = node.child_by_field_name("superclasses")
            if args_node is not None:
                for arg in args_node.named_children:
                    parents.append(get_node_text(arg))
        elif lang == "java":
            sc = node.child_by_field_name("superclass")
            if sc is not None:
                parents.append(get_node_text(sc))
            interfaces = node.child_by_field_name("interfaces")
            if interfaces is not None:
                for child in interfaces.named_children:
                    parents.append(get_node_text(child))

        results.append(
            ClassInfo(
                name=name,
                node=node,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                method_count=method_count,
                parent_classes=parents,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Numeric literal extraction
# ---------------------------------------------------------------------------


def find_numeric_literals(root: tree_sitter.Node, language: str) -> list[NumericLiteralInfo]:
    """Extract numeric literal nodes from the AST."""
    numeric_types = _get_types(_NUMERIC_TYPES, language, set())
    if not numeric_types:
        return []

    func_types = _get_types(_FUNCTION_DEF_TYPES, language, _DEFAULT_FUNC_TYPES)
    results: list[NumericLiteralInfo] = []

    for node in walk(root):
        if node.type not in numeric_types:
            continue
        results.append(
            NumericLiteralInfo(
                node=node,
                value_text=get_node_text(node),
                start_line=node.start_point[0] + 1,
                enclosing_function=_enclosing_function_name(node, func_types, language),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Return statement extraction
# ---------------------------------------------------------------------------


def find_return_statements(root: tree_sitter.Node, language: str) -> list[ReturnInfo]:
    """Extract return statements from the AST."""
    return_types = _get_types(_RETURN_TYPES, language, set())
    if not return_types:
        return []

    func_types = _get_types(_FUNCTION_DEF_TYPES, language, _DEFAULT_FUNC_TYPES)
    results: list[ReturnInfo] = []

    for node in walk(root):
        if node.type not in return_types:
            continue
        # Get the return value expression
        value_text = ""
        for child in node.named_children:
            value_text = get_node_text(child)
            break

        results.append(
            ReturnInfo(
                node=node,
                value_text=value_text,
                start_line=node.start_point[0] + 1,
                enclosing_function=_enclosing_function_name(node, func_types, language),
            )
        )
    return results
