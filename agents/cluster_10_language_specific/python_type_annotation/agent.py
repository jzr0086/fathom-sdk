"""Python Type Annotation Agent — detects missing or incorrect type annotations."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text, walk
from fathom_sdk.context.ast_parser import parse

# Bare container types that should use parameterised generics
_BARE_CONTAINERS = {"dict", "list", "tuple", "set", "frozenset"}

# Regex matching a standalone bare container type (not subscripted)
_BARE_CONTAINER_RE = re.compile(
    r"\b(dict|list|tuple|set|frozenset)\b(?!\s*\[)"
)

# Count uses of ``Any`` in annotation context
_ANY_RE = re.compile(r"\bAny\b")

# Threshold for "overuse of Any"
_ANY_THRESHOLD = 3


class PythonTypeAnnotationAgent(BaseReviewAgent):
    """Flags missing, incomplete, or imprecise Python type annotations."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="python_type_annotation",
            version="0.1.0",
            languages=["python"],
            domains=["web_development", "api_integration"],
            methodology="language_specific",
            axis_type="critical",
            tags=["python", "typing", "annotations"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, context: CodeContext) -> list[Finding]:  # noqa: C901
        if context.language.lower() != "python":
            return []
        if not context.source_code.strip():
            return []

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []

        for func in find_functions(root, context.language):
            func_node = func.node
            # If the top-level node is a decorated_definition, drill into the
            # inner function_definition for annotation inspection.
            inner = func_node
            if func_node.type == "decorated_definition":
                for child in func_node.children:
                    if child.type == "function_definition":
                        inner = child
                        break

            # --- 1. Missing return type annotation ---
            if not self._has_return_type(inner):
                findings.append(
                    Finding(
                        agent_name="python_type_annotation",
                        severity="low",
                        category="quality",
                        title=f"Missing return type annotation on '{func.name}'",
                        description=(
                            f"Function '{func.name}' does not declare a return "
                            f"type. Add a '-> <type>' annotation to improve "
                            f"readability and enable static analysis."
                        ),
                        file_path=context.file_path,
                        line_start=func.start_line,
                        line_end=func.end_line,
                        confidence=0.80,
                        tags=["python", "typing", "annotations"],
                    )
                )

            # --- 2 & 4. Parameter checks ---
            params_node = inner.child_by_field_name("parameters")
            if params_node is not None:
                for param in params_node.named_children:
                    if param.type == "comment":
                        continue
                    param_name = self._param_name(param)
                    if param_name in ("self", "cls"):
                        continue
                    if not param_name:
                        continue

                    type_node = param.child_by_field_name("type")
                    has_annotation = type_node is not None

                    if not has_annotation:
                        # Check for Optional misuse: default is None but no
                        # type annotation at all.
                        default_node = param.child_by_field_name("value")
                        if default_node is not None and get_node_text(default_node).strip() == "None":
                            findings.append(
                                Finding(
                                    agent_name="python_type_annotation",
                                    severity="medium",
                                    category="quality",
                                    title=(
                                        f"Parameter '{param_name}' defaults to None "
                                        f"without Optional annotation"
                                    ),
                                    description=(
                                        f"Parameter '{param_name}' in '{func.name}' has a "
                                        f"default of None but no type annotation. Use "
                                        f"'Optional[<type>]' or '<type> | None' to "
                                        f"express nullability explicitly."
                                    ),
                                    file_path=context.file_path,
                                    line_start=param.start_point[0] + 1,
                                    line_end=param.end_point[0] + 1,
                                    confidence=0.80,
                                    tags=["python", "typing", "annotations"],
                                )
                            )
                        else:
                            # Plain missing annotation
                            findings.append(
                                Finding(
                                    agent_name="python_type_annotation",
                                    severity="low",
                                    category="quality",
                                    title=(
                                        f"Missing type annotation for parameter "
                                        f"'{param_name}'"
                                    ),
                                    description=(
                                        f"Parameter '{param_name}' in '{func.name}' has no "
                                        f"type annotation. Adding a type hint improves "
                                        f"readability and enables static type checking."
                                    ),
                                    file_path=context.file_path,
                                    line_start=param.start_point[0] + 1,
                                    line_end=param.end_point[0] + 1,
                                    confidence=0.80,
                                    tags=["python", "typing", "annotations"],
                                )
                            )
                    else:
                        # Has annotation — check for Optional misuse
                        ann_text = get_node_text(type_node).strip()
                        default_node = param.child_by_field_name("value")
                        if default_node is not None and get_node_text(default_node).strip() == "None":
                            if not self._annotation_allows_none(ann_text):
                                findings.append(
                                    Finding(
                                        agent_name="python_type_annotation",
                                        severity="medium",
                                        category="quality",
                                        title=(
                                            f"Parameter '{param_name}' defaults to None "
                                            f"without Optional annotation"
                                        ),
                                        description=(
                                            f"Parameter '{param_name}' in '{func.name}' "
                                            f"has a default of None but its annotation "
                                            f"'{ann_text}' does not include Optional or "
                                            f"'| None'. Use 'Optional[{ann_text}]' or "
                                            f"'{ann_text} | None'."
                                        ),
                                        file_path=context.file_path,
                                        line_start=param.start_point[0] + 1,
                                        line_end=param.end_point[0] + 1,
                                        confidence=0.80,
                                        tags=["python", "typing", "annotations"],
                                    )
                                )

                        # --- 3. Bare container annotation ---
                        if ann_text in _BARE_CONTAINERS:
                            findings.append(
                                Finding(
                                    agent_name="python_type_annotation",
                                    severity="low",
                                    category="quality",
                                    title=(
                                        f"Bare '{ann_text}' annotation on parameter "
                                        f"'{param_name}'"
                                    ),
                                    description=(
                                        f"Parameter '{param_name}' in '{func.name}' uses "
                                        f"bare '{ann_text}' without type parameters. "
                                        f"Use e.g. '{ann_text}[str, int]' to specify "
                                        f"contained types."
                                    ),
                                    file_path=context.file_path,
                                    line_start=param.start_point[0] + 1,
                                    line_end=param.end_point[0] + 1,
                                    confidence=0.80,
                                    tags=["python", "typing", "annotations"],
                                )
                            )

            # --- 3. Bare container in return type ---
            return_type = inner.child_by_field_name("return_type")
            if return_type is not None:
                rt_text = get_node_text(return_type).strip()
                if rt_text in _BARE_CONTAINERS:
                    findings.append(
                        Finding(
                            agent_name="python_type_annotation",
                            severity="low",
                            category="quality",
                            title=(
                                f"Bare '{rt_text}' return annotation on '{func.name}'"
                            ),
                            description=(
                                f"Function '{func.name}' uses bare '{rt_text}' as its "
                                f"return type. Specify type parameters, e.g. "
                                f"'{rt_text}[str, int]'."
                            ),
                            file_path=context.file_path,
                            line_start=func.start_line,
                            line_end=func.end_line,
                            confidence=0.80,
                            tags=["python", "typing", "annotations"],
                        )
                    )

        # --- 5. Overuse of Any ---
        any_count = self._count_any_annotations(root)
        if any_count > _ANY_THRESHOLD:
            findings.append(
                Finding(
                    agent_name="python_type_annotation",
                    severity="low",
                    category="quality",
                    title=f"Overuse of 'Any' type ({any_count} occurrences)",
                    description=(
                        f"This file uses 'Any' {any_count} times in type "
                        f"annotations (threshold: {_ANY_THRESHOLD}). Excessive "
                        f"use of Any defeats the purpose of type checking. "
                        f"Replace with specific types where possible."
                    ),
                    file_path=context.file_path,
                    line_start=1,
                    line_end=len(context.source_code.splitlines()),
                    confidence=0.80,
                    tags=["python", "typing", "annotations"],
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Type annotations improve code readability, "
            f"enable static analysis tools like mypy, and make refactoring "
            f"safer. Adding precise annotations catches bugs before runtime."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_return_type(func_node) -> bool:
        """Check whether a function_definition node has a return type annotation."""
        return func_node.child_by_field_name("return_type") is not None

    @staticmethod
    def _param_name(param_node) -> str:
        """Extract the parameter name from a tree-sitter parameter node."""
        # typed_parameter or typed_default_parameter
        name_node = param_node.child_by_field_name("name")
        if name_node is not None:
            return get_node_text(name_node).strip()
        # typed_parameter: first named child is the identifier
        if param_node.type == "typed_parameter" and param_node.named_children:
            first = param_node.named_children[0]
            if first.type == "identifier":
                return get_node_text(first).strip()
        # plain identifier (no annotation)
        if param_node.type == "identifier":
            return get_node_text(param_node).strip()
        # default_parameter: name = value
        if param_node.type == "default_parameter":
            name_node = param_node.child_by_field_name("name")
            if name_node is not None:
                return get_node_text(name_node).strip()
        # *args, **kwargs
        if param_node.type in (
            "list_splat_pattern",
            "dictionary_splat_pattern",
        ):
            for child in param_node.children:
                if child.type == "identifier":
                    return get_node_text(child).strip()
        return ""

    @staticmethod
    def _annotation_allows_none(ann_text: str) -> bool:
        """Return True if the annotation text indicates the value can be None."""
        if "Optional" in ann_text:
            return True
        if "None" in ann_text:
            return True
        if "Any" in ann_text:
            return True
        return False

    @staticmethod
    def _count_any_annotations(root) -> int:
        """Count how many times ``Any`` appears in type annotation contexts."""
        count = 0
        for node in walk(root):
            # type annotations appear as children named "type" or "return_type"
            if node.type == "type":
                text = get_node_text(node)
                count += len(_ANY_RE.findall(text))
        return count
