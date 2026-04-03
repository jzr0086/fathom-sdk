"""Circular Dependency Detector — finds circular import chains in source files."""

from __future__ import annotations

import networkx as nx

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_imports
from fathom_sdk.context.ast_parser import parse


def _normalize_module(name: str) -> str:
    """Normalize a module/path string to a canonical dotted form.

    Strips file extensions, replaces path separators with dots, and
    removes leading dots so that ``./utils`` and ``utils.py`` both
    resolve to ``utils``.
    """
    result = name.strip()
    for ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go"):
        if result.endswith(ext):
            result = result[: -len(ext)]
            break
    result = result.replace("/", ".").replace("\\", ".")
    result = result.lstrip(".")
    return result


class CircularDependencyAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="circular_dependency",
            version="0.1.0",
            languages=["*"],
            domains=["web_development", "enterprise_engineering"],
            methodology="architecture",
            axis_type="agnostic",
            tags=["architecture", "dependency", "circular"],
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

        imports = find_imports(root, context.language)
        if not imports:
            return []

        # Derive the current module name from the file path.
        current_module = _normalize_module(context.file_path)

        # Build a directed graph of module -> imported module
        graph = nx.DiGraph()
        graph.add_node(current_module)

        # Map each imported module to the ImportInfo for line references
        import_line_map: dict[str, int] = {}
        for imp in imports:
            raw_target = imp.module.strip()
            if not raw_target:
                continue
            target = _normalize_module(raw_target)
            if not target:
                continue
            graph.add_edge(current_module, target)
            # Keep the first occurrence line for each imported module
            if target not in import_line_map:
                import_line_map[target] = imp.start_line

        findings: list[Finding] = []

        # Detect cycles using NetworkX
        for cycle in nx.simple_cycles(graph):
            cycle_desc = " -> ".join(cycle) + " -> " + cycle[0]
            # Use the line of the first import in the cycle that we control
            line = 1
            for mod in cycle:
                if mod in import_line_map:
                    line = import_line_map[mod]
                    break

            findings.append(
                Finding(
                    agent_name="circular_dependency",
                    severity="high",
                    category="architecture",
                    title=f"Circular dependency detected: {cycle_desc}",
                    description=(
                        f"A circular import chain was detected: {cycle_desc}. "
                        f"Circular dependencies make code harder to maintain, "
                        f"test, and reason about. Consider breaking the cycle "
                        f"by introducing an interface, moving shared code to a "
                        f"common module, or using lazy imports."
                    ),
                    file_path=context.file_path,
                    line_start=line,
                    line_end=line,
                    confidence=0.85,
                    tags=["architecture", "dependency", "circular"],
                )
            )

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Circular dependencies create tight coupling "
            f"between modules, making them difficult to test in isolation, "
            f"prone to import-time errors, and resistant to refactoring. "
            f"Break the cycle by extracting shared abstractions or using "
            f"dependency inversion."
        )
