"""Layer Violation Detector — flags imports that go UP the architectural layer stack.

In layered architectures (presentation -> service -> repository), dependencies
should only flow downward.  A repository module importing from a controller
is a layer violation that couples low-level data access to HTTP-specific
concerns, making the codebase brittle and hard to refactor.

This agent infers layers from directory names in the file path and import
module paths, then flags any import where the source layer is below the
target layer in the stack.
"""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_imports
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Layer definitions — ordered from top (presentation) to bottom (data).
# Each layer is identified by directory names commonly seen in projects.
# ---------------------------------------------------------------------------

_LAYER_NAMES: list[tuple[int, frozenset[str]]] = [
    # Layer 0 — Presentation / HTTP boundary
    (0, frozenset({"presentation", "views", "controllers", "handlers", "routes"})),
    # Layer 1 — Business / Service logic
    (1, frozenset({"service", "services", "business", "domain", "usecase"})),
    # Layer 2 — Data access / Persistence
    (2, frozenset({"repository", "repositories", "dal", "data", "persistence", "models", "db"})),
]

# Flat lookup: directory name -> layer rank (higher rank = lower layer)
_DIR_TO_LAYER: dict[str, int] = {}
for _rank, _names in _LAYER_NAMES:
    for _name in _names:
        _DIR_TO_LAYER[_name] = _rank


def _layer_rank(path: str) -> int | None:
    """Return the layer rank for a file path, or None if no layer is detected.

    We split the path on ``/`` and look for the **first** directory component
    that matches a known layer name.
    """
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        lower = part.lower()
        if lower in _DIR_TO_LAYER:
            return _DIR_TO_LAYER[lower]
    return None


def _layer_label(rank: int) -> str:
    """Human-friendly label for a layer rank."""
    labels = {0: "presentation", 1: "service", 2: "repository"}
    return labels.get(rank, f"layer-{rank}")


def _layer_rank_from_module(module: str) -> int | None:
    """Try to infer a layer rank from an import module path."""
    parts = module.replace("\\", "/").replace(".", "/").split("/")
    for part in parts:
        lower = part.lower()
        if lower in _DIR_TO_LAYER:
            return _DIR_TO_LAYER[lower]
    return None


class LayerViolationAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="layer_violation",
            version="0.1.0",
            languages=["*"],
            domains=["web_development", "enterprise_engineering", "api_integration"],
            methodology="architecture",
            axis_type="agnostic",
            tags=["architecture", "layers", "dependency"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []

        # Determine what layer this file lives in.
        file_rank = _layer_rank(context.file_path)
        if file_rank is None:
            return []

        # Parse AST if not already available.
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        imports = find_imports(root, context.language)
        findings: list[Finding] = []

        for imp in imports:
            target_rank = _layer_rank_from_module(imp.module)
            if target_rank is None:
                continue
            # A violation occurs when the imported module is in a HIGHER
            # layer (lower rank number) than the current file.
            if target_rank < file_rank:
                findings.append(
                    Finding(
                        agent_name="layer_violation",
                        severity="medium",
                        category="architecture",
                        title=(
                            f"Layer violation: {_layer_label(file_rank)} "
                            f"imports from {_layer_label(target_rank)}"
                        ),
                        description=(
                            f"File '{context.file_path}' is in the "
                            f"{_layer_label(file_rank)} layer but imports "
                            f"'{imp.module}' which belongs to the "
                            f"{_layer_label(target_rank)} layer. "
                            f"Dependencies should flow downward "
                            f"(presentation -> service -> repository), "
                            f"not upward."
                        ),
                        file_path=context.file_path,
                        line_start=imp.start_line,
                        line_end=imp.end_line,
                        confidence=0.75,
                        tags=["architecture", "layers", "dependency"],
                    )
                )

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. In a layered architecture, dependencies must "
            f"only point downward. Upward imports create tight coupling between "
            f"layers and make the codebase harder to test and refactor. "
            f"Extract a shared interface or move the dependency to a lower layer."
        )
