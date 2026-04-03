"""Co-Change Miner — detects implicit coupling between distant modules.

Uses git co-change history (when available) and source-level import analysis
to identify files that are tightly coupled to many disparate parts of the
codebase.  A file that imports from 3+ distinct top-level packages/directories
is a signal of high coupling and a maintenance risk: changes in any of those
distant modules may silently require updates here.
"""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_imports
from fathom_sdk.context.ast_parser import parse
from fathom_sdk.context.git_helpers import find_co_changed_files

# Threshold: a file importing from this many distinct top-level
# packages/directories is flagged as potentially over-coupled.
_TOP_LEVEL_PACKAGE_THRESHOLD = 3


def _top_level_package(module: str) -> str | None:
    """Extract the top-level package/directory from an import module path.

    For dotted Python modules like ``foo.bar.baz`` returns ``foo``.
    For relative JS/TS paths like ``../utils/helpers`` returns ``utils``.
    For Java packages like ``com.example.service`` returns ``com``.
    For Go imports like ``github.com/user/repo`` returns ``github``.

    Returns None for empty or unresolvable modules.
    """
    if not module or not module.strip():
        return None

    cleaned = module.strip().strip("'\"")
    if not cleaned:
        return None

    # Normalise path separators to dots
    normalized = cleaned.replace("/", ".").replace("\\", ".")

    # Strip leading dots (relative imports like . or ..)
    normalized = normalized.lstrip(".")
    if not normalized:
        return None

    parts = normalized.split(".")
    if not parts or not parts[0]:
        return None

    return parts[0]


class CoChangeMinerAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="co_change_miner",
            version="0.1.0",
            languages=["*"],
            domains=["enterprise_engineering", "web_development"],
            methodology="git_intelligence",
            axis_type="agnostic",
            tags=["git", "coupling", "co-change"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        findings: list[Finding] = []

        # --- Git co-change analysis ---
        co_changed = find_co_changed_files(
            context.commit_history, context.file_path
        )
        current_dir = _directory_of(context.file_path)
        for partner_file, count in co_changed:
            partner_dir = _directory_of(partner_file)
            if partner_dir != current_dir:
                findings.append(
                    Finding(
                        agent_name="co_change_miner",
                        severity="low",
                        category="maintenance",
                        title=(
                            f"Co-change coupling: '{context.file_path}' "
                            f"frequently changes with '{partner_file}'"
                        ),
                        description=(
                            f"'{context.file_path}' and '{partner_file}' have "
                            f"been modified together in {count} commits but "
                            f"reside in different directories "
                            f"('{current_dir}' vs '{partner_dir}'). "
                            f"This implicit coupling may indicate a hidden "
                            f"dependency that should be made explicit or "
                            f"refactored."
                        ),
                        file_path=context.file_path,
                        line_start=1,
                        line_end=1,
                        confidence=0.65,
                        tags=["git", "coupling", "co-change"],
                    )
                )

        # --- Source-level import heuristic ---
        if not context.source_code or not context.source_code.strip():
            return findings

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return findings

        imports = find_imports(root, context.language)
        if not imports:
            return findings

        # Collect distinct top-level packages
        top_level_packages: set[str] = set()
        first_import_line = imports[0].start_line if imports else 1
        last_import_line = imports[-1].end_line if imports else 1

        for imp in imports:
            pkg = _top_level_package(imp.module)
            if pkg is not None:
                top_level_packages.add(pkg)

        if len(top_level_packages) >= _TOP_LEVEL_PACKAGE_THRESHOLD:
            sorted_pkgs = sorted(top_level_packages)
            findings.append(
                Finding(
                    agent_name="co_change_miner",
                    severity="low",
                    category="maintenance",
                    title=(
                        f"High import coupling: file imports from "
                        f"{len(top_level_packages)} distinct top-level packages"
                    ),
                    description=(
                        f"'{context.file_path}' imports from "
                        f"{len(top_level_packages)} different top-level "
                        f"packages/directories: {', '.join(sorted_pkgs)}. "
                        f"This breadth of dependencies suggests tight coupling "
                        f"to many parts of the codebase. Changes in any of "
                        f"these packages may require updates here. Consider "
                        f"reducing the coupling by introducing a facade, "
                        f"splitting the file, or consolidating related imports."
                    ),
                    file_path=context.file_path,
                    line_start=first_import_line,
                    line_end=last_import_line,
                    confidence=0.65,
                    tags=["git", "coupling", "co-change"],
                )
            )

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Files that depend on many distant modules "
            f"are fragile — a change in any dependency may silently break "
            f"this file. Reducing coupling improves maintainability and "
            f"makes the codebase easier to reason about."
        )


def _directory_of(file_path: str) -> str:
    """Return the parent directory portion of a file path."""
    normalized = file_path.replace("\\", "/")
    if "/" in normalized:
        return normalized.rsplit("/", 1)[0]
    return ""
