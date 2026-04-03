"""Hotspot (Git Churn) Detector — flags files with high churn and complexity.

Combines git change frequency (churn) with code complexity to identify
maintenance hotspots: files that change often AND are complex.  These are
the highest-risk areas in a codebase and the best candidates for
refactoring investment.

Algorithm:
    hotspot_score = churn_score * avg_complexity

When commit history is available the churn score comes from
``compute_churn_metrics()``.  When it is absent the agent falls back to a
pure complexity-based risk heuristic derived from the number of functions
and a lightweight cyclomatic-complexity proxy.
"""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, walk
from fathom_sdk.context.ast_parser import parse
from fathom_sdk.context.git_helpers import compute_churn_metrics

# Node types that contribute to cyclomatic complexity (language-keyed).
_DECISION_TYPES: dict[str, set[str]] = {
    "python": {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "except_clause",
        "boolean_operator",
        "conditional_expression",
    },
    "javascript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "switch_case",
        "ternary_expression",
    },
    "typescript": {
        "if_statement",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "switch_case",
        "ternary_expression",
    },
    "java": {
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "catch_clause",
        "switch_label",
        "ternary_expression",
    },
    "go": {
        "if_statement",
        "for_statement",
        "expression_case",
        "type_case",
        "communication_case",
    },
}

# Severity thresholds for the hotspot score.
_SEVERITY_THRESHOLDS = [(50, "high"), (20, "medium"), (10, "low")]

# Churn score threshold -- below this value the file is not considered a hotspot.
_CHURN_THRESHOLD = 10


def _compute_avg_complexity(root: object, language: str) -> float:
    """Compute the average cyclomatic complexity across all functions.

    Returns 1.0 when no functions are found (trivial file) so that the
    hotspot score is driven entirely by churn in that case.
    """
    functions = find_functions(root, language)  # type: ignore[arg-type]
    if not functions:
        return 1.0

    decision_types = _DECISION_TYPES.get(language.lower(), set())
    total_complexity = 0
    for func in functions:
        complexity = 1  # base complexity for each function
        for node in walk(func.node):
            if node.type in decision_types:
                complexity += 1
        total_complexity += complexity
    return total_complexity / len(functions)


def _compute_complexity_risk(root: object, language: str) -> float:
    """Fallback risk score when commit history is absent.

    Uses function count and average complexity to estimate maintenance
    risk without any git data.  Returns 0.0 for simple files.
    """
    functions = find_functions(root, language)  # type: ignore[arg-type]
    if not functions:
        return 0.0
    avg_complexity = _compute_avg_complexity(root, language)
    return len(functions) * avg_complexity


class HotspotAgent(BaseReviewAgent):
    """Detects maintenance hotspots by combining git churn with code complexity."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="hotspot",
            version="0.1.0",
            languages=["*"],
            domains=["web_development", "enterprise_engineering"],
            methodology="git_intelligence",
            axis_type="agnostic",
            tags=["git", "churn", "hotspot", "maintenance"],
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

        avg_complexity = _compute_avg_complexity(root, context.language)

        # Primary path: use git churn when commit history is available.
        if context.commit_history:
            churn = compute_churn_metrics(context.commit_history, context.file_path)
            churn_score = churn["churn_score"]

            if churn_score <= _CHURN_THRESHOLD:
                return []

            hotspot_score = churn_score * avg_complexity

            severity = self._severity_for_score(hotspot_score)
            if severity is None:
                return []

            total_lines = len(context.source_code.splitlines())
            return [
                Finding(
                    agent_name="hotspot",
                    severity=severity,
                    category="maintenance",
                    title=f"Git churn hotspot in '{context.file_path}' (score {hotspot_score:.0f})",
                    description=(
                        f"File '{context.file_path}' has a hotspot score of "
                        f"{hotspot_score:.0f} (churn={churn_score:.0f}, "
                        f"avg_complexity={avg_complexity:.1f}). "
                        f"High churn combined with high complexity indicates a "
                        f"maintenance risk. Consider refactoring to reduce complexity."
                    ),
                    file_path=context.file_path,
                    line_start=1,
                    line_end=total_lines,
                    confidence=0.70,
                    tags=["git", "churn", "hotspot", "maintenance"],
                ),
            ]

        # Fallback: no commit history -- use pure complexity risk.
        risk = _compute_complexity_risk(root, context.language)
        severity = self._severity_for_score(risk)
        if severity is None:
            return []

        total_lines = len(context.source_code.splitlines())
        return [
            Finding(
                agent_name="hotspot",
                severity=severity,
                category="maintenance",
                title=f"Complexity risk in '{context.file_path}' (score {risk:.0f})",
                description=(
                    f"File '{context.file_path}' has a complexity-based risk score "
                    f"of {risk:.0f} (no git history available). "
                    f"The file contains many functions with high average complexity. "
                    f"Consider splitting into smaller modules."
                ),
                file_path=context.file_path,
                line_start=1,
                line_end=total_lines,
                confidence=0.70,
                tags=["git", "churn", "hotspot", "maintenance"],
            ),
        ]

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Hotspots are files that change frequently and are "
            f"complex. They carry the highest maintenance burden and are the most "
            f"likely places for bugs to emerge. Reducing complexity through "
            f"refactoring lowers risk and improves developer velocity."
        )

    @staticmethod
    def _severity_for_score(score: float) -> str | None:
        """Map a hotspot/risk score to a severity level, or None if below threshold."""
        for threshold, severity in _SEVERITY_THRESHOLDS:
            if score > threshold:
                return severity
        return None
