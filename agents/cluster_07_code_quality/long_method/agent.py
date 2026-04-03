"""Long Method Detector — flags functions exceeding line count thresholds."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions
from fathom_sdk.context.ast_parser import parse

_THRESHOLDS = [(50, "high"), (30, "medium"), (20, "low")]


class LongMethodAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="long_method",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="code_quality",
            axis_type="aware",
            tags=["quality", "maintainability", "long-method"],
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
        findings: list[Finding] = []
        for func in find_functions(root, context.language):
            for threshold, severity in _THRESHOLDS:
                if func.line_count > threshold:
                    findings.append(
                        Finding(
                            agent_name="long_method",
                            severity=severity,
                            category="quality",
                            title=f"Long method '{func.name}' ({func.line_count} lines)",
                            description=(
                                f"Function '{func.name}' is {func.line_count} lines long "
                                f"(threshold: {threshold}). Consider breaking it into "
                                f"smaller, focused functions."
                            ),
                            file_path=context.file_path,
                            line_start=func.start_line,
                            line_end=func.end_line,
                            confidence=0.95,
                            tags=["quality", "maintainability", "long-method"],
                        )
                    )
                    break
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Long methods are harder to understand, test, and "
            f"maintain. Extract logical sections into well-named helper functions."
        )
