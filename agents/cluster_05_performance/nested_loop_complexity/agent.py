"""Nested Loop Complexity — flags deeply nested loops as O(n^k) performance risk."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_loops
from fathom_sdk.context.ast_parser import parse

_SEVERITIES = {2: "low", 3: "medium"}  # depth >= 4 -> "high"


class NestedLoopComplexityAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="nested_loop_complexity",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="performance",
            axis_type="aware",
            tags=["performance", "complexity", "nested-loops"],
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
        for loop in find_loops(root, context.language):
            if loop.nesting_depth < 2:
                continue
            severity = _SEVERITIES.get(loop.nesting_depth, "high")
            findings.append(
                Finding(
                    agent_name="nested_loop_complexity",
                    severity=severity,
                    category="performance",
                    title=f"Nested loop depth {loop.nesting_depth} — O(n^{loop.nesting_depth})",
                    description=(
                        f"Loop at line {loop.start_line} is nested {loop.nesting_depth} "
                        f"levels deep, giving O(n^{loop.nesting_depth}) time complexity. "
                        f"Consider algorithmic optimization."
                    ),
                    file_path=context.file_path,
                    line_start=loop.start_line,
                    line_end=loop.end_line,
                    confidence=0.93,
                    tags=["performance", "complexity", "nested-loops"],
                )
            )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Deeply nested loops multiply execution time "
            f"exponentially with input size. Consider hash maps, sorting, "
            f"or other algorithmic approaches to reduce complexity."
        )
