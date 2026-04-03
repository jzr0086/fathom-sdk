"""N+1 Query Detector — flags database calls inside loop bodies."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, find_loops
from fathom_sdk.context.ast_parser import parse

_DB_CALL_NAMES: set[str] = {
    "execute",
    "executemany",
    "fetchone",
    "fetchall",
    "fetchmany",
    "query",
    "find",
    "findOne",
    "findAll",
    "findById",
    "filter",
    "select",
    "insert",
    "update",
    "delete",
    "save",
    "create",
    "Query",
    "Exec",
    "QueryRow",
    "Find",
    "First",
    "Where",
    "executeQuery",
    "executeUpdate",
    "prepareStatement",
    "get",
    "get_object_or_404",
    "get_or_create",
    "bulk_create",
}


class NPlusOneQueryAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="n_plus_one_query",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["database_engineering", "web_development"],
            methodology="performance",
            axis_type="aware",
            tags=["performance", "database", "n-plus-one"],
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

        loops = find_loops(root, context.language)
        if not loops:
            return []
        calls = find_calls(root, context.language)

        findings: list[Finding] = []
        for call in calls:
            if call.callee_name not in _DB_CALL_NAMES:
                continue
            for loop in loops:
                if call.start_line >= loop.start_line and call.end_line <= loop.end_line:
                    findings.append(
                        Finding(
                            agent_name="n_plus_one_query",
                            severity="high",
                            category="performance",
                            title=(f"N+1 query: '{call.callee_name}' called inside loop"),
                            description=(
                                f"Database call '{call.callee_name}' at line "
                                f"{call.start_line} is inside a loop starting at "
                                f"line {loop.start_line}. This causes N+1 queries. "
                                f"Use batch/bulk operations or eager loading."
                            ),
                            file_path=context.file_path,
                            line_start=call.start_line,
                            line_end=call.end_line,
                            confidence=0.88,
                            tags=["performance", "database", "n-plus-one"],
                        )
                    )
                    break  # only report once per call
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. N+1 queries execute one query per loop iteration "
            f"instead of a single batch query. This dramatically impacts performance. "
            f"Use prefetch_related/select_related (Django), eager loading (SQLAlchemy), "
            f"or IN clauses."
        )
