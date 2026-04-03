"""Unnecessary Recomputation — flags repeated expensive calls and loop-invariant ops."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, find_loops
from fathom_sdk.context.ast_parser import parse

_EXPENSIVE_CALLS: set[str] = {
    "sorted",
    "compile",
    "read",
    "load",
    "parse",
    "decode",
    "encode",
    "dumps",
    "loads",
    "deepcopy",
    "copy",
    "readlines",
    "readall",
    "fetchall",
    "query",
    "search",
    "findall",
    "match",
    "getElementsByTagName",
    "querySelectorAll",
    "getElementById",
    "getenv",
    "environ",
}


class UnnecessaryRecomputationAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="unnecessary_recomputation",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="performance",
            axis_type="aware",
            tags=["performance", "recomputation", "optimization"],
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
        loops = find_loops(root, context.language)
        calls = find_calls(root, context.language)

        # 1. Expensive calls inside loops
        for call in calls:
            if call.callee_name not in _EXPENSIVE_CALLS:
                continue
            for loop in loops:
                if call.start_line >= loop.start_line and call.end_line <= loop.end_line:
                    findings.append(
                        Finding(
                            agent_name="unnecessary_recomputation",
                            severity="medium",
                            category="performance",
                            title=(f"Expensive call '{call.callee_name}' inside loop"),
                            description=(
                                f"'{call.callee_name}' at line {call.start_line} "
                                f"is called inside a loop. If the result doesn't change "
                                f"per iteration, hoist it outside the loop."
                            ),
                            file_path=context.file_path,
                            line_start=call.start_line,
                            line_end=call.end_line,
                            confidence=0.80,
                            tags=["performance", "recomputation", "optimization"],
                        )
                    )
                    break

        # 2. Identical calls repeated in same scope
        scope_calls: dict[str | None, list[tuple[str, int, int]]] = {}
        for call in calls:
            key = call.enclosing_function
            scope_calls.setdefault(key, []).append(
                (call.full_text.strip(), call.start_line, call.end_line)
            )

        for scope, scope_list in scope_calls.items():
            seen: dict[str, tuple[int, int]] = {}
            for text, start, end in scope_list:
                if len(text) < 10:  # skip trivially short calls
                    continue
                if text in seen:
                    first_start, first_end = seen[text]
                    findings.append(
                        Finding(
                            agent_name="unnecessary_recomputation",
                            severity="low",
                            category="performance",
                            title=f"Duplicate call: {text[:50]}",
                            description=(
                                f"Identical call at lines {first_start} and {start}. "
                                f"Consider caching the result in a variable."
                            ),
                            file_path=context.file_path,
                            line_start=start,
                            line_end=end,
                            confidence=0.72,
                            tags=["performance", "recomputation"],
                        )
                    )
                else:
                    seen[text] = (start, end)
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Repeated expensive operations waste CPU cycles. "
            f"Cache results in a variable and reuse them."
        )
