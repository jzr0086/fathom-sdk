"""Taint Flow Analyzer — traces data flows from untrusted sources to dangerous sinks.

This is the comprehensive, unified taint analysis agent for Fathom.  Other
security agents (sql_injection, xss_pattern, etc.) have their own pattern-based
detection; this agent provides a graph-relational view by tracing variable
assignments from taint sources through the program to security-sensitive sinks.
"""

from __future__ import annotations

from typing import Optional

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, CodeFix, Finding
from fathom_sdk.context.ast_parser import parse
from fathom_sdk.context.taint_analysis import TaintFlow, trace_taint_flows

# Map vulnerability types to Finding severity levels.
_VULN_SEVERITY: dict[str, str] = {
    "sql_injection": "critical",
    "command_injection": "critical",
    "deserialization": "critical",
    "path_traversal": "high",
    "ssrf": "high",
    "xss": "high",
}

_DEFAULT_SEVERITY = "high"


class TaintFlowAnalyzerAgent(BaseReviewAgent):
    """Traces taint flows from untrusted sources to dangerous sinks."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="taint_flow_analyzer",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development"],
            methodology="graph_relational",
            axis_type="aware",
            tags=["security", "taint", "data-flow"],
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

        flows = trace_taint_flows(root, context.language, context.source_code)
        if not flows:
            return []

        findings: list[Finding] = []
        for flow in flows:
            severity = _VULN_SEVERITY.get(flow.vuln_type, _DEFAULT_SEVERITY)
            title = (
                f"{flow.vuln_type} via taint flow from "
                f"{flow.source.source_type} to {flow.sink.sink_type}"
            )
            description = (
                f"Tainted data flows from {flow.source.source_type} source "
                f"'{flow.source.variable}' (line {flow.source.line}) to "
                f"{flow.sink.sink_type} sink '{flow.sink.callee}' "
                f"(line {flow.sink.line}) through variable "
                f"'{flow.tainted_variable}'. This may lead to {flow.vuln_type}."
            )
            findings.append(
                Finding(
                    agent_name="taint_flow_analyzer",
                    severity=severity,
                    category="security",
                    title=title,
                    description=description,
                    file_path=context.file_path,
                    line_start=flow.source.line,
                    line_end=flow.sink.line,
                    confidence=flow.confidence,
                    tags=["security", "taint", flow.vuln_type],
                )
            )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Taint analysis traces untrusted user input "
            f"through variable assignments to security-sensitive operations. "
            f"Sanitize or validate all data before it reaches dangerous sinks "
            f"such as SQL queries, shell commands, file paths, or HTML output."
        )

    def suggest_fix(self, finding: Finding) -> Optional[CodeFix]:
        return None
