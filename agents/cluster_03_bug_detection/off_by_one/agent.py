"""Off-by-One Error Classifier — detects common boundary errors."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_parser import parse

_PATTERNS: list[tuple[re.Pattern[str], str, str, str, float]] = [
    # (pattern, title, description, severity, confidence)
    (
        re.compile(r"\[\s*len\s*\(\s*\w+\s*\)\s*\]"),
        "Array access at len()",
        "Accessing array[len(array)] is out of bounds. Use len()-1 for the last element.",
        "high",
        0.90,
    ),
    (
        re.compile(r"\[\s*\w+\.length\s*\]"),
        "Array access at .length",
        "Accessing array[array.length] is out of bounds. Use .length-1 for the last.",
        "high",
        0.90,
    ),
    (
        re.compile(r"\[\s*\w+\.size\(\)\s*\]"),
        "Array access at .size()",
        "Accessing list[list.size()] is out of bounds.",
        "high",
        0.88,
    ),
    (
        re.compile(r"\.get\s*\(\s*\w+\.size\(\)\s*\)"),
        "List.get(list.size()) out of bounds",
        "Accessing list.get(list.size()) is out of bounds. Use .size()-1 for the last.",
        "high",
        0.88,
    ),
    (
        re.compile(r"<=\s*len\s*\("),
        "Loop bound <= len()",
        "Using <= len() in a loop condition often causes an off-by-one error. Use < len().",
        "medium",
        0.82,
    ),
    (
        re.compile(r"<=\s*\w+\.length\b"),
        "Loop bound <= .length",
        "Using <= .length in a loop condition causes an off-by-one error. Use < .length.",
        "medium",
        0.82,
    ),
    (
        re.compile(r"<=\s*\w+\.size\(\)"),
        "Loop bound <= .size()",
        "Using <= .size() in a loop condition causes an off-by-one error.",
        "medium",
        0.82,
    ),
    (
        re.compile(r"range\s*\(\s*1\s*,\s*len\s*\(\s*\w+\s*\)\s*\)"),
        "range(1, len()) skips first element",
        "range(1, len(x)) skips index 0. Verify this is intentional.",
        "low",
        0.70,
    ),
    (
        re.compile(r"range\s*\(\s*len\s*\(\s*\w+\s*\)\s*\+\s*1\s*\)"),
        "range(len()+1) may exceed bounds",
        "range(len(x)+1) iterates one past the last index. Check array accesses inside.",
        "medium",
        0.78,
    ),
    (
        re.compile(r"range\s*\(\s*len\s*\(\s*\w+\s*\)\s*-\s*1\s*\)"),
        "range(len()-1) skips last element",
        "range(len(x)-1) stops one element early. Verify this is intentional.",
        "low",
        0.65,
    ),
    (
        re.compile(r"substring\s*\(\s*0\s*,\s*\w+\.length\(\)\s*\+\s*1\s*\)"),
        "substring beyond string length",
        "substring(0, str.length()+1) exceeds string bounds.",
        "high",
        0.88,
    ),
]


class OffByOneAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="off_by_one",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="bug_detection",
            axis_type="aware",
            tags=["bug", "off-by-one", "array-bounds"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        findings: list[Finding] = []
        for i, line in enumerate(context.source_code.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            for pattern, title, desc, severity, confidence in _PATTERNS:
                if pattern.search(line):
                    findings.append(
                        Finding(
                            agent_name="off_by_one",
                            severity=severity,
                            category="bug",
                            title=f"Possible off-by-one: {title}",
                            description=f"Line {i}: {desc}",
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=confidence,
                            tags=["bug", "off-by-one", "array-bounds"],
                        )
                    )
                    break  # one finding per line
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Off-by-one errors are among the most common bugs. "
            f"Double-check array indices and loop bounds at boundaries."
        )
