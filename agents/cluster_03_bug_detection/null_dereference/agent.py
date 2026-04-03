"""Null Dereference Predictor — tracks null assignments and flags unguarded use."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_parser import parse

_NULL_VALUES: dict[str, set[str]] = {
    "python": {"None"},
    "javascript": {"null", "undefined"},
    "typescript": {"null", "undefined"},
    "java": {"null"},
    "go": {"nil"},
}

_NULL_CHECK_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"if\s+\w+\s+is\s+not\s+None"),
        re.compile(r"if\s+\w+\s*:"),
        re.compile(r"if\s+\w+\s+is\s+None"),
        re.compile(r"if\s+not\s+\w+"),
    ],
    "javascript": [
        re.compile(r"if\s*\(\s*\w+\s*[!=]==?\s*null"),
        re.compile(r"if\s*\(\s*\w+\s*\)"),
        re.compile(r"\w+\s*\?\.\s*"),
        re.compile(r"\w+\s*&&\s*\w+\."),
    ],
    "typescript": [
        re.compile(r"if\s*\(\s*\w+\s*[!=]==?\s*null"),
        re.compile(r"if\s*\(\s*\w+\s*\)"),
        re.compile(r"\w+\s*\?\.\s*"),
        re.compile(r"\w+\s*&&\s*\w+\."),
    ],
    "java": [
        re.compile(r"if\s*\(\s*\w+\s*!=\s*null"),
        re.compile(r"if\s*\(\s*\w+\s*==\s*null"),
        re.compile(r"Objects\.requireNonNull"),
        re.compile(r"Optional\.ofNullable"),
    ],
    "go": [
        re.compile(r"if\s+\w+\s*!=\s*nil"),
        re.compile(r"if\s+\w+\s*==\s*nil"),
        re.compile(r"if\s+err\s*!=\s*nil"),
    ],
}

# Match assignments, including those with type prefixes like:
# x = None, let x = null, String x = null, var x = nil
_ASSIGN_PATTERN = re.compile(
    r"(?:(?:let|var|const|final)\s+)?"  # optional JS/Go keyword
    r"(?:(?:[A-Z]\w*(?:<[^>]+>)?)\s+)?"  # optional Java type prefix
    r"(\w+)\s*[:=]=?\s*(.+)"
)


class NullDereferenceAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="null_dereference",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="bug_detection",
            axis_type="aware",
            tags=["bug", "null", "dereference", "npe"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        null_vals = _NULL_VALUES.get(lang)
        if not null_vals:
            return []
        check_patterns = _NULL_CHECK_PATTERNS.get(lang, [])

        lines = context.source_code.splitlines()
        findings: list[Finding] = []

        # Track variables assigned null per scope (simplified: per-file)
        null_vars: dict[str, int] = {}  # var_name -> line where assigned null
        guarded_vars: set[str] = set()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue

            # Check for null guards
            for cp in check_patterns:
                if cp.search(stripped):
                    # Extract variable name from guard
                    words = re.findall(r"\b\w+\b", stripped)
                    for w in words:
                        if w in null_vars:
                            guarded_vars.add(w)

            # Check for null assignments
            m = _ASSIGN_PATTERN.match(stripped)
            if m:
                var_name = m.group(1)
                value = m.group(2).strip().rstrip(";")
                if value in null_vals:
                    null_vars[var_name] = i
                    guarded_vars.discard(var_name)
                elif var_name in null_vars:
                    # Re-assigned to non-null
                    del null_vars[var_name]

            # Check for dereference of null-assigned variables
            for var_name, assign_line in list(null_vars.items()):
                if var_name in guarded_vars:
                    continue
                if i <= assign_line:
                    continue
                # Look for var.method() or var.property
                deref_pattern = re.compile(rf"\b{re.escape(var_name)}\s*\.")
                if deref_pattern.search(stripped):
                    # Make sure it's not inside a null check on this line
                    is_guarded = any(cp.search(stripped) for cp in check_patterns)
                    if not is_guarded:
                        findings.append(
                            Finding(
                                agent_name="null_dereference",
                                severity="high",
                                category="bug",
                                title=(
                                    f"Possible null dereference: '{var_name}' "
                                    f"used without null check"
                                ),
                                description=(
                                    f"'{var_name}' was assigned null/None/nil at "
                                    f"line {assign_line} and dereferenced at line {i} "
                                    f"without a null check."
                                ),
                                file_path=context.file_path,
                                line_start=i,
                                line_end=i,
                                confidence=0.82,
                                tags=["bug", "null", "dereference", "npe"],
                            )
                        )
                        # Stop tracking this var after first finding
                        del null_vars[var_name]
                        break
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Dereferencing null/None/nil causes runtime crashes. "
            f"Add a null check before accessing methods or properties."
        )
