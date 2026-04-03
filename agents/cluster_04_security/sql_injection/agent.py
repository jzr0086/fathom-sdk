"""SQL Injection Detector — flags string interpolation in database queries."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls
from fathom_sdk.context.ast_parser import parse

_DB_CALL_NAMES: set[str] = {
    "execute",
    "executemany",
    "raw",
    "query",
    "prepare",
    "exec",
    "Exec",
    "Query",
    "Prepare",
    "QueryRow",
    "executeQuery",
    "executeUpdate",
    "prepareStatement",
    "rawQuery",
}

_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|UNION|WHERE)\b", re.IGNORECASE
)

_INTERPOLATION_SIGNALS = [
    re.compile(r'f["\']'),  # Python f-string
    re.compile(r"\$\{"),  # JS template literal
    re.compile(r"\.format\s*\("),  # .format()
    re.compile(r'\+\s*["\']|\+\s*\w'),  # string concatenation
    re.compile(r'["\'\`]\s*\+'),  # string + variable
]

# %s/%d is only an injection signal when used with the % operator, not as a
# parameterized query placeholder like execute("...%s", (val,))
_PERCENT_FORMAT = re.compile(r"%[sd]")
_PARAMETERIZED_HINT = re.compile(r"""%s['"]?\s*,""")


def _has_interpolation(text: str) -> bool:
    """Check if text contains interpolation signals (not parameterized placeholders)."""
    for signal in _INTERPOLATION_SIGNALS:
        if signal.search(text):
            return True
    # %s is only a signal if used with % operator, not as parameterized placeholder
    if _PERCENT_FORMAT.search(text) and not _PARAMETERIZED_HINT.search(text):
        return True
    return False


class SqlInjectionAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="sql_injection",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["database_engineering", "web_development", "security_engineering"],
            methodology="security",
            axis_type="aware",
            tags=["security", "injection", "sql"],
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
        seen_lines: set[int] = set()

        # AST-based: find DB calls with interpolated strings
        for call in find_calls(root, context.language):
            if call.callee_name not in _DB_CALL_NAMES:
                continue
            text = call.full_text
            if not _SQL_KEYWORDS.search(text):
                continue
            if _has_interpolation(text):
                if call.start_line not in seen_lines:
                    seen_lines.add(call.start_line)
                    findings.append(
                        Finding(
                            agent_name="sql_injection",
                            severity="critical",
                            category="security",
                            title="Potential SQL injection via string interpolation",
                            description=(
                                f"Database query at line {call.start_line} uses string "
                                f"interpolation/concatenation instead of parameterized "
                                f"queries. This is vulnerable to SQL injection."
                            ),
                            file_path=context.file_path,
                            line_start=call.start_line,
                            line_end=call.end_line,
                            confidence=0.90,
                            tags=["security", "injection", "sql"],
                        )
                    )

        # Source-level fallback for multi-line patterns
        lines = context.source_code.splitlines()
        for i, line in enumerate(lines, 1):
            if i in seen_lines:
                continue
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            lower = stripped.lower()
            # Look for execute/query with f-string or format on same line
            if any(name in lower for name in ("execute", "query", "exec")):
                if _SQL_KEYWORDS.search(line):
                    if _has_interpolation(line):
                        seen_lines.add(i)
                        findings.append(
                            Finding(
                                agent_name="sql_injection",
                                severity="critical",
                                category="security",
                                title="Potential SQL injection via string interpolation",
                                description=(
                                    f"Line {i} constructs a SQL query using string "
                                    f"interpolation. Use parameterized queries instead."
                                ),
                                file_path=context.file_path,
                                line_start=i,
                                line_end=i,
                                confidence=0.85,
                                tags=["security", "injection", "sql"],
                            )
                        )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. SQL injection allows attackers to execute arbitrary "
            f"database commands. Always use parameterized queries (?, %s, $1) instead "
            f"of string concatenation or interpolation."
        )
