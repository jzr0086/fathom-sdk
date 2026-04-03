"""SSRF (Server-Side Request Forgery) Detector — flags user-controlled URLs in HTTP client calls."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, get_node_text
from fathom_sdk.context.ast_parser import parse
from fathom_sdk.context.taint_analysis import trace_taint_flows

# HTTP client function/method names that perform outbound requests.
_HTTP_CLIENT_NAMES: set[str] = {
    # Python
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
    "request",
    "urlopen",
    # JavaScript / TypeScript
    "fetch",
    # Go
    "Get",
    "Post",
    "Do",
    "NewRequest",
    # Java
    "openConnection",
}

# Broader patterns matched against the full call text to catch qualified calls
# such as ``requests.get(...)``, ``axios.post(...)``, etc.
_HTTP_CALL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brequests\.(get|post|put|patch|delete|head|options|request)\s*\("),
    re.compile(r"\burllib\.request\.urlopen\s*\("),
    re.compile(r"\bhttpx\.(get|post|put|patch|delete|head|options|request)\s*\("),
    re.compile(r"\bfetch\s*\("),
    re.compile(r"\baxios\.(get|post|put|patch|delete|head|options|request)\s*\("),
    re.compile(r"\bhttp\.Get\s*\("),
    re.compile(r"\bhttp\.Post\s*\("),
    re.compile(r"\bhttp\.NewRequest\s*\("),
    re.compile(r"\bHttpURLConnection\b"),
    re.compile(r"\burlopen\s*\("),
]

# Signals that the URL argument is built via interpolation / concatenation
# rather than being a safe, static string.
_INTERPOLATION_SIGNALS: list[re.Pattern[str]] = [
    re.compile(r'f["\']'),          # Python f-string
    re.compile(r"\$\{"),            # JS/TS template literal
    re.compile(r"\.format\s*\("),   # Python str.format()
    re.compile(r'\+\s*["\']|\+\s*\w'),  # string concatenation (right)
    re.compile(r'["\'\`]\s*\+'),         # string concatenation (left)
]


def _is_http_client_call(callee_name: str, full_text: str) -> bool:
    """Return True if the call is a known HTTP client invocation."""
    if callee_name in _HTTP_CLIENT_NAMES:
        # Verify via full text to reduce false positives (e.g. a local ``get()``).
        for pattern in _HTTP_CALL_PATTERNS:
            if pattern.search(full_text):
                return True
    # Also check full text patterns for cases where callee extraction is lossy.
    for pattern in _HTTP_CALL_PATTERNS:
        if pattern.search(full_text):
            return True
    return False


def _has_interpolation(text: str) -> bool:
    """Return True if *text* contains string interpolation or concatenation signals."""
    for signal in _INTERPOLATION_SIGNALS:
        if signal.search(text):
            return True
    return False


class SsrfAgent(BaseReviewAgent):
    """Detects user-controlled URLs flowing into HTTP client calls (SSRF)."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="ssrf",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development"],
            methodology="security",
            axis_type="aware",
            tags=["security", "ssrf", "injection"],
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

        findings: list[Finding] = []
        seen_lines: set[int] = set()

        # --- Pass 1: AST-based detection of interpolated URL arguments ----
        for call in find_calls(root, context.language):
            if not _is_http_client_call(call.callee_name, call.full_text):
                continue
            if _has_interpolation(call.full_text):
                if call.start_line not in seen_lines:
                    seen_lines.add(call.start_line)
                    findings.append(
                        self._make_finding(
                            title="Potential SSRF via interpolated URL",
                            description=(
                                f"HTTP request at line {call.start_line} uses string "
                                f"interpolation/concatenation to build the URL. "
                                f"User-controlled input in the URL can lead to "
                                f"server-side request forgery."
                            ),
                            file_path=context.file_path,
                            line_start=call.start_line,
                            line_end=call.end_line,
                        )
                    )

        # --- Pass 2: source-level fallback for patterns the AST may miss --
        lines = context.source_code.splitlines()
        for i, line in enumerate(lines, 1):
            if i in seen_lines:
                continue
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            is_http = any(p.search(line) for p in _HTTP_CALL_PATTERNS)
            if is_http and _has_interpolation(line):
                seen_lines.add(i)
                findings.append(
                    self._make_finding(
                        title="Potential SSRF via interpolated URL",
                        description=(
                            f"Line {i} builds an HTTP request URL using string "
                            f"interpolation. Validate and restrict the target URL "
                            f"to prevent SSRF."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                    )
                )

        # --- Pass 3: taint analysis for tracked source-to-sink flows ------
        try:
            flows = trace_taint_flows(root, context.language, context.source_code)
            for flow in flows:
                if flow.vuln_type != "ssrf":
                    continue
                if flow.sink.line in seen_lines:
                    continue
                seen_lines.add(flow.sink.line)
                findings.append(
                    self._make_finding(
                        title="Potential SSRF via tainted URL",
                        description=(
                            f"Tainted variable '{flow.tainted_variable}' "
                            f"(from {flow.source.source_type} at line "
                            f"{flow.source.line}) flows into HTTP call "
                            f"'{flow.sink.callee}' at line {flow.sink.line}."
                        ),
                        file_path=context.file_path,
                        line_start=flow.sink.line,
                        line_end=flow.sink.line,
                    )
                )
        except Exception:
            pass  # taint analysis is best-effort; never crash the agent

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Server-Side Request Forgery (SSRF) allows an attacker "
            f"to make the server issue arbitrary HTTP requests. This can expose "
            f"internal services, cloud metadata endpoints, and sensitive data. "
            f"Always validate and allowlist target URLs."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_finding(
        *,
        title: str,
        description: str,
        file_path: str,
        line_start: int,
        line_end: int,
    ) -> Finding:
        return Finding(
            agent_name="ssrf",
            severity="high",
            category="security",
            title=title,
            description=description,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            confidence=0.82,
            tags=["security", "ssrf", "injection"],
        )
