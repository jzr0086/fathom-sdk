"""XSS Pattern Detector — flags common cross-site scripting patterns."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_parser import parse

_XSS_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"\.innerHTML\s*="),
        "innerHTML assignment",
        "Setting innerHTML with dynamic content enables XSS.",
    ),
    (
        re.compile(r"dangerouslySetInnerHTML"),
        "dangerouslySetInnerHTML",
        "React dangerouslySetInnerHTML bypasses XSS protection.",
    ),
    (
        re.compile(r"document\.write\s*\("),
        "document.write()",
        "document.write() with dynamic content enables XSS.",
    ),
    (
        re.compile(r"\|\s*safe\b"),
        "Jinja2 |safe filter",
        "The |safe filter disables HTML escaping in templates.",
    ),
    (
        re.compile(r"v-html\s*="),
        "Vue v-html directive",
        "v-html renders raw HTML and is vulnerable to XSS.",
    ),
    (
        re.compile(r"\[innerHTML\]\s*="),
        "Angular innerHTML binding",
        "Angular innerHTML binding can be exploited for XSS.",
    ),
    (
        re.compile(r"mark_safe\s*\("),
        "Django mark_safe()",
        "mark_safe() disables HTML escaping in Django templates.",
    ),
    (
        re.compile(r"Markup\s*\("),
        "Jinja2 Markup()",
        "Markup() disables auto-escaping in Jinja2.",
    ),
    (
        re.compile(r"template\.HTML\s*\("),
        "Go template.HTML()",
        "template.HTML() bypasses Go template escaping.",
    ),
    (
        re.compile(r"\.outerHTML\s*="),
        "outerHTML assignment",
        "Setting outerHTML with dynamic content enables XSS.",
    ),
    (
        re.compile(r"\.insertAdjacentHTML\s*\("),
        "insertAdjacentHTML()",
        "insertAdjacentHTML with unsanitized input enables XSS.",
    ),
]


class XssPatternAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="xss_pattern",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "security_engineering"],
            methodology="security",
            axis_type="aware",
            tags=["security", "xss", "injection"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        findings: list[Finding] = []
        for i, line in enumerate(context.source_code.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            for pattern, title, desc in _XSS_PATTERNS:
                if pattern.search(line):
                    findings.append(
                        Finding(
                            agent_name="xss_pattern",
                            severity="high",
                            category="security",
                            title=f"Potential XSS: {title}",
                            description=f"Line {i}: {desc}",
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=0.88,
                            tags=["security", "xss", "injection"],
                        )
                    )
                    break  # one finding per line
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Cross-site scripting (XSS) allows attackers to inject "
            f"malicious scripts. Always sanitize user input and use framework-provided "
            f"escaping mechanisms."
        )
