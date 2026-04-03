"""Hardcoded Secrets Detector — flags credentials in source code."""

from __future__ import annotations

import math
import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_parser import parse

_SECRET_KEYWORDS = re.compile(
    r"(password|passwd|secret|api_?key|apikey|token|auth|credential|"
    r"private_?key|access_?key|client_?secret|db_?pass|database_?pass|"
    r"encryption_?key|signing_?key|jwt_?secret)",
    re.IGNORECASE,
)

_ASSIGNMENT_PATTERN = re.compile(
    r"""(?:^|\s)(\w*(?:password|passwd|secret|api_?key|apikey|token|auth|"""
    r"""credential|private_?key|access_?key|client_?secret|jwt_?secret)\w*)"""
    r"""\s*[:=]\s*(['"`])(.+?)\2""",
    re.IGNORECASE | re.MULTILINE,
)

# Placeholders and non-secrets
_PLACEHOLDER_PATTERNS = re.compile(
    r"^(xxx+|placeholder|changeme|todo|fixme|your[_-]?\w*here|example|"
    r"test|dummy|fake|sample|replace[_-]?me|\*+|\.{3,}|<[^>]+>|\$\{.+\})$",
    re.IGNORECASE,
)


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


class HardcodedSecretsAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="hardcoded_secrets",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development", "cloud_infrastructure"],
            methodology="security",
            axis_type="aware",
            tags=["security", "secrets", "credentials"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        # This agent works on source text, no AST needed
        findings: list[Finding] = []
        seen_lines: set[int] = set()

        for i, line in enumerate(context.source_code.splitlines(), 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue

            for match in _ASSIGNMENT_PATTERN.finditer(line):
                var_name = match.group(1)
                value = match.group(3)

                # Skip placeholders and empty/short values
                if not value or len(value) < 4:
                    continue
                if _PLACEHOLDER_PATTERNS.match(value):
                    continue
                # Skip environment variable references
                if value.startswith(("os.environ", "process.env", "System.getenv", "os.Getenv")):
                    continue
                if value.startswith("${") or value.startswith("%("):
                    continue

                if i not in seen_lines and _SECRET_KEYWORDS.search(var_name):
                    seen_lines.add(i)
                    entropy = _shannon_entropy(value)
                    confidence = 0.88 if entropy > 3.5 else 0.80
                    findings.append(
                        Finding(
                            agent_name="hardcoded_secrets",
                            severity="critical",
                            category="security",
                            title=f"Hardcoded secret in '{var_name}'",
                            description=(
                                f"Variable '{var_name}' at line {i} contains what appears "
                                f"to be a hardcoded credential. Use environment variables "
                                f"or a secrets manager instead."
                            ),
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=confidence,
                            tags=["security", "secrets", "credentials"],
                        )
                    )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Hardcoded secrets can be extracted from source code, "
            f"version history, or compiled binaries. Use environment variables, "
            f"secret managers (Vault, AWS Secrets Manager), or .env files."
        )
