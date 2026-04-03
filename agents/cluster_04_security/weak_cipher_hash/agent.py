"""Weak Cipher/Hash Detector — flags use of known weak crypto algorithms."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls
from fathom_sdk.context.ast_parser import parse

# Callee names that are direct weak crypto calls
_WEAK_DIRECT_CALLS: set[str] = {"md5", "sha1", "MD5", "SHA1"}

# Patterns: (callee_name, regex on full_text) for parameterized calls
_WEAK_PARAM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("new", re.compile(r"new\s+DES|new\s+RC4|new\s+Blowfish", re.IGNORECASE)),
    ("createHash", re.compile(r'createHash\s*\(\s*["\'](?:md5|sha1)["\']', re.IGNORECASE)),
    ("createCipher", re.compile(r'createCipher\s*\(\s*["\'](?:des|rc4|blowfish)', re.IGNORECASE)),
    (
        "getInstance",
        re.compile(r'getInstance\s*\(\s*["\'](?:MD5|SHA-1|SHA1|DES|RC4)', re.IGNORECASE),
    ),
    ("New", re.compile(r"(?:md5|sha1|des)\.New", re.IGNORECASE)),
    ("NewCipher", re.compile(r"des\.NewCipher", re.IGNORECASE)),
]

# Source-level patterns for broader detection
_SOURCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"hashlib\.md5\s*\("),
    re.compile(r"hashlib\.sha1\s*\("),
    re.compile(r"Crypto\.Cipher\.DES"),
    re.compile(r"from\s+Crypto\.Cipher\s+import\s+DES"),
    re.compile(r"md5\.New\s*\("),
    re.compile(r"sha1\.New\s*\("),
    re.compile(r"des\.NewCipher\s*\("),
]


class WeakCipherHashAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="weak_cipher_hash",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development"],
            methodology="security",
            axis_type="aware",
            tags=["security", "crypto", "weak-hash"],
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

        # AST-based detection via calls
        for call in find_calls(root, context.language):
            flagged = False
            if call.callee_name in _WEAK_DIRECT_CALLS:
                flagged = True
            else:
                for callee, pattern in _WEAK_PARAM_PATTERNS:
                    if call.callee_name == callee and pattern.search(call.full_text):
                        flagged = True
                        break
            if flagged and call.start_line not in seen_lines:
                seen_lines.add(call.start_line)
                findings.append(
                    Finding(
                        agent_name="weak_cipher_hash",
                        severity="high",
                        category="security",
                        title=f"Weak cryptographic algorithm: {call.callee_name}",
                        description=(
                            f"Call to '{call.full_text.split(chr(10))[0][:80]}' uses a weak "
                            f"cryptographic algorithm. MD5 and SHA-1 are broken for security "
                            f"use. DES, RC4, and Blowfish are obsolete."
                        ),
                        file_path=context.file_path,
                        line_start=call.start_line,
                        line_end=call.end_line,
                        confidence=0.93,
                        tags=["security", "crypto", "weak-hash"],
                    )
                )

        # Source-level fallback for patterns not caught by AST
        for i, line in enumerate(context.source_code.splitlines(), 1):
            if i in seen_lines:
                continue
            for pat in _SOURCE_PATTERNS:
                if pat.search(line):
                    seen_lines.add(i)
                    findings.append(
                        Finding(
                            agent_name="weak_cipher_hash",
                            severity="high",
                            category="security",
                            title="Weak cryptographic algorithm detected",
                            description=f"Line {i} uses a weak crypto algorithm: {line.strip()[:80]}",
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=0.90,
                            tags=["security", "crypto", "weak-hash"],
                        )
                    )
                    break
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. MD5 and SHA-1 have known collision attacks. "
            f"Use SHA-256+ for hashing, AES-256 for encryption."
        )
