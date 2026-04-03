"""Insecure Randomness Detector — flags non-cryptographic PRNGs."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls
from fathom_sdk.context.ast_parser import parse

# (callee_name_set, full_text_pattern, languages)
_PATTERNS: list[tuple[set[str], re.Pattern[str] | None, set[str]]] = [
    # Python: random module
    (
        {"random", "randint", "choice", "randrange", "uniform", "shuffle", "sample", "getrandbits"},
        re.compile(r"random\.\w+\s*\("),
        {"python"},
    ),
    # JS/TS: Math.random()
    (
        {"random"},
        re.compile(r"Math\.random\s*\("),
        {"javascript", "typescript"},
    ),
    # Java: Random class (not SecureRandom) — callee name alone suffices
    (
        {"nextInt", "nextLong", "nextDouble", "nextFloat", "nextBoolean", "nextGaussian"},
        None,
        {"java"},
    ),
    # Go: math/rand
    (
        {"Intn", "Int", "Float64", "Float32", "Int63", "Int31", "Seed"},
        re.compile(r"rand\.\w+\s*\("),
        {"go"},
    ),
]

# Source-level patterns
_SOURCE_PATTERNS: list[tuple[re.Pattern[str], set[str]]] = [
    (re.compile(r"import\s+random\b"), {"python"}),
    (re.compile(r"from\s+random\s+import"), {"python"}),
    (re.compile(r"Math\.random\s*\("), {"javascript", "typescript"}),
    (re.compile(r"new\s+Random\s*\("), {"java"}),
    (re.compile(r'"math/rand"'), {"go"}),
]


class InsecureRandomnessAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="insecure_randomness",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development"],
            methodology="security",
            axis_type="aware",
            tags=["security", "randomness", "prng"],
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

        lang = context.language.lower()
        findings: list[Finding] = []
        seen_lines: set[int] = set()

        # For Go, check if crypto/rand is imported (not insecure)
        uses_crypto_rand = lang == "go" and '"crypto/rand"' in context.source_code

        for call in find_calls(root, context.language):
            for callee_set, text_pat, langs in _PATTERNS:
                if lang not in langs:
                    continue
                if call.callee_name not in callee_set:
                    continue
                if text_pat is not None and not text_pat.search(call.full_text):
                    continue
                # Go: skip if using crypto/rand (secure)
                if uses_crypto_rand and lang == "go":
                    continue
                if call.start_line in seen_lines:
                    continue
                seen_lines.add(call.start_line)
                findings.append(
                    Finding(
                        agent_name="insecure_randomness",
                        severity="medium",
                        category="security",
                        title=f"Insecure PRNG: {call.full_text.split(chr(10))[0][:60]}",
                        description=(
                            f"Non-cryptographic random at line {call.start_line}. "
                            f"Use secrets/os.urandom (Python), crypto.getRandomValues (JS), "
                            f"SecureRandom (Java), or crypto/rand (Go) for security."
                        ),
                        file_path=context.file_path,
                        line_start=call.start_line,
                        line_end=call.end_line,
                        confidence=0.85,
                        tags=["security", "randomness", "prng"],
                    )
                )
                break
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Non-cryptographic PRNGs are predictable and must "
            f"never be used for tokens, keys, passwords, or security decisions."
        )
