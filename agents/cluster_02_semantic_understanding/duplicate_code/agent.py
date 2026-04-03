"""Duplicate Code Detector — finds functions with identical normalized bodies."""

from __future__ import annotations

import hashlib
import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text
from fathom_sdk.context.ast_parser import parse

_MIN_LINES = 5
_IDENTIFIER = re.compile(r"\b[a-zA-Z_]\w*\b")
_STRING_LIT = re.compile(r'(?:"[^"]*"|\'[^\']*\'|`[^`]*`)')
_NUMBER_LIT = re.compile(r"\b\d+(?:\.\d+)?\b")


def _normalize(text: str) -> str:
    """Normalize a function body for duplicate comparison."""
    # Strip comments (simplified: line comments only)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("#", "//", "/*", "*")):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    # Replace strings and numbers with placeholders
    text = _STRING_LIT.sub("STR", text)
    text = _NUMBER_LIT.sub("NUM", text)
    # Replace identifiers with placeholder
    text = _IDENTIFIER.sub("ID", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


class DuplicateCodeAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="duplicate_code",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="semantic_understanding",
            axis_type="agnostic",
            tags=["quality", "duplication", "dry"],
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

        functions = find_functions(root, context.language)
        # Filter to functions with enough lines
        functions = [f for f in functions if f.line_count >= _MIN_LINES]
        if len(functions) < 2:
            return []

        # Hash normalized bodies
        hash_groups: dict[str, list] = {}
        for func in functions:
            body_text = get_node_text(func.node)
            # Remove the function signature (first line)
            body_lines = body_text.split("\n", 1)
            body = body_lines[1] if len(body_lines) > 1 else body_text
            normalized = _normalize(body)
            if len(normalized) < 20:  # skip trivially short normalized text
                continue
            h = _hash_text(normalized)
            hash_groups.setdefault(h, []).append(func)

        findings: list[Finding] = []
        for h, group in hash_groups.items():
            if len(group) < 2:
                continue
            names = [f.name for f in group]
            for func in group:
                others = [n for n in names if n != func.name]
                findings.append(
                    Finding(
                        agent_name="duplicate_code",
                        severity="medium",
                        category="quality",
                        title=f"Duplicate function '{func.name}'",
                        description=(
                            f"Function '{func.name}' (lines {func.start_line}-"
                            f"{func.end_line}) has the same normalized body as: "
                            f"{', '.join(others)}. Extract shared logic."
                        ),
                        file_path=context.file_path,
                        line_start=func.start_line,
                        line_end=func.end_line,
                        confidence=0.85,
                        tags=["quality", "duplication", "dry"],
                    )
                )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Duplicate code means duplicate bugs and duplicate "
            f"maintenance effort. Extract the shared logic into a reusable function."
        )
