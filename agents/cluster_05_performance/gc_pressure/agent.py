"""GC Pressure Predictor — flags allocation-heavy patterns in loops."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_loops, get_node_text, walk
from fathom_sdk.context.ast_parser import parse

# Patterns indicating string concatenation with string literals in loop
_LITERAL_CONCAT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"\w+\s*\+=\s*['\"]"),  # s += "..."
        re.compile(r"\w+\s*=\s*\w+\s*\+\s*['\"]"),  # s = s + "..."
    ],
    "java": [
        re.compile(r"String\s+\w+\s*\+="),  # String s +=
        re.compile(r"\w+\s*\+=\s*\""),  # s += "..."
        re.compile(r"\w+\s*=\s*\w+\s*\+\s*\""),  # s = s + "..."
        re.compile(r"new\s+String\s*\("),  # new String()
    ],
    "go": [
        re.compile(r"\w+\s*\+=\s*\""),  # s += "..."
        re.compile(r"\w+\s*=\s*\w+\s*\+\s*\""),  # s = s + "..."
    ],
}

# Patterns for += with a variable, only flagged if target was string-initialized
_VAR_CONCAT_PATTERN = re.compile(r"(\w+)\s*\+=\s*(\w+)\s*$")
_VAR_ADD_PATTERN = re.compile(r"(\w+)\s*=\s*\1\s*\+\s*(\w+)\s*$")

# Patterns to detect string variable initialization before loops
_STRING_INIT_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"""(\w+)\s*=\s*['"]['"]"""),
    "java": re.compile(r"""String\s+(\w+)\s*=\s*\""""),
    "go": re.compile(r"""(\w+)\s*:=\s*\""""),
}

_AUTOBOXING_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "java": [
        re.compile(r"Integer\.valueOf\s*\("),
        re.compile(r"Long\.valueOf\s*\("),
        re.compile(r"Double\.valueOf\s*\("),
        re.compile(r"Boolean\.valueOf\s*\("),
        re.compile(r"new\s+Integer\s*\("),
        re.compile(r"new\s+Long\s*\("),
        re.compile(r"new\s+Double\s*\("),
    ],
}

_FIX_SUGGESTIONS: dict[str, str] = {
    "python": "Use ''.join(parts) or io.StringIO instead of += in loops.",
    "java": "Use StringBuilder instead of String concatenation in loops.",
    "go": "Use strings.Builder instead of += for string building in loops.",
}


class GcPressureAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="gc_pressure",
            version="0.1.0",
            languages=["python", "java", "go"],
            domains=["web_development", "systems_programming"],
            methodology="performance",
            axis_type="critical",
            tags=["performance", "gc", "memory", "allocation"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        if lang not in ("python", "java", "go"):
            return []
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []
        literal_patterns = _LITERAL_CONCAT_PATTERNS.get(lang, [])
        autobox_patterns = _AUTOBOXING_PATTERNS.get(lang, [])
        loops = find_loops(root, context.language)

        # Identify string-initialized variables from source lines before loops
        string_vars: set[str] = set()
        init_pat = _STRING_INIT_PATTERNS.get(lang)
        if init_pat:
            for src_line in context.source_code.splitlines():
                m = init_pat.search(src_line.strip())
                if m:
                    string_vars.add(m.group(1))

        for loop in loops:
            loop_text = get_node_text(loop.node)
            loop_lines = loop_text.splitlines()

            for j, line in enumerate(loop_lines):
                abs_line = loop.start_line + j
                found = False

                # String concatenation with literals in loop
                for pat in literal_patterns:
                    if pat.search(line):
                        fix = _FIX_SUGGESTIONS.get(lang, "Avoid allocations in loops.")
                        findings.append(
                            Finding(
                                agent_name="gc_pressure",
                                severity="medium",
                                category="performance",
                                title="String concatenation in loop creates GC pressure",
                                description=(
                                    f"String concatenation at line {abs_line} inside a "
                                    f"loop allocates a new string each iteration. {fix}"
                                ),
                                file_path=context.file_path,
                                line_start=abs_line,
                                line_end=abs_line,
                                confidence=0.85,
                                tags=["performance", "gc", "memory", "string-concat"],
                            )
                        )
                        found = True
                        break

                # Variable += var, only if target is a known string variable
                if not found and string_vars:
                    m = _VAR_CONCAT_PATTERN.search(line.strip())
                    if not m:
                        m = _VAR_ADD_PATTERN.search(line.strip())
                    if m and m.group(1) in string_vars:
                        fix = _FIX_SUGGESTIONS.get(lang, "Avoid allocations in loops.")
                        findings.append(
                            Finding(
                                agent_name="gc_pressure",
                                severity="medium",
                                category="performance",
                                title="String concatenation in loop creates GC pressure",
                                description=(
                                    f"String concatenation at line {abs_line} inside a "
                                    f"loop allocates a new string each iteration. {fix}"
                                ),
                                file_path=context.file_path,
                                line_start=abs_line,
                                line_end=abs_line,
                                confidence=0.80,
                                tags=["performance", "gc", "memory", "string-concat"],
                            )
                        )
                        found = True

                # Autoboxing in loop (Java)
                if not found:
                    for pat in autobox_patterns:
                        if pat.search(line):
                            findings.append(
                                Finding(
                                    agent_name="gc_pressure",
                                    severity="medium",
                                    category="performance",
                                    title="Autoboxing in loop creates GC pressure",
                                    description=(
                                        f"Boxing operation at line {abs_line} inside a loop "
                                        f"creates wrapper objects per iteration. Use primitive "
                                        f"types or pre-allocate."
                                    ),
                                    file_path=context.file_path,
                                    line_start=abs_line,
                                    line_end=abs_line,
                                    confidence=0.82,
                                    tags=["performance", "gc", "memory", "autoboxing"],
                                )
                            )
                            break
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Excessive allocations inside loops create garbage "
            f"collection pressure, causing latency spikes and reduced throughput."
        )
