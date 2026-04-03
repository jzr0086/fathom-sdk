"""Deserialization Vulnerability Detector — flags unsafe deserialization calls."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, get_node_text
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Python unsafe deserialization callees
# ---------------------------------------------------------------------------

_PYTHON_CRITICAL_CALLS: set[str] = {
    "loads",   # pickle.loads / cPickle.loads / marshal.loads
    "load",    # pickle.load / cPickle.load
    "open",    # shelve.open
}

_PYTHON_CRITICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"pickle\.loads?\s*\("),
    re.compile(r"cPickle\.loads?\s*\("),
    re.compile(r"marshal\.loads?\s*\("),
    re.compile(r"shelve\.open\s*\("),
]

# yaml.load() without SafeLoader
_YAML_LOAD_PATTERN = re.compile(r"yaml\.load\s*\(")
_YAML_SAFE_PATTERN = re.compile(r"(SafeLoader|safe_load|Loader\s*=\s*SafeLoader|Loader\s*=\s*yaml\.SafeLoader)")

# ---------------------------------------------------------------------------
# JavaScript / TypeScript unsafe deserialization callees
# ---------------------------------------------------------------------------

_JS_CRITICAL_CALLS: set[str] = {
    "eval",         # eval()
    "Function",     # Function() constructor
    "unserialize",  # node-serialize unserialize()
}

_JS_CRITICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\beval\s*\("),
    re.compile(r"\bFunction\s*\("),
    re.compile(r"\bunserialize\s*\("),
]

# ---------------------------------------------------------------------------
# Java unsafe deserialization callees
# ---------------------------------------------------------------------------

_JAVA_CRITICAL_CALLS: set[str] = {
    "readObject",    # ObjectInputStream.readObject()
    "readUnshared",  # ObjectInputStream.readUnshared()
    "XMLDecoder",    # new XMLDecoder(...)
}

_JAVA_CRITICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.readObject\s*\("),
    re.compile(r"\.readUnshared\s*\("),
    re.compile(r"new\s+XMLDecoder\s*\("),
]

# ---------------------------------------------------------------------------
# Go unsafe deserialization callees
# ---------------------------------------------------------------------------

_GO_CRITICAL_CALLS: set[str] = {
    "Decode",     # gob.Decode()
    "NewDecoder", # gob.NewDecoder()
}

_GO_CRITICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"gob\.Decode\s*\("),
    re.compile(r"gob\.NewDecoder\s*\("),
]

_GO_GOB_IMPORT = re.compile(r'"encoding/gob"')

# ---------------------------------------------------------------------------
# Language to pattern mapping
# ---------------------------------------------------------------------------

_LANG_CALL_MAP: dict[str, set[str]] = {
    "python": _PYTHON_CRITICAL_CALLS,
    "javascript": _JS_CRITICAL_CALLS,
    "typescript": _JS_CRITICAL_CALLS,
    "java": _JAVA_CRITICAL_CALLS,
    "go": _GO_CRITICAL_CALLS,
}

_LANG_SOURCE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": _PYTHON_CRITICAL_PATTERNS,
    "javascript": _JS_CRITICAL_PATTERNS,
    "typescript": _JS_CRITICAL_PATTERNS,
    "java": _JAVA_CRITICAL_PATTERNS,
    "go": _GO_CRITICAL_PATTERNS,
}


def _is_python_unsafe_deser(call_full_text: str, callee_name: str) -> bool:
    """Check whether a Python call is an unsafe deserialization call."""
    for pat in _PYTHON_CRITICAL_PATTERNS:
        if pat.search(call_full_text):
            return True
    return False


def _is_python_yaml_unsafe(call_full_text: str, callee_name: str) -> bool:
    """Check if a yaml.load() call is missing SafeLoader."""
    if callee_name != "load":
        return False
    if not _YAML_LOAD_PATTERN.search(call_full_text):
        return False
    # If SafeLoader or safe_load is referenced, it's safe
    if _YAML_SAFE_PATTERN.search(call_full_text):
        return False
    return True


class DeserializationAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="deserialization",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development"],
            methodology="security",
            axis_type="aware",
            tags=["security", "deserialization", "injection"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []

        lang = context.language.lower()
        if lang not in _LANG_CALL_MAP:
            return []

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []
        seen_lines: set[int] = set()
        call_names = _LANG_CALL_MAP[lang]

        # AST-based detection
        for call in find_calls(root, context.language):
            # --- Python: pickle, cPickle, marshal, shelve ---
            if lang == "python":
                if call.callee_name in call_names and _is_python_unsafe_deser(
                    call.full_text, call.callee_name
                ):
                    if call.start_line not in seen_lines:
                        seen_lines.add(call.start_line)
                        findings.append(
                            self._make_finding(
                                title="Unsafe deserialization: "
                                + call.full_text.split("\n")[0][:60],
                                description=(
                                    f"Unsafe deserialization at line {call.start_line}. "
                                    f"pickle, cPickle, marshal, and shelve can execute "
                                    f"arbitrary code during deserialization. Use safe "
                                    f"alternatives like json.loads()."
                                ),
                                severity="critical",
                                file_path=context.file_path,
                                line_start=call.start_line,
                                line_end=call.end_line,
                            )
                        )
                    continue

                # yaml.load() without SafeLoader
                if _is_python_yaml_unsafe(call.full_text, call.callee_name):
                    if call.start_line not in seen_lines:
                        seen_lines.add(call.start_line)
                        findings.append(
                            self._make_finding(
                                title="Unsafe YAML deserialization: yaml.load() without SafeLoader",
                                description=(
                                    f"yaml.load() at line {call.start_line} does not use "
                                    f"SafeLoader. This can execute arbitrary Python code. "
                                    f"Use yaml.safe_load() or pass Loader=SafeLoader."
                                ),
                                severity="high",
                                file_path=context.file_path,
                                line_start=call.start_line,
                                line_end=call.end_line,
                            )
                        )
                    continue

            # --- JavaScript / TypeScript: eval, Function, unserialize ---
            elif lang in ("javascript", "typescript"):
                if call.callee_name in call_names:
                    if call.start_line not in seen_lines:
                        seen_lines.add(call.start_line)
                        findings.append(
                            self._make_finding(
                                title=f"Unsafe deserialization: {call.callee_name}()",
                                description=(
                                    f"{call.callee_name}() at line {call.start_line} can "
                                    f"execute arbitrary code. Avoid eval(), Function() "
                                    f"constructor, and unserialize() with untrusted data."
                                ),
                                severity="critical",
                                file_path=context.file_path,
                                line_start=call.start_line,
                                line_end=call.end_line,
                            )
                        )

            # --- Java: readObject, readUnshared, XMLDecoder ---
            elif lang == "java":
                if call.callee_name in call_names:
                    if call.start_line not in seen_lines:
                        seen_lines.add(call.start_line)
                        findings.append(
                            self._make_finding(
                                title=f"Unsafe deserialization: {call.callee_name}()",
                                description=(
                                    f"Java deserialization via {call.callee_name}() at "
                                    f"line {call.start_line} can lead to remote code "
                                    f"execution. Use allowlists or safe serialization "
                                    f"formats like JSON."
                                ),
                                severity="critical",
                                file_path=context.file_path,
                                line_start=call.start_line,
                                line_end=call.end_line,
                            )
                        )

            # --- Go: gob.Decode, gob.NewDecoder ---
            elif lang == "go":
                if call.callee_name in call_names:
                    for pat in _GO_CRITICAL_PATTERNS:
                        if pat.search(call.full_text):
                            if call.start_line not in seen_lines:
                                seen_lines.add(call.start_line)
                                findings.append(
                                    self._make_finding(
                                        title=f"Unsafe deserialization: gob.{call.callee_name}()",
                                        description=(
                                            f"encoding/gob deserialization at line "
                                            f"{call.start_line} with untrusted input can "
                                            f"cause unexpected behavior. Validate and "
                                            f"sanitize input before decoding."
                                        ),
                                        severity="critical",
                                        file_path=context.file_path,
                                        line_start=call.start_line,
                                        line_end=call.end_line,
                                    )
                                )
                            break

        # Source-level fallback for patterns not caught by AST
        source_patterns = _LANG_SOURCE_PATTERNS.get(lang, [])
        lines = context.source_code.splitlines()
        for i, line in enumerate(lines, 1):
            if i in seen_lines:
                continue
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            for pat in source_patterns:
                if pat.search(line):
                    seen_lines.add(i)
                    findings.append(
                        self._make_finding(
                            title="Unsafe deserialization detected",
                            description=(
                                f"Line {i} contains an unsafe deserialization call: "
                                f"{stripped[:80]}. This may allow arbitrary code execution."
                            ),
                            severity="critical",
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                        )
                    )
                    break

            # Python yaml.load() fallback (source-level)
            if lang == "python" and i not in seen_lines:
                if _YAML_LOAD_PATTERN.search(line) and not _YAML_SAFE_PATTERN.search(line):
                    seen_lines.add(i)
                    findings.append(
                        self._make_finding(
                            title="Unsafe YAML deserialization: yaml.load() without SafeLoader",
                            description=(
                                f"yaml.load() at line {i} does not use SafeLoader. "
                                f"Use yaml.safe_load() or pass Loader=SafeLoader."
                            ),
                            severity="high",
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                        )
                    )

        return findings

    def _make_finding(
        self,
        *,
        title: str,
        description: str,
        severity: str,
        file_path: str,
        line_start: int,
        line_end: int,
    ) -> Finding:
        return Finding(
            agent_name="deserialization",
            severity=severity,
            category="security",
            title=title,
            description=description,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            confidence=0.90,
            tags=["security", "deserialization", "injection"],
        )

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Unsafe deserialization allows attackers to execute "
            f"arbitrary code by crafting malicious serialized payloads. Always use "
            f"safe alternatives: json.loads() instead of pickle, yaml.safe_load() "
            f"instead of yaml.load(), and avoid eval()/Function() with untrusted data."
        )
