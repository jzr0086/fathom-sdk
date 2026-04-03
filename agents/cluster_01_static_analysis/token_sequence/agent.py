"""Token Sequence Classifier — sliding-window pattern matching over source lines.

Detects suspicious n-gram patterns such as repeated identical statements,
self-assignments, duplicate dictionary keys, unreachable code after return,
and comparisons of identical operands.
"""

from __future__ import annotations

import re
from typing import Optional

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Self-assignment: ``x = x`` (with optional whitespace / semicolons)
_SELF_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*=\s*\1\s*;?\s*$"
)

# Comparison of identical operands: ``x == x``, ``x != x``, ``x > x``, etc.
_SELF_COMPARE_RE = re.compile(
    r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*(==|!=|>=|<=|>(?!=)|<(?!=))\s*\1(?!\w)"
)

# Duplicate dictionary / object keys (Python ``{"a": 1, "a": 2}`` style)
_DICT_KEY_RE = re.compile(r"""(?:"|')([^"']+)(?:"|')\s*:""")

# Return/throw followed by code on the very next non-blank line
_RETURN_LINE_RE = re.compile(
    r"^\s*(return\b|throw\b|raise\b)", re.IGNORECASE
)

# Catch / except block opening (used for repeated-catch detection)
_CATCH_OPEN_RE = re.compile(
    r"^\s*(except\b.*:|catch\s*\()"
)


def _strip_comment(line: str) -> str:
    """Very rough comment removal (single-line only)."""
    # Remove # comments (Python) but not inside strings — good enough heuristic
    in_str: Optional[str] = None
    out: list[str] = []
    for ch in line:
        if in_str:
            out.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        if ch == "/" and out and out[-1] == "/":
            out.pop()
            break
        out.append(ch)
    return "".join(out).rstrip()


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _detect_consecutive_duplicates(lines: list[str]) -> list[Finding]:
    """Flag consecutive identical non-trivial lines (copy-paste smell)."""
    findings: list[Finding] = []
    prev = ""
    prev_lineno = 0
    for lineno, raw in enumerate(lines, start=1):
        stripped = _strip_comment(raw).strip()
        if not stripped or stripped in ("{", "}", "pass", ""):
            prev = ""
            continue
        if stripped == prev:
            findings.append(
                _make_finding(
                    title="Consecutive duplicate statement",
                    description=(
                        f"Line {lineno} is identical to line {prev_lineno}: "
                        f"`{stripped[:80]}`"
                    ),
                    line=lineno,
                    severity="low",
                )
            )
        prev = stripped
        prev_lineno = lineno
    return findings


def _detect_self_assignment(lines: list[str]) -> list[Finding]:
    """Flag ``x = x`` patterns."""
    findings: list[Finding] = []
    for lineno, raw in enumerate(lines, start=1):
        stripped = _strip_comment(raw)
        if _SELF_ASSIGN_RE.match(stripped):
            name = _SELF_ASSIGN_RE.match(stripped).group(1)  # type: ignore[union-attr]
            findings.append(
                _make_finding(
                    title="Self-assignment has no effect",
                    description=f"`{name}` is assigned to itself at line {lineno}.",
                    line=lineno,
                    severity="low",
                )
            )
    return findings


def _detect_self_comparison(lines: list[str]) -> list[Finding]:
    """Flag ``x == x``, ``x != x``, ``x > x``, etc."""
    findings: list[Finding] = []
    for lineno, raw in enumerate(lines, start=1):
        stripped = _strip_comment(raw)
        m = _SELF_COMPARE_RE.search(stripped)
        if m:
            var = m.group(1)
            op = m.group(2)
            findings.append(
                _make_finding(
                    title="Comparison of identical operands",
                    description=(
                        f"`{var} {op} {var}` at line {lineno} is always "
                        f"{'true' if op == '==' else 'false' if op == '!=' else 'trivially constant'}."
                    ),
                    line=lineno,
                    severity="low",
                )
            )
    return findings


def _detect_return_followed_by_code(lines: list[str]) -> list[Finding]:
    """Flag a return/raise/throw with code on the very next non-blank line."""
    findings: list[Finding] = []
    return_lineno: Optional[int] = None
    for lineno, raw in enumerate(lines, start=1):
        stripped = _strip_comment(raw).strip()
        if not stripped:
            continue
        if return_lineno is not None:
            # Next non-blank line after a return — is it code?
            # Skip closing braces, except/catch, else, finally
            if stripped not in ("}", ")", "]") and not re.match(
                r"^\s*(except\b|catch\b|else\b|elif\b|finally\b|case\b|\})", stripped
            ):
                findings.append(
                    _make_finding(
                        title="Unreachable code after return/raise/throw",
                        description=(
                            f"Line {lineno} appears unreachable — it follows a "
                            f"return/raise/throw at line {return_lineno}."
                        ),
                        line=lineno,
                        severity="medium",
                    )
                )
            return_lineno = None
        if _RETURN_LINE_RE.match(stripped):
            return_lineno = lineno
    return findings


def _detect_duplicate_dict_keys(lines: list[str]) -> list[Finding]:
    """Flag duplicate dictionary/object literal keys within a proximity window."""
    findings: list[Finding] = []
    # Collect all keys with their line numbers
    keys: dict[str, list[int]] = {}
    brace_depth = 0
    scope_keys: list[dict[str, int]] = [{}]

    for lineno, raw in enumerate(lines, start=1):
        stripped = _strip_comment(raw)
        # Track brace depth for scope awareness
        for ch in stripped:
            if ch == "{":
                brace_depth += 1
                scope_keys.append({})
            elif ch == "}":
                brace_depth = max(0, brace_depth - 1)
                if len(scope_keys) > 1:
                    scope_keys.pop()

        if brace_depth > 0 or "{" in stripped:
            for m in _DICT_KEY_RE.finditer(stripped):
                key = m.group(1)
                current_scope = scope_keys[-1] if scope_keys else {}
                if key in current_scope:
                    findings.append(
                        _make_finding(
                            title="Duplicate dictionary/object key",
                            description=(
                                f"Key `{key}` at line {lineno} duplicates an "
                                f"earlier occurrence at line {current_scope[key]}."
                            ),
                            line=lineno,
                            severity="low",
                        )
                    )
                else:
                    current_scope[key] = lineno

    return findings


def _detect_repeated_catch_blocks(lines: list[str]) -> list[Finding]:
    """Flag consecutive catch/except blocks with identical bodies."""
    findings: list[Finding] = []
    # Collect catch blocks as (header_line, body_text)
    blocks: list[tuple[int, str]] = []
    in_catch = False
    catch_start = 0
    body_lines: list[str] = []

    for lineno, raw in enumerate(lines, start=1):
        stripped = _strip_comment(raw).strip()
        if _CATCH_OPEN_RE.match(stripped):
            if in_catch and body_lines:
                blocks.append((catch_start, "\n".join(body_lines).strip()))
            in_catch = True
            catch_start = lineno
            body_lines = []
        elif in_catch:
            # Detect end of catch body (rough: de-indent or new keyword)
            if stripped and not stripped.startswith(("#", "//", "/*")):
                body_lines.append(stripped)

    if in_catch and body_lines:
        blocks.append((catch_start, "\n".join(body_lines).strip()))

    # Check for consecutive identical bodies
    for i in range(1, len(blocks)):
        if blocks[i][1] and blocks[i][1] == blocks[i - 1][1]:
            findings.append(
                _make_finding(
                    title="Repeated identical catch/except block",
                    description=(
                        f"Catch block at line {blocks[i][0]} has the same body as "
                        f"the catch block at line {blocks[i - 1][0]} — possible "
                        f"copy-paste error."
                    ),
                    line=blocks[i][0],
                    severity="low",
                )
            )

    return findings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FILE_PATH_PLACEHOLDER = "<set-by-analyze>"


def _make_finding(
    *,
    title: str,
    description: str,
    line: int,
    severity: str,
) -> Finding:
    """Build a Finding with common defaults; ``file_path`` is patched later."""
    return Finding(
        agent_name="token_sequence",
        severity=severity,
        category="quality",
        title=title,
        description=description,
        file_path=_FILE_PATH_PLACEHOLDER,
        line_start=line,
        line_end=line,
        confidence=0.85,
        tags=["static-analysis", "token", "pattern"],
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class TokenSequenceAgent(BaseReviewAgent):
    """Sliding-window token / line pattern matcher.

    Uses ``context.token_stream`` when available; otherwise falls back to
    regex-based source-line scanning.
    """

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="token_sequence",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="static_analysis",
            axis_type="aware",
            tags=["static-analysis", "token", "pattern"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []

        lines = context.source_code.splitlines()

        findings: list[Finding] = []
        findings.extend(_detect_consecutive_duplicates(lines))
        findings.extend(_detect_self_assignment(lines))
        findings.extend(_detect_self_comparison(lines))
        findings.extend(_detect_return_followed_by_code(lines))
        findings.extend(_detect_duplicate_dict_keys(lines))
        findings.extend(_detect_repeated_catch_blocks(lines))

        # Patch file_path on every finding
        for f in findings:
            f.file_path = context.file_path

        return findings

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. {finding.description} "
            f"This pattern is usually a copy-paste error or logic mistake — "
            f"review the code and remove or fix the duplication."
        )
