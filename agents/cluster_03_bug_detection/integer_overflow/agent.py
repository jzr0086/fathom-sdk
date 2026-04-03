"""Integer Overflow Detector — flags unchecked arithmetic on parsed integers."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, get_node_text
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Conversion calls that produce fixed-width integers per language
# ---------------------------------------------------------------------------

_INT_CONVERSION_CALLS: dict[str, set[str]] = {
    "python": {"int"},
    "javascript": {"parseInt", "parseFloat", "Number"},
    "typescript": {"parseInt", "parseFloat", "Number"},
    "java": {"parseInt", "parseLong", "parseShort", "valueOf", "intValue"},
    "go": {"Atoi", "ParseInt", "ParseUint"},
}

# ---------------------------------------------------------------------------
# Source-level regex patterns
# ---------------------------------------------------------------------------

# Pattern 1: Arithmetic on values from user/external input
_USER_INPUT_ARITH: list[tuple[re.Pattern[str], str, str]] = [
    # Python: int(request.args.get(...)) * something
    (
        re.compile(
            r"int\s*\(\s*\w+\.\w+\.get\s*\([^)]*\)\s*\)\s*[*+\-]"
        ),
        "Arithmetic on user input without bounds check",
        "int() conversion from external input used directly in arithmetic. "
        "Validate and bound the value before performing calculations.",
    ),
    # Python: int(input(...)) in arithmetic
    (
        re.compile(r"int\s*\(\s*input\s*\([^)]*\)\s*\)\s*[*+\-]"),
        "Arithmetic on user input without bounds check",
        "int() conversion from input() used directly in arithmetic. "
        "Validate and bound the value before performing calculations.",
    ),
    # JS/TS: parseInt(req.params...) or parseInt(req.query...) in arithmetic
    (
        re.compile(
            r"parseInt\s*\(\s*req\.\w+[\[.][^)]*\)\s*[*+\-]"
        ),
        "Arithmetic on user input without bounds check",
        "parseInt() from request parameter used directly in arithmetic. "
        "Validate the value is within expected range before calculations.",
    ),
    # JS/TS: Number(req.body...) in arithmetic
    (
        re.compile(
            r"Number\s*\(\s*req\.\w+[\[.][^)]*\)\s*[*+\-]"
        ),
        "Arithmetic on user input without bounds check",
        "Number() from request parameter used directly in arithmetic. "
        "Check for Number.MAX_SAFE_INTEGER before calculations.",
    ),
]

# Pattern 2: Java/Go MAX_VALUE proximity patterns
_MAX_VALUE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"Integer\.MAX_VALUE\s*[+*]"),
        "Arithmetic near Integer.MAX_VALUE",
        "Adding to or multiplying Integer.MAX_VALUE will overflow silently in Java.",
    ),
    (
        re.compile(r"Long\.MAX_VALUE\s*[+*]"),
        "Arithmetic near Long.MAX_VALUE",
        "Adding to or multiplying Long.MAX_VALUE will overflow silently in Java.",
    ),
    (
        re.compile(r"[+*]\s*Integer\.MAX_VALUE"),
        "Arithmetic near Integer.MAX_VALUE",
        "Arithmetic involving Integer.MAX_VALUE will overflow silently in Java.",
    ),
    (
        re.compile(r"[+*]\s*Long\.MAX_VALUE"),
        "Arithmetic near Long.MAX_VALUE",
        "Arithmetic involving Long.MAX_VALUE will overflow silently in Java.",
    ),
    (
        re.compile(r"math\.MaxInt(?:32|64)?\s*[+*]"),
        "Arithmetic near math.MaxInt",
        "Adding to or multiplying math.MaxInt will wrap around silently in Go.",
    ),
    (
        re.compile(r"[+*]\s*math\.MaxInt(?:32|64)?"),
        "Arithmetic near math.MaxInt",
        "Arithmetic involving math.MaxInt will wrap around silently in Go.",
    ),
]

# Pattern 3: Unchecked int conversion used in arithmetic
_UNCHECKED_PARSE_ARITH: list[tuple[re.Pattern[str], str, str]] = [
    # Java: Integer.parseInt(...) * / + - something
    (
        re.compile(
            r"Integer\.parseInt\s*\([^)]+\)\s*[*+\-/]\s*\w"
        ),
        "Unchecked parseInt in arithmetic",
        "Integer.parseInt() result used directly in arithmetic. "
        "Java int arithmetic overflows silently — validate range first.",
    ),
    (
        re.compile(
            r"\w\s*[*+\-/]\s*Integer\.parseInt\s*\([^)]+\)"
        ),
        "Unchecked parseInt in arithmetic",
        "Integer.parseInt() result used directly in arithmetic. "
        "Java int arithmetic overflows silently — validate range first.",
    ),
    # Go: strconv.Atoi used in arithmetic
    (
        re.compile(
            r"strconv\.Atoi\s*\([^)]+\)"
        ),
        "Unchecked Atoi result in arithmetic",
        "strconv.Atoi() result may be used in arithmetic without range validation. "
        "Go integer arithmetic wraps silently on overflow.",
    ),
    # JS/TS: parseInt used in multiplication
    (
        re.compile(
            r"parseInt\s*\([^)]+\)\s*\*\s*\w"
        ),
        "Unchecked parseInt in multiplication",
        "parseInt() result used in multiplication. Check for "
        "Number.MAX_SAFE_INTEGER to avoid precision loss.",
    ),
    (
        re.compile(
            r"\w\s*\*\s*parseInt\s*\([^)]+\)"
        ),
        "Unchecked parseInt in multiplication",
        "parseInt() result used in multiplication. Check for "
        "Number.MAX_SAFE_INTEGER to avoid precision loss.",
    ),
]

# Pattern 4: Large multiplications without overflow guards (x * y * z)
_LARGE_MULTIPLY = re.compile(
    r"\b\w+\s*\*\s*\w+\s*\*\s*\w+"
)

# Bounds-check / overflow-guard indicators on the same line or nearby
_OVERFLOW_GUARD_INDICATORS = re.compile(
    r"(MAX_VALUE|MAX_SAFE_INTEGER|MaxInt|BigInteger|BigDecimal|"
    r"math/big|checked|overflow|bounds|clamp|safemath|"
    r"Math\.addExact|Math\.multiplyExact|"
    r"if\s+.*[<>]=?\s*|"
    r"assert\s+|"
    r"try\s*[:{(])",
    re.IGNORECASE,
)

# Things that make a triple multiplication benign
_BENIGN_MULTIPLY = re.compile(
    r"^\s*(?:#|//|/\*|\*)"  # comment line
)


class IntegerOverflowAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="integer_overflow",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["systems_programming", "web_development"],
            methodology="bug_detection",
            axis_type="aware",
            tags=["bug", "overflow", "integer"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        if lang not in ("python", "javascript", "typescript", "java", "go"):
            return []

        findings: list[Finding] = []
        seen_lines: set[int] = set()
        lines = context.source_code.splitlines()

        # --- AST-based detection via find_calls ---
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                root = None

        if root is not None:
            conversion_calls = _INT_CONVERSION_CALLS.get(lang, set())
            for call in find_calls(root, context.language):
                if call.callee_name not in conversion_calls:
                    continue
                # Check if the call text is embedded in an arithmetic expression
                call_text = get_node_text(call.node)
                parent = call.node.parent
                if parent is not None:
                    parent_text = get_node_text(parent)
                    # Check if the parent is a binary expression with arithmetic
                    if parent.type in (
                        "binary_expression",
                        "binary_operator",
                        "augmented_assignment",
                    ):
                        line_text = lines[call.start_line - 1] if call.start_line <= len(lines) else ""
                        if not _OVERFLOW_GUARD_INDICATORS.search(line_text):
                            if call.start_line not in seen_lines:
                                seen_lines.add(call.start_line)
                                findings.append(
                                    Finding(
                                        agent_name="integer_overflow",
                                        severity="medium",
                                        category="bug",
                                        title=(
                                            f"Integer conversion '{call.callee_name}()' "
                                            f"in arithmetic without overflow check"
                                        ),
                                        description=(
                                            f"'{call.callee_name}()' at line {call.start_line} "
                                            f"is used in arithmetic without bounds validation. "
                                            f"This may lead to integer overflow or precision loss."
                                        ),
                                        file_path=context.file_path,
                                        line_start=call.start_line,
                                        line_end=call.end_line,
                                        confidence=0.72,
                                        tags=["bug", "overflow", "integer"],
                                    )
                                )

        # --- Source-level regex patterns ---

        for i, line in enumerate(lines, 1):
            if i in seen_lines:
                continue
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue

            # Pattern 1: Arithmetic on user input
            for pat, title, desc in _USER_INPUT_ARITH:
                if pat.search(line):
                    if not _OVERFLOW_GUARD_INDICATORS.search(line):
                        if i not in seen_lines:
                            seen_lines.add(i)
                            findings.append(
                                Finding(
                                    agent_name="integer_overflow",
                                    severity="medium",
                                    category="bug",
                                    title=title,
                                    description=f"Line {i}: {desc}",
                                    file_path=context.file_path,
                                    line_start=i,
                                    line_end=i,
                                    confidence=0.72,
                                    tags=["bug", "overflow", "integer"],
                                )
                            )
                            break

            if i in seen_lines:
                continue

            # Pattern 2: MAX_VALUE proximity
            for pat, title, desc in _MAX_VALUE_PATTERNS:
                if pat.search(line):
                    if i not in seen_lines:
                        seen_lines.add(i)
                        findings.append(
                            Finding(
                                agent_name="integer_overflow",
                                severity="medium",
                                category="bug",
                                title=title,
                                description=f"Line {i}: {desc}",
                                file_path=context.file_path,
                                line_start=i,
                                line_end=i,
                                confidence=0.72,
                                tags=["bug", "overflow", "integer"],
                            )
                        )
                        break

            if i in seen_lines:
                continue

            # Pattern 3: Unchecked conversion in arithmetic
            # Skip Python for unchecked parse patterns (arbitrary precision)
            if lang != "python":
                for pat, title, desc in _UNCHECKED_PARSE_ARITH:
                    if pat.search(line):
                        if not _OVERFLOW_GUARD_INDICATORS.search(line):
                            if i not in seen_lines:
                                seen_lines.add(i)
                                findings.append(
                                    Finding(
                                        agent_name="integer_overflow",
                                        severity="medium",
                                        category="bug",
                                        title=title,
                                        description=f"Line {i}: {desc}",
                                        file_path=context.file_path,
                                        line_start=i,
                                        line_end=i,
                                        confidence=0.72,
                                        tags=["bug", "overflow", "integer"],
                                    )
                                )
                                break

            if i in seen_lines:
                continue

            # Pattern 4: Large multiplications (x * y * z) in Java/Go
            if lang in ("java", "go"):
                if _LARGE_MULTIPLY.search(stripped):
                    if _BENIGN_MULTIPLY.match(stripped):
                        continue
                    if not _OVERFLOW_GUARD_INDICATORS.search(line):
                        # Only flag if there's no constant-only multiplication
                        # (e.g. 2 * 3 * 4 is fine)
                        match = _LARGE_MULTIPLY.search(stripped)
                        if match:
                            tokens = re.findall(r"\b\w+\b", match.group())
                            all_numeric = all(
                                re.match(r"^\d+$", t) for t in tokens
                            )
                            if not all_numeric:
                                seen_lines.add(i)
                                findings.append(
                                    Finding(
                                        agent_name="integer_overflow",
                                        severity="medium",
                                        category="bug",
                                        title="Large multiplication chain without overflow guard",
                                        description=(
                                            f"Line {i}: Chained multiplication (x * y * z) "
                                            f"in {lang.title()} may overflow silently. Consider "
                                            f"using Math.multiplyExact (Java) or math/big (Go)."
                                        ),
                                        file_path=context.file_path,
                                        line_start=i,
                                        line_end=i,
                                        confidence=0.72,
                                        tags=["bug", "overflow", "integer"],
                                    )
                                )

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Integer overflow causes silent wraparound in Java and Go, "
            f"and precision loss in JavaScript. Always validate ranges before performing "
            f"arithmetic on externally-provided or parsed integer values."
        )
