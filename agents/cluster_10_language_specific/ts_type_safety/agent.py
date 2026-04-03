"""TypeScript Type Safety Deep Agent — flags unsafe type patterns."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_parser import parse

# Pattern: `: any` type annotation (colon followed by any, not inside a word)
_ANY_ANNOTATION = re.compile(r":\s*any\b")

# Pattern: `as any` type cast
_AS_ANY_CAST = re.compile(r"\bas\s+any\b")

# Pattern: @ts-ignore or @ts-nocheck directives
_TS_DIRECTIVE = re.compile(r"@ts-ignore|@ts-nocheck")

# Pattern: non-null assertion operator (variable!. or variable! at word boundary)
# Matches identifier followed by ! then . or end-of-expression context
_NON_NULL_ASSERTION = re.compile(r"[a-zA-Z_$]\w*\s*!\s*(?:\.|;|\)|\]|,|\s|$)")

# Pattern: type assertion chain like `(x as Type1) as Type2`
_TYPE_ASSERTION_CHAIN = re.compile(
    r"\bas\s+[A-Za-z_$][\w.<>,\s|&]*\)\s*as\s+[A-Za-z_$][\w.<>,\s|&]*"
)

_ANY_ANNOTATION_THRESHOLD = 3
_NON_NULL_ASSERTION_THRESHOLD = 5


class TsTypeSafetyAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="ts_type_safety",
            version="0.1.0",
            languages=["typescript"],
            domains=["web_development", "api_integration"],
            methodology="language_specific",
            axis_type="critical",
            tags=["typescript", "type-safety", "any"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if context.language.lower() != "typescript":
            return []
        if not context.source_code.strip():
            return []

        findings: list[Finding] = []
        lines = context.source_code.splitlines()

        # --- 1. `: any` annotation abuse (threshold-based) ---
        any_annotation_count = 0
        any_annotation_first_line = 0
        any_annotation_last_line = 0
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*")):
                continue
            matches = _ANY_ANNOTATION.findall(line)
            if matches:
                any_annotation_count += len(matches)
                if any_annotation_first_line == 0:
                    any_annotation_first_line = i
                any_annotation_last_line = i

        if any_annotation_count > _ANY_ANNOTATION_THRESHOLD:
            findings.append(
                Finding(
                    agent_name="ts_type_safety",
                    severity="medium",
                    category="quality",
                    title=f"Excessive `any` type annotations ({any_annotation_count} occurrences)",
                    description=(
                        f"File contains {any_annotation_count} `: any` type annotations "
                        f"(threshold: {_ANY_ANNOTATION_THRESHOLD}). Overuse of `any` defeats "
                        f"TypeScript's type system and hides potential bugs. Use specific "
                        f"types, `unknown`, or generics instead."
                    ),
                    file_path=context.file_path,
                    line_start=any_annotation_first_line,
                    line_end=any_annotation_last_line,
                    confidence=0.88,
                    tags=["typescript", "type-safety", "any"],
                )
            )

        # --- 2. `as any` casts (flag each one) ---
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*")):
                continue
            if _AS_ANY_CAST.search(line):
                findings.append(
                    Finding(
                        agent_name="ts_type_safety",
                        severity="high",
                        category="quality",
                        title="`as any` type cast",
                        description=(
                            f"Line {i}: `as any` casts bypass TypeScript's type checker "
                            f"entirely. Use type guards, `unknown`, or proper type narrowing "
                            f"instead."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.88,
                        tags=["typescript", "type-safety", "any"],
                    )
                )

        # --- 3. @ts-ignore / @ts-nocheck directives (flag each one) ---
        for i, line in enumerate(lines, 1):
            if _TS_DIRECTIVE.search(line):
                directive = (
                    "@ts-nocheck" if "@ts-nocheck" in line else "@ts-ignore"
                )
                findings.append(
                    Finding(
                        agent_name="ts_type_safety",
                        severity="medium",
                        category="quality",
                        title=f"`{directive}` directive suppresses type checking",
                        description=(
                            f"Line {i}: `{directive}` silences the TypeScript compiler. "
                            f"Fix the underlying type error instead of suppressing it. "
                            f"Use `@ts-expect-error` with a description if suppression "
                            f"is truly needed."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.88,
                        tags=["typescript", "type-safety", "any"],
                    )
                )

        # --- 4. Non-null assertion `!` overuse (threshold-based) ---
        non_null_lines: list[int] = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*")):
                continue
            if _NON_NULL_ASSERTION.search(line):
                non_null_lines.append(i)

        if len(non_null_lines) > _NON_NULL_ASSERTION_THRESHOLD:
            findings.append(
                Finding(
                    agent_name="ts_type_safety",
                    severity="low",
                    category="quality",
                    title=f"Excessive non-null assertions ({len(non_null_lines)} occurrences)",
                    description=(
                        f"File contains {len(non_null_lines)} non-null assertion operators `!` "
                        f"(threshold: {_NON_NULL_ASSERTION_THRESHOLD}). Overuse of `!` "
                        f"bypasses null checks and can lead to runtime errors. Use optional "
                        f"chaining (`?.`), nullish coalescing (`??`), or proper null guards."
                    ),
                    file_path=context.file_path,
                    line_start=non_null_lines[0],
                    line_end=non_null_lines[-1],
                    confidence=0.88,
                    tags=["typescript", "type-safety", "any"],
                )
            )

        # --- 5. Type assertion chains: `(x as Type1) as Type2` ---
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("//", "/*", "*")):
                continue
            if _TYPE_ASSERTION_CHAIN.search(line):
                findings.append(
                    Finding(
                        agent_name="ts_type_safety",
                        severity="high",
                        category="quality",
                        title="Type assertion chain detected",
                        description=(
                            f"Line {i}: chaining multiple `as` type assertions "
                            f"(e.g. `(x as Type1) as Type2`) is a strong code smell "
                            f"indicating a type design problem. Refactor the types or "
                            f"use a type guard function."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.88,
                        tags=["typescript", "type-safety", "any"],
                    )
                )

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. TypeScript's type system exists to catch bugs at "
            f"compile time. Bypassing it with `any`, `as any`, `!`, or `@ts-ignore` "
            f"defeats that purpose and pushes errors to runtime where they are harder "
            f"to diagnose."
        )
