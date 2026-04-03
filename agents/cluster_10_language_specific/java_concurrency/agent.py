"""Java Concurrency Agent — detects common Java threading and synchronization hazards."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding

# ---------------------------------------------------------------------------
# Pattern 1: synchronized on a String (interned string locks are dangerous)
# ---------------------------------------------------------------------------
_SYNC_ON_STRING_RE = re.compile(
    r"synchronized\s*\(\s*"
    r"(?:"
    r'"[^"]*"'          # string literal
    r"|"
    r"\w+\.?\w*"        # variable name (we check type heuristically below)
    r")"
    r"\s*\)",
)

_SYNC_ON_STRING_LITERAL_RE = re.compile(
    r'synchronized\s*\(\s*"[^"]*"\s*\)',
)

_STRING_FIELD_RE = re.compile(
    r"(?:String|final\s+String)\s+(\w+)\s*[=;]",
)

# ---------------------------------------------------------------------------
# Pattern 2: double-checked locking without volatile
# ---------------------------------------------------------------------------
_DOUBLE_CHECKED_LOCK_RE = re.compile(
    r"if\s*\(\s*(\w+)\s*==\s*null\s*\)\s*\{[^}]*"
    r"synchronized\s*\([^)]*\)\s*\{[^}]*"
    r"if\s*\(\s*\1\s*==\s*null\s*\)",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Pattern 3: volatile field with ++ or += (not atomic)
# ---------------------------------------------------------------------------
_VOLATILE_FIELD_RE = re.compile(
    r"volatile\s+\w+\s+(\w+)\s*[=;]",
)

_INCREMENT_RE_TEMPLATE = r"(?:\b{field}\s*\+\+|\+\+\s*\b{field}|\b{field}\s*\+=)"

# ---------------------------------------------------------------------------
# Pattern 4: synchronized method on a public class
# ---------------------------------------------------------------------------
_SYNC_METHOD_RE = re.compile(
    r"^\s*(?:public\s+)?synchronized\s+\w+",
    re.MULTILINE,
)

_PUBLIC_CLASS_RE = re.compile(
    r"public\s+class\s+\w+",
)

# ---------------------------------------------------------------------------
# Pattern 5: lock.lock() without try-finally { lock.unlock() }
# ---------------------------------------------------------------------------
_LOCK_ACQUIRE_RE = re.compile(
    r"(\w+)\.lock\(\)",
)


def _find_volatile_increment(source: str, context: CodeContext) -> list[Finding]:
    """Detect volatile fields that are incremented non-atomically."""
    findings: list[Finding] = []
    volatile_fields: list[tuple[str, int]] = []

    for i, line in enumerate(source.splitlines(), 1):
        m = _VOLATILE_FIELD_RE.search(line)
        if m:
            volatile_fields.append((m.group(1), i))

    for field, decl_line in volatile_fields:
        pattern = re.compile(_INCREMENT_RE_TEMPLATE.format(field=re.escape(field)))
        for i, line in enumerate(source.splitlines(), 1):
            if pattern.search(line):
                findings.append(
                    Finding(
                        agent_name="java_concurrency",
                        severity="high",
                        category="bug",
                        title=f"Non-atomic increment of volatile field '{field}'",
                        description=(
                            f"Volatile field '{field}' (declared at line {decl_line}) "
                            f"is incremented at line {i} with ++ or +=. volatile does "
                            f"not make compound operations atomic. Use AtomicInteger "
                            f"or AtomicLong instead."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.82,
                        tags=["java", "concurrency", "threading"],
                    )
                )
    return findings


def _find_synchronized_on_string(source: str, context: CodeContext) -> list[Finding]:
    """Detect synchronized blocks that lock on String objects."""
    findings: list[Finding] = []

    # Collect known String field names
    string_fields: set[str] = set()
    for m in _STRING_FIELD_RE.finditer(source):
        string_fields.add(m.group(1))

    for m in _SYNC_ON_STRING_LITERAL_RE.finditer(source):
        line_num = source[: m.start()].count("\n") + 1
        findings.append(
            Finding(
                agent_name="java_concurrency",
                severity="critical",
                category="bug",
                title="Synchronized on string literal",
                description=(
                    f"synchronized block at line {line_num} locks on a string "
                    f"literal. Interned strings are shared across the JVM, so "
                    f"unrelated code may contend on the same lock. Use a "
                    f"private final Object instead."
                ),
                file_path=context.file_path,
                line_start=line_num,
                line_end=line_num,
                confidence=0.82,
                tags=["java", "concurrency", "threading"],
            )
        )

    for m in _SYNC_ON_STRING_RE.finditer(source):
        # Skip if already matched as literal
        text = m.group(0)
        if '"' in text:
            continue
        # Extract the variable name
        var_match = re.search(r"synchronized\s*\(\s*(\w+\.?\w*)\s*\)", text)
        if var_match:
            var = var_match.group(1)
            # Only flag if the variable was declared as String
            base_var = var.split(".")[0]
            if base_var in string_fields:
                line_num = source[: m.start()].count("\n") + 1
                findings.append(
                    Finding(
                        agent_name="java_concurrency",
                        severity="critical",
                        category="bug",
                        title=f"Synchronized on String variable '{var}'",
                        description=(
                            f"synchronized block at line {line_num} locks on "
                            f"String variable '{var}'. Interned strings are "
                            f"shared across the JVM, creating unpredictable "
                            f"lock contention. Use a private final Object instead."
                        ),
                        file_path=context.file_path,
                        line_start=line_num,
                        line_end=line_num,
                        confidence=0.82,
                        tags=["java", "concurrency", "threading"],
                    )
                )

    return findings


def _find_double_checked_locking(source: str, context: CodeContext) -> list[Finding]:
    """Detect double-checked locking without volatile."""
    findings: list[Finding] = []

    for m in _DOUBLE_CHECKED_LOCK_RE.finditer(source):
        field = m.group(1)
        # Check whether the field is declared volatile
        volatile_pat = re.compile(rf"volatile\s+\w+\s+{re.escape(field)}\b")
        if not volatile_pat.search(source):
            line_num = source[: m.start()].count("\n") + 1
            findings.append(
                Finding(
                    agent_name="java_concurrency",
                    severity="high",
                    category="bug",
                    title=f"Double-checked locking on non-volatile field '{field}'",
                    description=(
                        f"Double-checked locking at line {line_num} on field "
                        f"'{field}' without volatile. Without volatile, the JVM "
                        f"may reorder writes and a thread can observe a partially "
                        f"constructed object. Declare '{field}' as volatile."
                    ),
                    file_path=context.file_path,
                    line_start=line_num,
                    line_end=line_num,
                    confidence=0.82,
                    tags=["java", "concurrency", "threading"],
                )
            )

    return findings


def _find_synchronized_method(source: str, context: CodeContext) -> list[Finding]:
    """Detect synchronized methods on public classes."""
    findings: list[Finding] = []

    if not _PUBLIC_CLASS_RE.search(source):
        return findings

    for m in _SYNC_METHOD_RE.finditer(source):
        line_num = source[: m.start()].count("\n") + 1
        findings.append(
            Finding(
                agent_name="java_concurrency",
                severity="low",
                category="bug",
                title="Synchronized method on public class",
                description=(
                    f"Synchronized method at line {line_num} locks on 'this', "
                    f"which is visible to external code. Prefer a private lock "
                    f"object to prevent external code from interfering with "
                    f"synchronization."
                ),
                file_path=context.file_path,
                line_start=line_num,
                line_end=line_num,
                confidence=0.82,
                tags=["java", "concurrency", "threading"],
            )
        )

    return findings


def _find_lock_without_try_finally(source: str, context: CodeContext) -> list[Finding]:
    """Detect lock.lock() calls without corresponding try-finally unlock."""
    findings: list[Finding] = []
    lines = source.splitlines()

    for i, line in enumerate(lines, 1):
        lock_match = _LOCK_ACQUIRE_RE.search(line)
        if not lock_match:
            continue
        lock_var = lock_match.group(1)

        # Look ahead for a try block within the next few lines
        has_try_finally = False
        remaining = "\n".join(lines[i:])  # lines after lock()
        # Check if a try block follows reasonably soon (within 3 lines)
        next_lines = lines[i: i + 3]
        for next_line in next_lines:
            if "try" in next_line:
                # Now check if there's a finally block with unlock
                finally_pat = re.compile(
                    r"finally\s*\{[^}]*" + re.escape(lock_var) + r"\.unlock\(\)",
                    re.DOTALL,
                )
                if finally_pat.search(remaining):
                    has_try_finally = True
                break

        if not has_try_finally:
            findings.append(
                Finding(
                    agent_name="java_concurrency",
                    severity="medium",
                    category="bug",
                    title=f"Lock '{lock_var}' acquired without try-finally unlock",
                    description=(
                        f"'{lock_var}.lock()' at line {i} is not followed by a "
                        f"try-finally block that calls '{lock_var}.unlock()'. "
                        f"If an exception occurs, the lock will never be released, "
                        f"causing a deadlock."
                    ),
                    file_path=context.file_path,
                    line_start=i,
                    line_end=i,
                    confidence=0.82,
                    tags=["java", "concurrency", "threading"],
                )
            )

    return findings


class JavaConcurrencyAgent(BaseReviewAgent):
    """Detects common Java concurrency bugs via source-level regex patterns."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="java_concurrency",
            version="0.1.0",
            languages=["java"],
            domains=["systems_programming", "distributed_systems"],
            methodology="language_specific",
            axis_type="critical",
            tags=["java", "concurrency", "threading"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if context.language.lower() != "java":
            return []
        source = context.source_code
        if not source.strip():
            return []

        findings: list[Finding] = []
        findings.extend(_find_synchronized_on_string(source, context))
        findings.extend(_find_double_checked_locking(source, context))
        findings.extend(_find_volatile_increment(source, context))
        findings.extend(_find_synchronized_method(source, context))
        findings.extend(_find_lock_without_try_finally(source, context))
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Java concurrency bugs cause intermittent, "
            f"hard-to-reproduce failures in production. Use java.util.concurrent "
            f"utilities, volatile where required, and always release locks in "
            f"finally blocks."
        )
