"""Go Goroutine Safety Agent — detects common goroutine concurrency pitfalls.

Checks for:
1. Goroutine leaks (go func without WaitGroup or context cancellation)
2. Channel deadlocks (unbuffered channel with send/receive in same goroutine)
3. Lock without defer Unlock
4. WaitGroup misuse (wg.Add inside goroutine instead of before it)
"""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import get_node_text, walk
from fathom_sdk.context.ast_parser import parse


def _detect_goroutine_leak(root, context: CodeContext) -> list[Finding]:
    """Detect goroutines launched without WaitGroup or context cancellation.

    Heuristic: a ``go`` statement whose enclosing function body does not
    contain ``wg.Add``, ``wg.Wait``, or ``ctx`` usage.
    """
    findings: list[Finding] = []

    for node in walk(root):
        if node.type != "go_statement":
            continue

        go_line = node.start_point[0] + 1
        go_text = get_node_text(node)

        # Walk up to find the enclosing function body
        enclosing = node.parent
        while enclosing is not None and enclosing.type not in (
            "function_declaration",
            "method_declaration",
            "func_literal",
        ):
            enclosing = enclosing.parent

        if enclosing is None:
            # Top-level go statement with no enclosing function — flag it
            findings.append(
                Finding(
                    agent_name="go_goroutine_safety",
                    severity="high",
                    category="bug",
                    title="Goroutine launched without WaitGroup or context",
                    description=(
                        f"Goroutine at line {go_line} is launched without "
                        f"a WaitGroup or context for lifecycle management. "
                        f"This may cause a goroutine leak."
                    ),
                    file_path=context.file_path,
                    line_start=go_line,
                    line_end=node.end_point[0] + 1,
                    confidence=0.78,
                    tags=["go", "goroutine", "concurrency", "leak"],
                )
            )
            continue

        enclosing_text = get_node_text(enclosing)

        has_waitgroup = "wg.Add" in enclosing_text or "wg.Wait" in enclosing_text
        has_context = "ctx" in enclosing_text

        if not has_waitgroup and not has_context:
            findings.append(
                Finding(
                    agent_name="go_goroutine_safety",
                    severity="high",
                    category="bug",
                    title="Goroutine launched without WaitGroup or context",
                    description=(
                        f"Goroutine at line {go_line} is launched without "
                        f"a WaitGroup or context for lifecycle management. "
                        f"This may cause a goroutine leak."
                    ),
                    file_path=context.file_path,
                    line_start=go_line,
                    line_end=node.end_point[0] + 1,
                    confidence=0.78,
                    tags=["go", "goroutine", "concurrency", "leak"],
                )
            )

    return findings


def _detect_channel_deadlock(root, context: CodeContext) -> list[Finding]:
    """Detect unbuffered channels that are sent to and received from in the same goroutine.

    Heuristic: ``make(chan ...)`` without a buffer size, where the enclosing
    function contains both ``<-ch`` and ``ch <-`` patterns but no ``go``
    statement that uses the channel.
    """
    findings: list[Finding] = []
    source = context.source_code
    lines = source.splitlines()

    # Find unbuffered channel declarations via regex
    chan_pattern = re.compile(r"(\w+)\s*:?=\s*make\(\s*chan\s+\w+\s*\)")
    for match in chan_pattern.finditer(source):
        ch_name = match.group(1)
        decl_line = source[: match.start()].count("\n") + 1

        # Find the enclosing function boundaries using simple brace counting
        func_start = None
        func_end = None
        brace_depth = 0
        for i, line in enumerate(lines):
            if re.search(r"\bfunc\b", line):
                func_start = i
                brace_depth = 0
            if func_start is not None:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0 and i > func_start:
                    func_end = i
                    break

        if func_start is None or func_end is None:
            continue

        func_text = "\n".join(lines[func_start : func_end + 1])

        # Check for send and receive on same channel
        send_pattern = re.compile(rf"\b{re.escape(ch_name)}\s*<-")
        recv_pattern = re.compile(rf"<-\s*{re.escape(ch_name)}\b")
        has_send = send_pattern.search(func_text) is not None
        has_recv = recv_pattern.search(func_text) is not None

        # Check if there's a goroutine that uses this channel
        has_goroutine_with_chan = False
        go_pattern = re.compile(r"\bgo\s+func\s*\(")
        for go_match in go_pattern.finditer(func_text):
            # Find the goroutine body (approximate: next brace block)
            go_start = go_match.start()
            go_brace = 0
            go_body = ""
            for ci, ch in enumerate(func_text[go_start:]):
                if ch == "{":
                    go_brace += 1
                elif ch == "}":
                    go_brace -= 1
                    if go_brace == 0:
                        go_body = func_text[go_start : go_start + ci + 1]
                        break
            if ch_name in go_body:
                has_goroutine_with_chan = True
                break

        # Only flag if both send and receive exist but no goroutine uses the channel
        if has_send and has_recv and not has_goroutine_with_chan:
            findings.append(
                Finding(
                    agent_name="go_goroutine_safety",
                    severity="high",
                    category="bug",
                    title=f"Potential channel deadlock on '{ch_name}'",
                    description=(
                        f"Unbuffered channel '{ch_name}' at line {decl_line} is "
                        f"both sent to and received from in the same goroutine "
                        f"without a separate goroutine handling the other end. "
                        f"This will deadlock."
                    ),
                    file_path=context.file_path,
                    line_start=decl_line,
                    line_end=decl_line,
                    confidence=0.78,
                    tags=["go", "goroutine", "concurrency", "deadlock"],
                )
            )

    return findings


def _detect_lock_without_defer_unlock(root, context: CodeContext) -> list[Finding]:
    """Detect mu.Lock() calls without a corresponding defer mu.Unlock().

    Scans source lines for ``.Lock()`` calls and checks whether a
    ``defer ...Unlock()`` appears within the next few lines.
    """
    findings: list[Finding] = []
    source = context.source_code
    lines = source.splitlines()

    lock_pattern = re.compile(r"(\w+)\.Lock\(\)")
    for i, line in enumerate(lines):
        match = lock_pattern.search(line)
        if match is None:
            continue

        var_name = match.group(1)
        line_num = i + 1

        # Look at the next 3 lines for defer Unlock
        window = lines[i : i + 4]
        window_text = "\n".join(window)

        defer_unlock = re.compile(
            rf"defer\s+{re.escape(var_name)}\.Unlock\(\)"
        )
        if not defer_unlock.search(window_text):
            findings.append(
                Finding(
                    agent_name="go_goroutine_safety",
                    severity="medium",
                    category="bug",
                    title=f"Lock without defer Unlock on '{var_name}'",
                    description=(
                        f"'{var_name}.Lock()' at line {line_num} is not "
                        f"followed by 'defer {var_name}.Unlock()'. If the "
                        f"function returns or panics before Unlock is called, "
                        f"the mutex will remain locked, causing a deadlock."
                    ),
                    file_path=context.file_path,
                    line_start=line_num,
                    line_end=line_num,
                    confidence=0.78,
                    tags=["go", "goroutine", "concurrency", "mutex"],
                )
            )

    return findings


def _detect_waitgroup_misuse(root, context: CodeContext) -> list[Finding]:
    """Detect wg.Add() called inside a goroutine instead of before it.

    When ``wg.Add`` is called inside a ``go func() { ... }``, the main
    goroutine may reach ``wg.Wait()`` before the child has incremented the
    counter, causing a premature return.
    """
    findings: list[Finding] = []

    for node in walk(root):
        if node.type != "go_statement":
            continue

        go_text = get_node_text(node)

        # Check if wg.Add is inside the goroutine body
        if "wg.Add" in go_text:
            go_line = node.start_point[0] + 1
            findings.append(
                Finding(
                    agent_name="go_goroutine_safety",
                    severity="high",
                    category="bug",
                    title="WaitGroup.Add called inside goroutine",
                    description=(
                        f"'wg.Add' at line {go_line} is called inside a "
                        f"goroutine. The Add call must happen before the go "
                        f"statement to avoid a race where Wait returns before "
                        f"the goroutine increments the counter."
                    ),
                    file_path=context.file_path,
                    line_start=go_line,
                    line_end=node.end_point[0] + 1,
                    confidence=0.78,
                    tags=["go", "goroutine", "concurrency", "waitgroup"],
                )
            )

    return findings


class GoGoroutineSafetyAgent(BaseReviewAgent):
    """Detects common Go goroutine safety issues."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="go_goroutine_safety",
            version="0.1.0",
            languages=["go"],
            domains=["systems_programming", "distributed_systems"],
            methodology="language_specific",
            axis_type="critical",
            tags=["go", "goroutine", "concurrency"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if context.language.lower() != "go":
            return []
        if not context.source_code.strip():
            return []

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []
        findings.extend(_detect_goroutine_leak(root, context))
        findings.extend(_detect_channel_deadlock(root, context))
        findings.extend(_detect_lock_without_defer_unlock(root, context))
        findings.extend(_detect_waitgroup_misuse(root, context))
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Goroutine safety issues lead to subtle "
            f"concurrency bugs including deadlocks, goroutine leaks, and "
            f"race conditions. Use WaitGroups or contexts for lifecycle "
            f"management, defer Unlock after Lock, and call wg.Add before "
            f"launching goroutines."
        )
