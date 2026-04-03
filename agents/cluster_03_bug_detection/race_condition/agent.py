"""Race Condition Detector — per-language concurrency hazard detection."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text, walk
from fathom_sdk.context.ast_parser import parse


def _detect_go_races(root, context) -> list[Finding]:
    """Detect Go goroutine race conditions."""
    findings: list[Finding] = []

    for node in walk(root):
        if node.type != "go_statement":
            continue

        go_line = node.start_point[0] + 1
        go_text = get_node_text(node)

        # Check for loop variable capture (classic Go bug)
        parent = node.parent
        while parent is not None:
            if parent.type == "for_statement":
                # Get the loop variable(s)
                loop_text = (
                    get_node_text(parent).split("{")[0] if "{" in get_node_text(parent) else ""
                )
                # Extract range variable names
                range_vars = re.findall(r"(\w+)\s*(?:,\s*(\w+))?\s*:=\s*range", loop_text)
                for match in range_vars:
                    for var in match:
                        if var and var in go_text:
                            findings.append(
                                Finding(
                                    agent_name="race_condition",
                                    severity="high",
                                    category="bug",
                                    title=f"Goroutine captures loop variable '{var}'",
                                    description=(
                                        f"Goroutine at line {go_line} captures loop "
                                        f"variable '{var}'. All goroutines will share the "
                                        f"same variable. Pass it as a parameter instead."
                                    ),
                                    file_path=context.file_path,
                                    line_start=go_line,
                                    line_end=node.end_point[0] + 1,
                                    confidence=0.88,
                                    tags=["bug", "concurrency", "race-condition", "goroutine"],
                                )
                            )
                break
            parent = parent.parent

    return findings


def _detect_python_races(root, context) -> list[Finding]:
    """Detect Python threading race conditions."""
    findings: list[Finding] = []
    source = context.source_code

    # Find Thread target= with shared variable access
    thread_pattern = re.compile(r"threading\.Thread\s*\(\s*target\s*=\s*(\w+)", re.MULTILINE)
    for match in thread_pattern.finditer(source):
        line_num = source[: match.start()].count("\n") + 1
        findings.append(
            Finding(
                agent_name="race_condition",
                severity="medium",
                category="bug",
                title=f"Thread created — verify shared state is synchronized",
                description=(
                    f"threading.Thread at line {line_num} targets '{match.group(1)}'. "
                    f"Ensure shared state is protected with locks."
                ),
                file_path=context.file_path,
                line_start=line_num,
                line_end=line_num,
                confidence=0.70,
                tags=["bug", "concurrency", "race-condition", "threading"],
            )
        )
    return findings


def _detect_java_races(root, context) -> list[Finding]:
    """Detect Java concurrency issues."""
    findings: list[Finding] = []
    source = context.source_code
    lines = source.splitlines()

    # Find field modifications in methods that aren't synchronized
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "synchronized" not in stripped and "volatile" not in stripped:
            # Check for field access patterns in Runnable/Thread subclasses
            if re.search(r"new\s+Thread\s*\(", stripped):
                findings.append(
                    Finding(
                        agent_name="race_condition",
                        severity="medium",
                        category="bug",
                        title="Thread creation — verify synchronization",
                        description=(
                            f"New Thread at line {i}. Ensure shared mutable state "
                            f"is protected with synchronized blocks or concurrent types."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.70,
                        tags=["bug", "concurrency", "race-condition"],
                    )
                )
    return findings


class RaceConditionAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="race_condition",
            version="0.1.0",
            languages=["python", "java", "go"],
            domains=["distributed_systems", "systems_programming"],
            methodology="bug_detection",
            axis_type="critical",
            tags=["bug", "concurrency", "race-condition"],
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

        if lang == "go":
            return _detect_go_races(root, context)
        elif lang == "python":
            return _detect_python_races(root, context)
        elif lang == "java":
            return _detect_java_races(root, context)
        return []

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Race conditions cause intermittent, hard-to-debug "
            f"failures. Use proper synchronization: mutexes, channels (Go), "
            f"locks (Python), or synchronized/concurrent types (Java)."
        )
