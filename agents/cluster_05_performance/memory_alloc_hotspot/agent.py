"""Memory Allocation Hotspot Detector — flags allocation-heavy patterns inside loops."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, find_loops, get_node_text, walk
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Go allocation patterns inside loops
# ---------------------------------------------------------------------------

_GO_ALLOC_CALLS: set[str] = {"make", "new"}

# Matches append(slice, ...) — we flag append in loops when no pre-allocation
_GO_APPEND_RE = re.compile(r"\bappend\s*\(")

# Heuristic: detect pre-allocated slices via make([]T, 0, N) or make([]T, N)
_GO_PREALLOC_RE = re.compile(r"\bmake\s*\(\s*\[\s*\]")

# ---------------------------------------------------------------------------
# Java allocation patterns inside loops
# ---------------------------------------------------------------------------

_JAVA_ALLOC_RE: list[re.Pattern[str]] = [
    re.compile(r"\bnew\s+ArrayList\s*(<[^>]*>)?\s*\("),
    re.compile(r"\bnew\s+HashMap\s*(<[^>]*>)?\s*\("),
    re.compile(r"\bnew\s+HashSet\s*(<[^>]*>)?\s*\("),
    re.compile(r"\bnew\s+LinkedList\s*(<[^>]*>)?\s*\("),
    re.compile(r"\bnew\s+TreeMap\s*(<[^>]*>)?\s*\("),
    re.compile(r"\bnew\s+StringBuilder\s*\("),
    re.compile(r"\bnew\s+StringBuffer\s*\("),
    re.compile(r"\bnew\s+Object\s*\("),
    re.compile(r"\bnew\s+\w+\s*\("),  # generic new X() — lower priority catch-all
]

# ---------------------------------------------------------------------------
# Python allocation patterns inside loops
# ---------------------------------------------------------------------------

# Pattern: [].append() or list.append() inside a loop (not list comprehensions)
_PYTHON_APPEND_IN_LOOP_RE = re.compile(r"\w+\.append\s*\(")

# Pattern: object creation inside loop — ClassName(args)
_PYTHON_OBJ_CREATION_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*\s*\(")

# Known standard-library / builtin constructors that are cheap or expected
_PYTHON_CHEAP_CONSTRUCTORS: set[str] = {
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "int",
    "float",
    "str",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "type",
    "super",
    "len",
    "print",
    "isinstance",
    "issubclass",
    "hasattr",
    "getattr",
    "setattr",
    "ValueError",
    "TypeError",
    "KeyError",
    "RuntimeError",
    "Exception",
    "StopIteration",
    "IndexError",
    "AttributeError",
    "OSError",
    "IOError",
    "FileNotFoundError",
}

# ---------------------------------------------------------------------------
# Fix suggestions
# ---------------------------------------------------------------------------

_FIX_SUGGESTIONS: dict[str, str] = {
    "go": (
        "Pre-allocate the slice with make([]T, 0, expectedLen) before the loop "
        "to avoid repeated allocations during append."
    ),
    "java": (
        "Move the collection/object creation outside the loop, or reuse "
        "instances via clear() where applicable."
    ),
    "python": (
        "Consider caching the object outside the loop, using a list "
        "comprehension, or pre-sizing collections to reduce allocations."
    ),
}


class MemoryAllocHotspotAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="memory_alloc_hotspot",
            version="0.1.0",
            languages=["python", "java", "go"],
            domains=["systems_programming", "performance"],
            methodology="performance",
            axis_type="critical",
            tags=["performance", "memory", "allocation"],
            model_required=False,
            estimated_cost_cents=0.0,
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
        loops = find_loops(root, context.language)
        calls = find_calls(root, context.language)

        if lang == "go":
            findings.extend(self._analyze_go(context, loops, calls))
        elif lang == "java":
            findings.extend(self._analyze_java(context, loops, calls))
        elif lang == "python":
            findings.extend(self._analyze_python(context, loops, calls))

        return findings

    # ------------------------------------------------------------------
    # Go analysis
    # ------------------------------------------------------------------

    def _analyze_go(
        self,
        context: CodeContext,
        loops: list,
        calls: list,
    ) -> list[Finding]:
        findings: list[Finding] = []

        # Check if there's a pre-allocation (make([]...) before loops)
        prealloc_lines: set[int] = set()
        for line_no, line in enumerate(context.source_code.splitlines(), 1):
            if _GO_PREALLOC_RE.search(line):
                prealloc_lines.add(line_no)

        for call in calls:
            if not self._is_inside_loop(call, loops):
                continue

            # make() / new() inside loop
            if call.callee_name in _GO_ALLOC_CALLS:
                findings.append(
                    self._make_finding(
                        context,
                        call.start_line,
                        call.end_line,
                        f"'{call.callee_name}()' allocation inside loop",
                        (
                            f"'{call.callee_name}()' at line {call.start_line} "
                            f"allocates memory inside a loop. "
                            f"{_FIX_SUGGESTIONS['go']}"
                        ),
                    )
                )

            # append() inside loop without pre-allocation
            if call.callee_name == "append":
                # Check if there's any pre-allocation before this loop
                enclosing_loop = self._get_enclosing_loop(call, loops)
                has_prealloc = any(
                    ln < enclosing_loop.start_line for ln in prealloc_lines
                ) if enclosing_loop else False
                if not has_prealloc:
                    findings.append(
                        self._make_finding(
                            context,
                            call.start_line,
                            call.end_line,
                            "append() in loop without pre-allocated slice",
                            (
                                f"'append()' at line {call.start_line} grows a "
                                f"slice inside a loop without pre-allocation. "
                                f"{_FIX_SUGGESTIONS['go']}"
                            ),
                        )
                    )

        return findings

    # ------------------------------------------------------------------
    # Java analysis
    # ------------------------------------------------------------------

    def _analyze_java(
        self,
        context: CodeContext,
        loops: list,
        calls: list,
    ) -> list[Finding]:
        findings: list[Finding] = []

        for loop in loops:
            loop_text = get_node_text(loop.node)
            loop_lines = loop_text.splitlines()

            for j, line in enumerate(loop_lines):
                abs_line = loop.start_line + j
                for pat in _JAVA_ALLOC_RE:
                    if pat.search(line):
                        findings.append(
                            self._make_finding(
                                context,
                                abs_line,
                                abs_line,
                                "Object allocation inside loop",
                                (
                                    f"Allocation at line {abs_line} inside a loop "
                                    f"creates a new object each iteration. "
                                    f"{_FIX_SUGGESTIONS['java']}"
                                ),
                            )
                        )
                        break  # one finding per line

        return findings

    # ------------------------------------------------------------------
    # Python analysis
    # ------------------------------------------------------------------

    def _analyze_python(
        self,
        context: CodeContext,
        loops: list,
        calls: list,
    ) -> list[Finding]:
        findings: list[Finding] = []

        for call in calls:
            if not self._is_inside_loop(call, loops):
                continue

            callee = call.callee_name
            full = call.full_text

            # Skip cheap/expected builtins
            if callee in _PYTHON_CHEAP_CONSTRUCTORS:
                continue

            # Object creation: ClassName(...) — uppercase first letter
            if callee and callee[0].isupper():
                findings.append(
                    self._make_finding(
                        context,
                        call.start_line,
                        call.end_line,
                        f"Object creation '{callee}()' inside loop",
                        (
                            f"'{callee}()' at line {call.start_line} creates a new "
                            f"object each iteration. {_FIX_SUGGESTIONS['python']}"
                        ),
                    )
                )

        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_inside_loop(call, loops) -> bool:  # type: ignore[type-arg]
        """Check if a call is inside any loop by line range."""
        for loop in loops:
            if call.start_line >= loop.start_line and call.end_line <= loop.end_line:
                return True
        return False

    @staticmethod
    def _get_enclosing_loop(call, loops):  # type: ignore[type-arg]
        """Return the innermost loop enclosing this call, or None."""
        best = None
        for loop in loops:
            if call.start_line >= loop.start_line and call.end_line <= loop.end_line:
                if best is None or (loop.end_line - loop.start_line) < (
                    best.end_line - best.start_line
                ):
                    best = loop
        return best

    @staticmethod
    def _make_finding(
        context: CodeContext,
        line_start: int,
        line_end: int,
        title: str,
        description: str,
    ) -> Finding:
        return Finding(
            agent_name="memory_alloc_hotspot",
            severity="medium",
            category="performance",
            title=title,
            description=description,
            file_path=context.file_path,
            line_start=line_start,
            line_end=line_end,
            confidence=0.75,
            tags=["performance", "memory", "allocation"],
        )

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Allocating memory inside tight loops causes "
            f"excessive garbage collection pressure and cache thrashing, "
            f"degrading throughput and increasing latency."
        )
