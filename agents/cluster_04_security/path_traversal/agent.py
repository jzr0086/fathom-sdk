"""Path Traversal Detector — flags unsanitized user input in file operations."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, get_node_text
from fathom_sdk.context.ast_parser import parse
from fathom_sdk.context.taint_analysis import trace_taint_flows

# File I/O call names that accept paths
_FILE_IO_CALLS: set[str] = {
    # Python
    "open",
    "read",
    "write",
    "readlines",
    # os / os.path / pathlib
    "join",          # os.path.join
    "Path",          # pathlib.Path
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "listdir",
    "scandir",
    "remove",
    "unlink",
    "rename",
    "mkdir",
    "rmdir",
    "stat",
    # Node.js fs
    "readFile",
    "readFileSync",
    "writeFile",
    "writeFileSync",
    "createReadStream",
    "createWriteStream",
    "readdir",
    "readdirSync",
    "access",
    "accessSync",
    "existsSync",
    "unlinkSync",
    # Java
    "FileInputStream",
    "FileOutputStream",
    "FileReader",
    "FileWriter",
    "File",
    # Go
    "Open",
    "Create",
    "ReadFile",
    "WriteFile",
    "ReadDir",
    "Stat",
    "Remove",
    "Join",          # filepath.Join
}

# Patterns that indicate path traversal strings near file operations
_TRAVERSAL_LITERAL = re.compile(r"\.\./|\.\.\\")

# Sanitization functions that neutralise path traversal
_SANITIZERS: set[str] = {
    "basename",      # os.path.basename / path.basename
    "realpath",      # os.path.realpath
    "abspath",       # os.path.abspath
    "normpath",      # os.path.normpath
    "resolve",       # pathlib resolve / Node path.resolve
    "Clean",         # filepath.Clean (Go)
    "Abs",           # filepath.Abs (Go)
    "normalize",     # path.normalize (Node)
    "getCanonicalPath",  # Java File.getCanonicalPath
    "toRealPath",        # Java Path.toRealPath
}

# Regex patterns for user-input sources flowing into file I/O (source-level)
_USER_INPUT_PATTERNS = [
    re.compile(r"request\.\w+"),       # Flask/Django/Express request attrs
    re.compile(r"req\.\w+"),           # Express req.params / req.query / req.body
    re.compile(r"getParameter\s*\("),  # Java servlet
    re.compile(r"FormValue\s*\("),     # Go http
    re.compile(r"input\s*\("),         # Python input()
    re.compile(r"sys\.argv"),          # Python command-line
    re.compile(r"process\.argv"),      # Node command-line
    re.compile(r"args\b"),             # Generic args variable
]

# File operation patterns (source-level fallback)
_FILE_OP_PATTERNS = [
    re.compile(r"\bopen\s*\("),
    re.compile(r"os\.path\.join\s*\("),
    re.compile(r"pathlib\.Path\s*\("),
    re.compile(r"Path\s*\("),
    re.compile(r"fs\.\w+\s*\("),
    re.compile(r"readFile\w*\s*\("),
    re.compile(r"writeFile\w*\s*\("),
    re.compile(r"createReadStream\s*\("),
    re.compile(r"createWriteStream\s*\("),
    re.compile(r"FileInputStream\s*\("),
    re.compile(r"FileReader\s*\("),
    re.compile(r"os\.Open\s*\("),
    re.compile(r"os\.ReadFile\s*\("),
    re.compile(r"ioutil\.ReadFile\s*\("),
    re.compile(r"filepath\.Join\s*\("),
]

_SANITIZER_PATTERNS = [
    re.compile(r"os\.path\.basename\s*\("),
    re.compile(r"os\.path\.realpath\s*\("),
    re.compile(r"os\.path\.abspath\s*\("),
    re.compile(r"os\.path\.normpath\s*\("),
    re.compile(r"\.resolve\s*\("),
    re.compile(r"path\.resolve\s*\("),
    re.compile(r"path\.normalize\s*\("),
    re.compile(r"filepath\.Clean\s*\("),
    re.compile(r"filepath\.Abs\s*\("),
    re.compile(r"\.getCanonicalPath\s*\("),
    re.compile(r"\.toRealPath\s*\("),
    re.compile(r"\.basename\s*\("),
    re.compile(r"path\.basename\s*\("),
]


def _line_has_sanitizer(line: str) -> bool:
    """Return True if the line contains a known path-sanitization call."""
    return any(pat.search(line) for pat in _SANITIZER_PATTERNS)


def _function_has_sanitizer(source_code: str, start_line: int) -> bool:
    """Check surrounding context (within 5 lines before) for sanitisation."""
    lines = source_code.splitlines()
    lo = max(0, start_line - 6)  # start_line is 1-based
    hi = start_line  # exclusive, includes the target line
    for line in lines[lo:hi]:
        if _line_has_sanitizer(line):
            return True
    return False


class PathTraversalAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="path_traversal",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development"],
            methodology="security",
            axis_type="aware",
            tags=["security", "path-traversal", "injection"],
            model_required=False,
            estimated_cost_cents=0.0,
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

        findings: list[Finding] = []
        seen_lines: set[int] = set()

        # --- Strategy 1: taint-flow analysis ----------------------------------
        try:
            flows = trace_taint_flows(root, context.language, context.source_code)
        except Exception:
            flows = []

        for flow in flows:
            if flow.vuln_type != "path_traversal":
                continue
            if flow.sink.line in seen_lines:
                continue
            # Skip if a sanitizer appears near the sink
            if _function_has_sanitizer(context.source_code, flow.sink.line):
                continue
            seen_lines.add(flow.sink.line)
            findings.append(
                Finding(
                    agent_name="path_traversal",
                    severity="high",
                    category="security",
                    title="Path traversal: user input flows to file operation",
                    description=(
                        f"Tainted variable '{flow.tainted_variable}' from "
                        f"{flow.source.source_type} (line {flow.source.line}) "
                        f"reaches file operation '{flow.sink.callee}' at line "
                        f"{flow.sink.line} without path sanitization."
                    ),
                    file_path=context.file_path,
                    line_start=flow.sink.line,
                    line_end=flow.sink.line,
                    confidence=0.82,
                    tags=["security", "path-traversal", "injection"],
                )
            )

        # --- Strategy 2: regex-based detection --------------------------------
        lines = context.source_code.splitlines()

        # Build a set of variable names assigned from user-input sources.
        # This enables cross-line detection (input on line N, file op on line M).
        _ASSIGN_RE = re.compile(r"(\w+)\s*[:=]\s*(.+)")
        tainted_vars: set[str] = set()
        sanitized_vars: set[str] = set()
        for _line in lines:
            s = _line.strip()
            if s.startswith(("#", "//", "/*", "*")):
                continue
            m = _ASSIGN_RE.match(s)
            if m:
                var, val = m.group(1), m.group(2)
                # Check if value comes from user input
                if any(pat.search(val) for pat in _USER_INPUT_PATTERNS):
                    tainted_vars.add(var)
                # Propagate: if a tainted var appears in the RHS, the LHS is tainted
                elif any(tv in val for tv in tainted_vars):
                    tainted_vars.add(var)
                # Track sanitized assignments
                if _line_has_sanitizer(val):
                    sanitized_vars.add(var)

        for i, line in enumerate(lines, 1):
            if i in seen_lines:
                continue
            stripped = line.strip()
            # Skip comments
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            # Skip lines with sanitization already present
            if _line_has_sanitizer(line):
                continue

            # 2a: literal "../" in a file-operation context
            if _TRAVERSAL_LITERAL.search(line):
                has_file_op = any(pat.search(line) for pat in _FILE_OP_PATTERNS)
                if has_file_op:
                    seen_lines.add(i)
                    findings.append(
                        Finding(
                            agent_name="path_traversal",
                            severity="high",
                            category="security",
                            title="Path traversal: literal '../' in file operation",
                            description=(
                                f"Line {i} contains a file operation with a literal "
                                f"'../' traversal pattern. Ensure this path cannot be "
                                f"influenced by user input."
                            ),
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=0.82,
                            tags=["security", "path-traversal", "injection"],
                        )
                    )
                    continue

            # 2b: user input flowing directly into file operations on the same line
            has_user_input = any(pat.search(line) for pat in _USER_INPUT_PATTERNS)
            has_file_op = any(pat.search(line) for pat in _FILE_OP_PATTERNS)

            if has_user_input and has_file_op:
                # Check surrounding lines for sanitisation
                if _function_has_sanitizer(context.source_code, i):
                    continue
                seen_lines.add(i)
                findings.append(
                    Finding(
                        agent_name="path_traversal",
                        severity="high",
                        category="security",
                        title="Path traversal: user input in file operation",
                        description=(
                            f"Line {i} passes user-controlled input directly to a "
                            f"file operation without path sanitization. Use "
                            f"os.path.basename() or os.path.realpath() to restrict "
                            f"the path."
                        ),
                        file_path=context.file_path,
                        line_start=i,
                        line_end=i,
                        confidence=0.82,
                        tags=["security", "path-traversal", "injection"],
                    )
                )
                continue

            # 2c: cross-line — tainted variable used in a file operation
            if has_file_op and not has_user_input:
                # Check if any tainted (and not sanitized) variable appears in this line
                for tv in tainted_vars:
                    if tv in sanitized_vars:
                        continue
                    # Ensure the variable appears as a word (not substring of another)
                    if re.search(r"\b" + re.escape(tv) + r"\b", line):
                        if _function_has_sanitizer(context.source_code, i):
                            break
                        seen_lines.add(i)
                        findings.append(
                            Finding(
                                agent_name="path_traversal",
                                severity="high",
                                category="security",
                                title="Path traversal: user input in file operation",
                                description=(
                                    f"Line {i} passes user-controlled variable "
                                    f"'{tv}' to a file operation without path "
                                    f"sanitization. Use os.path.basename() or "
                                    f"os.path.realpath() to restrict the path."
                                ),
                                file_path=context.file_path,
                                line_start=i,
                                line_end=i,
                                confidence=0.82,
                                tags=["security", "path-traversal", "injection"],
                            )
                        )
                        break

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Path traversal allows attackers to read or write "
            f"arbitrary files by injecting sequences like '../' into file paths. "
            f"Always validate and sanitize file paths using os.path.basename(), "
            f"os.path.realpath(), or equivalent functions to confine access to "
            f"the intended directory."
        )
