"""Taint flow analysis engine for security agents.

Traces data flows from untrusted sources (user input) to dangerous sinks
(SQL queries, shell commands, file paths, URLs, HTML output).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tree_sitter

from fathom_sdk.context.ast_helpers import (
    find_assignments,
    find_calls,
    get_node_text,
    walk,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaintSource:
    """A location where untrusted data enters the program."""

    variable: str
    source_type: str  # "user_input", "http_param", "env_var", "file_read"
    line: int


@dataclass(slots=True)
class TaintSink:
    """A dangerous operation that should not receive tainted data."""

    callee: str
    sink_type: str  # "sql", "shell", "file", "url", "html", "deserialize"
    line: int
    full_text: str


@dataclass(slots=True)
class TaintFlow:
    """A traced flow from source to sink."""

    source: TaintSource
    sink: TaintSink
    tainted_variable: str
    vuln_type: str  # "sql_injection", "command_injection", "path_traversal", "ssrf", "xss"
    confidence: float


# ---------------------------------------------------------------------------
# Per-language source/sink definitions
# ---------------------------------------------------------------------------

_SOURCE_PATTERNS: dict[str, dict[str, list[str]]] = {
    "python": {
        "user_input": ["input", "raw_input"],
        "http_param": [
            "request.args.get", "request.form.get", "request.GET.get",
            "request.POST.get", "request.params", "request.json",
            "request.data", "request.query_params",
        ],
        "env_var": ["os.environ.get", "os.getenv"],
        "file_read": ["stdin.read", "sys.stdin"],
    },
    "javascript": {
        "user_input": ["prompt", "readline"],
        "http_param": [
            "req.params", "req.query", "req.body", "req.headers",
            "request.params", "request.query", "request.body",
        ],
        "env_var": ["process.env"],
    },
    "typescript": {
        "user_input": ["prompt", "readline"],
        "http_param": [
            "req.params", "req.query", "req.body", "req.headers",
            "request.params", "request.query", "request.body",
        ],
        "env_var": ["process.env"],
    },
    "java": {
        "user_input": ["Scanner", "readLine", "nextLine"],
        "http_param": [
            "getParameter", "getQueryString", "getHeader",
            "getRequestURI", "getPathInfo",
        ],
        "env_var": ["System.getenv"],
    },
    "go": {
        "user_input": ["Scanf", "ReadString", "ReadLine"],
        "http_param": [
            "FormValue", "URL.Query", "Header.Get",
            "r.URL", "r.Body",
        ],
        "env_var": ["os.Getenv"],
    },
}

_SINK_PATTERNS: dict[str, dict[str, list[str]]] = {
    "python": {
        "sql": ["execute", "executemany", "raw", "cursor.execute"],
        "shell": ["os.system", "subprocess.run", "subprocess.call", "subprocess.Popen", "popen"],
        "file": ["open", "os.path.join", "pathlib.Path"],
        "url": ["requests.get", "requests.post", "urlopen", "urllib.request.urlopen", "httpx.get"],
        "html": ["render_template_string", "Markup", "format_html"],
        "deserialize": ["pickle.loads", "yaml.load", "marshal.loads"],
    },
    "javascript": {
        "sql": ["query", "execute", "raw"],
        "shell": ["exec", "execSync", "spawn", "execFile"],
        "file": ["readFile", "readFileSync", "writeFile", "createReadStream"],
        "url": ["fetch", "axios.get", "axios.post", "http.get", "http.request"],
        "html": ["innerHTML", "document.write", "insertAdjacentHTML"],
        "deserialize": ["JSON.parse", "eval"],
    },
    "typescript": {
        "sql": ["query", "execute", "raw"],
        "shell": ["exec", "execSync", "spawn", "execFile"],
        "file": ["readFile", "readFileSync", "writeFile", "createReadStream"],
        "url": ["fetch", "axios.get", "axios.post", "http.get", "http.request"],
        "html": ["innerHTML", "document.write", "insertAdjacentHTML"],
        "deserialize": ["JSON.parse", "eval"],
    },
    "java": {
        "sql": ["executeQuery", "executeUpdate", "execute", "prepareStatement"],
        "shell": ["Runtime.exec", "ProcessBuilder"],
        "file": ["FileInputStream", "FileReader", "Paths.get", "new File"],
        "url": ["URL", "HttpURLConnection", "HttpClient"],
        "html": ["PrintWriter.write", "response.getWriter"],
        "deserialize": ["ObjectInputStream", "readObject", "XMLDecoder"],
    },
    "go": {
        "sql": ["Query", "Exec", "QueryRow", "Prepare"],
        "shell": ["exec.Command", "os.StartProcess"],
        "file": ["os.Open", "ioutil.ReadFile", "os.ReadFile", "filepath.Join"],
        "url": ["http.Get", "http.Post", "http.NewRequest"],
        "html": ["template.HTML", "fmt.Fprintf"],
        "deserialize": ["json.Unmarshal", "gob.Decode"],
    },
}

# Map sink types to vulnerability categories
_SINK_TO_VULN: dict[str, str] = {
    "sql": "sql_injection",
    "shell": "command_injection",
    "file": "path_traversal",
    "url": "ssrf",
    "html": "xss",
    "deserialize": "deserialization",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trace_taint_flows(
    root: tree_sitter.Node,
    language: str,
    source_code: str,
) -> list[TaintFlow]:
    """Trace data flows from taint sources to sinks.

    This is a simplified intra-procedural analysis that tracks variable
    assignments from known source patterns to known sink patterns.
    """
    lang = language.lower()
    source_defs = _SOURCE_PATTERNS.get(lang, {})
    sink_defs = _SINK_PATTERNS.get(lang, {})
    if not source_defs and not sink_defs:
        return []

    # Step 1: identify tainted variables from source patterns
    tainted_vars: dict[str, TaintSource] = {}

    # Check assignments for source calls
    for assign in find_assignments(root, language):
        for source_type, patterns in source_defs.items():
            for pattern in patterns:
                if pattern in assign.value_text:
                    tainted_vars[assign.target] = TaintSource(
                        variable=assign.target,
                        source_type=source_type,
                        line=assign.start_line,
                    )

    # Also check for function parameters that look like request handlers
    source_lines = source_code.splitlines()
    for i, line in enumerate(source_lines, 1):
        stripped = line.strip()
        # Common request handler patterns
        for source_type, patterns in source_defs.items():
            for pattern in patterns:
                if pattern in stripped:
                    # Extract variable name from assignment
                    if "=" in stripped:
                        var = stripped.split("=")[0].strip().split()[-1]
                        if var and var.isidentifier():
                            tainted_vars[var] = TaintSource(
                                variable=var,
                                source_type=source_type,
                                line=i,
                            )

    # Step 2: propagate taint through assignments
    for assign in find_assignments(root, language):
        for tainted_var in list(tainted_vars.keys()):
            if tainted_var in assign.value_text and assign.target != tainted_var:
                tainted_vars[assign.target] = TaintSource(
                    variable=assign.target,
                    source_type=tainted_vars[tainted_var].source_type,
                    line=assign.start_line,
                )

    if not tainted_vars:
        return []

    # Step 3: find sinks and check for tainted data
    flows: list[TaintFlow] = []
    seen: set[tuple[str, int]] = set()

    for call in find_calls(root, language):
        for sink_type, patterns in sink_defs.items():
            if not any(p in call.callee_name or p in call.full_text for p in patterns):
                continue
            # Check if any tainted variable appears in the call
            for var, source in tainted_vars.items():
                if var in call.full_text:
                    key = (var, call.start_line)
                    if key in seen:
                        continue
                    seen.add(key)
                    vuln_type = _SINK_TO_VULN.get(sink_type, sink_type)
                    flows.append(
                        TaintFlow(
                            source=source,
                            sink=TaintSink(
                                callee=call.callee_name,
                                sink_type=sink_type,
                                line=call.start_line,
                                full_text=call.full_text,
                            ),
                            tainted_variable=var,
                            vuln_type=vuln_type,
                            confidence=0.80,
                        )
                    )

    return flows
