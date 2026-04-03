"""Resource Leak Detector — flags file/connection opens without proper cleanup."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, walk
from fathom_sdk.context.ast_parser import parse

_RESOURCE_OPEN_CALLS: dict[str, set[str]] = {
    "python": {"open", "connect", "socket", "cursor", "urlopen"},
    "javascript": {
        "openSync",
        "createReadStream",
        "createWriteStream",
        "createConnection",
        "connect",
    },
    "typescript": {
        "openSync",
        "createReadStream",
        "createWriteStream",
        "createConnection",
        "connect",
    },
    "java": {
        "FileInputStream",
        "FileOutputStream",
        "BufferedReader",
        "BufferedWriter",
        "getConnection",
        "openConnection",
        "openStream",
        "newInputStream",
        "newOutputStream",
    },
    "go": {
        "Open",
        "Create",
        "OpenFile",
        "Dial",
        "Listen",
        "NewReader",
        "NewWriter",
        "NewScanner",
    },
}


def _is_inside_with(node, language: str) -> bool:
    """Check if a node is inside a `with` statement (Python) or try-with-resources (Java)."""
    current = node.parent
    while current is not None:
        if language.lower() == "python" and current.type == "with_statement":
            return True
        if language.lower() == "java" and current.type == "try_with_resources_statement":
            return True
        current = current.parent
    return False


def _has_defer_close(node, root, language: str) -> bool:
    """Check if there's a defer .Close() after this node (Go)."""
    if language.lower() != "go":
        return False
    # Look for defer statements after this line in the same function scope
    node_line = node.start_point[0]
    for n in walk(root):
        if n.type == "defer_statement" and n.start_point[0] > node_line:
            from fathom_sdk.context.ast_helpers import get_node_text

            text = get_node_text(n)
            if "Close()" in text or "close()" in text:
                return True
    return False


class ResourceLeakAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="resource_leak",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "systems_programming"],
            methodology="bug_detection",
            axis_type="aware",
            tags=["bug", "resource-leak", "file-handle"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        open_calls = _RESOURCE_OPEN_CALLS.get(lang)
        if not open_calls:
            return []
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []
        for call in find_calls(root, context.language):
            if call.callee_name not in open_calls:
                continue

            # Check for proper resource management
            if lang == "python":
                if _is_inside_with(call.node, context.language):
                    continue
            elif lang == "java":
                if _is_inside_with(call.node, context.language):
                    continue
            elif lang == "go":
                if _has_defer_close(call.node, root, context.language):
                    continue

            findings.append(
                Finding(
                    agent_name="resource_leak",
                    severity="medium",
                    category="bug",
                    title=f"Potential resource leak: '{call.callee_name}' without cleanup",
                    description=(
                        f"Resource opened via '{call.callee_name}' at line "
                        f"{call.start_line} may not be properly closed. "
                        f"Use 'with' (Python), try-with-resources (Java), "
                        f"or defer (Go)."
                    ),
                    file_path=context.file_path,
                    line_start=call.start_line,
                    line_end=call.end_line,
                    confidence=0.83,
                    tags=["bug", "resource-leak", "file-handle"],
                )
            )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Leaked resources (files, connections, sockets) "
            f"consume system resources and can cause outages. Always ensure "
            f"cleanup via context managers, try-finally, or defer."
        )
