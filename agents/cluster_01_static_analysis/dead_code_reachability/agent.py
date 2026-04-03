"""Dead Code Reachability — detects unreachable code after return/raise/throw."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text, walk
from fathom_sdk.context.ast_parser import parse

_TERMINATOR_TYPES: dict[str, set[str]] = {
    "python": {"return_statement", "raise_statement"},
    "javascript": {"return_statement", "throw_statement"},
    "typescript": {"return_statement", "throw_statement"},
    "java": {"return_statement", "throw_statement"},
    "go": {"return_statement"},
}


def _find_dead_code_in_block(block_node, language: str) -> list[tuple[int, int, str]]:
    """Find unreachable statements after terminators in a block."""
    lang = language.lower()
    terminators = _TERMINATOR_TYPES.get(lang, set())
    if not terminators:
        return []

    dead: list[tuple[int, int, str]] = []

    # Go wraps statements in block -> statement_list; unwrap if needed
    children = [c for c in block_node.named_children if c.type != "comment"]
    if len(children) == 1 and children[0].type == "statement_list":
        children = [c for c in children[0].named_children if c.type != "comment"]

    found_terminator = False
    for child in children:
        if found_terminator:
            dead.append(
                (
                    child.start_point[0] + 1,
                    child.end_point[0] + 1,
                    get_node_text(child)[:60],
                )
            )
        elif child.type in terminators:
            found_terminator = True
    return dead


class DeadCodeReachabilityAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="dead_code_reachability",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "api_integration"],
            methodology="static_analysis",
            axis_type="aware",
            tags=["quality", "dead-code", "unreachable"],
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

        lang = context.language.lower()
        terminators = _TERMINATOR_TYPES.get(lang, set())
        if not terminators:
            return []

        findings: list[Finding] = []

        # Check inside function bodies
        for func in find_functions(root, context.language):
            body = func.node.child_by_field_name("body")
            if body is None:
                continue
            dead = _find_dead_code_in_block(body, context.language)
            for start, end, preview in dead:
                findings.append(
                    Finding(
                        agent_name="dead_code_reachability",
                        severity="low",
                        category="quality",
                        title=f"Unreachable code after return/raise in '{func.name}'",
                        description=(
                            f"Code at line {start} is unreachable because a "
                            f"return/raise/throw statement precedes it. "
                            f"Preview: {preview}"
                        ),
                        file_path=context.file_path,
                        line_start=start,
                        line_end=end,
                        confidence=0.92,
                        tags=["quality", "dead-code", "unreachable"],
                    )
                )

        # Also check nested blocks (if/else/try bodies)
        for node in walk(root):
            if node.type in ("block", "statement_block"):
                parent = node.parent
                if parent is not None and parent.type not in (
                    "function_definition",
                    "method_declaration",
                    "function_declaration",
                    "method_definition",
                    "arrow_function",
                    "constructor_declaration",
                ):
                    dead = _find_dead_code_in_block(node, context.language)
                    for start, end, preview in dead:
                        findings.append(
                            Finding(
                                agent_name="dead_code_reachability",
                                severity="low",
                                category="quality",
                                title="Unreachable code after return/raise",
                                description=(
                                    f"Code at line {start} is unreachable. Preview: {preview}"
                                ),
                                file_path=context.file_path,
                                line_start=start,
                                line_end=end,
                                confidence=0.92,
                                tags=["quality", "dead-code", "unreachable"],
                            )
                        )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Dead code adds confusion and maintenance burden. "
            f"Remove it or restructure the control flow."
        )
