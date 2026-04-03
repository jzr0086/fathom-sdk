"""Exception Swallowing Detector — flags empty catch/except blocks."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_try_catch_blocks, get_node_text
from fathom_sdk.context.ast_parser import parse


def _is_empty_handler(catch_node, language: str) -> bool:
    """Check if a catch/except handler body is empty or just pass."""
    lang = language.lower()
    if lang == "python":
        # except_clause children: "except", optional exception, ":", block
        # The block/body contains the statements
        body = catch_node.child_by_field_name("body")
        if body is None:
            # Find the block child
            for child in catch_node.children:
                if child.type == "block":
                    body = child
                    break
        if body is None:
            return True
        named = [c for c in body.named_children if c.type != "comment"]
        if not named:
            return True
        if len(named) == 1 and named[0].type == "pass_statement":
            return True
        return False
    else:
        # JS/TS/Java: catch_clause -> body (statement_block)
        body = catch_node.child_by_field_name("body")
        if body is None:
            for child in catch_node.children:
                if child.type in ("statement_block", "block"):
                    body = child
                    break
        if body is None:
            return True
        named = [c for c in body.named_children if c.type != "comment"]
        return len(named) == 0


class ExceptionSwallowingAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="exception_swallowing",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java"],
            domains=["web_development", "api_integration"],
            methodology="bug_detection",
            axis_type="aware",
            tags=["bug", "exception", "error-handling"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        if context.language.lower() not in ("python", "javascript", "typescript", "java"):
            return []
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []
        findings: list[Finding] = []
        for tc in find_try_catch_blocks(root, context.language):
            for catch_node in tc.catch_nodes:
                if _is_empty_handler(catch_node, context.language):
                    findings.append(
                        Finding(
                            agent_name="exception_swallowing",
                            severity="medium",
                            category="bug",
                            title="Empty exception handler swallows errors",
                            description=(
                                f"The catch/except block at line "
                                f"{catch_node.start_point[0] + 1} silently swallows "
                                f"exceptions. At minimum, log the error."
                            ),
                            file_path=context.file_path,
                            line_start=catch_node.start_point[0] + 1,
                            line_end=catch_node.end_point[0] + 1,
                            confidence=0.92,
                            tags=["bug", "exception", "error-handling"],
                        )
                    )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Swallowing exceptions hides bugs and makes debugging "
            f"extremely difficult. Always log, re-raise, or handle exceptions explicitly."
        )
