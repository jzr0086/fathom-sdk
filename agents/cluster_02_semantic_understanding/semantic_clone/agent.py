"""Semantic Clone Detection — finds Type-3/4 code clones via CodeBERT embeddings.

Type-3 clones are near-miss clones with modifications (added/removed statements,
changed expressions). Type-4 clones are semantically equivalent functions with
entirely different syntax. Both are missed by textual duplicate detectors but
caught by comparing learned code embeddings.
"""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_functions, get_node_text
from fathom_sdk.context.ast_parser import parse
from fathom_sdk.context.embeddings import find_similar_pairs, get_code_embeddings

_SIMILARITY_THRESHOLD = 0.85
_MIN_FUNCTION_LINES = 3


class SemanticCloneAgent(BaseReviewAgent):
    """Detects semantically similar function pairs using CodeBERT embeddings.

    Functions are extracted from the AST, their source text is embedded via
    CodeBERT (sentence-transformers), and cosine similarity is computed
    pairwise. Pairs above the threshold are reported as semantic clones.

    When ML dependencies (torch, sentence-transformers) are not installed,
    the agent gracefully returns an empty list.
    """

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="semantic_clone",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["web_development", "enterprise_engineering"],
            methodology="semantic_understanding",
            axis_type="aware",
            tags=["semantic", "clone", "duplication"],
            model_required=True,
            estimated_cost_cents=2.0,
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

        functions = find_functions(root, context.language)
        # Need at least two non-trivial functions to compare
        functions = [f for f in functions if f.line_count >= _MIN_FUNCTION_LINES]
        if len(functions) < 2:
            return []

        # Extract source text of each function for embedding
        function_texts = [get_node_text(f.node) for f in functions]

        # Compute embeddings via CodeBERT
        embeddings = get_code_embeddings(function_texts)
        if embeddings is None:
            # ML dependencies not available — gracefully degrade
            return []

        # Find similar pairs above the threshold
        pairs = find_similar_pairs(embeddings, threshold=_SIMILARITY_THRESHOLD)

        findings: list[Finding] = []
        for idx_a, idx_b, similarity in pairs:
            func_a = functions[idx_a]
            func_b = functions[idx_b]
            findings.append(
                Finding(
                    agent_name="semantic_clone",
                    severity="medium",
                    category="quality",
                    title=(
                        f"Semantic clone: '{func_a.name}' and '{func_b.name}' "
                        f"({similarity:.0%} similar)"
                    ),
                    description=(
                        f"Functions '{func_a.name}' (lines {func_a.start_line}-"
                        f"{func_a.end_line}) and '{func_b.name}' (lines "
                        f"{func_b.start_line}-{func_b.end_line}) are semantically "
                        f"similar ({similarity:.2%} cosine similarity). This may "
                        f"indicate a Type-3 or Type-4 clone. Consider extracting "
                        f"the shared logic into a single reusable function."
                    ),
                    file_path=context.file_path,
                    line_start=func_a.start_line,
                    line_end=func_b.end_line,
                    confidence=similarity,
                    tags=["semantic", "clone", "duplication"],
                )
            )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Semantic clones are functions that perform "
            f"essentially the same computation despite having different variable "
            f"names, minor structural changes, or different coding styles. "
            f"They lead to duplicated maintenance effort and inconsistent bug "
            f"fixes. Extract the common logic into a shared helper."
        )
