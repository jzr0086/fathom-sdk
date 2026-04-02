"""CodeContext — the shared, parse-once data container for all agents.

This dataclass is populated by the orchestrator at the start of a review
and passed to every agent.  Individual agents MUST NOT re-parse source
code; they consume the pre-built representations here.

We use plain dataclasses (not Pydantic) because the container holds
heterogeneous, non-serialisable objects such as tree-sitter AST nodes,
NetworkX graphs, and NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import networkx as nx
    import numpy as np

    from fathom_sdk.schema.finding import Finding


@dataclass(frozen=True, slots=True)
class PRMetadata:
    """Metadata pulled from the hosting platform (GitHub, GitLab, etc.)."""

    pr_number: int
    title: str
    author: str
    base_branch: str
    head_branch: str
    labels: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Commit:
    """A single commit in the file's history."""

    sha: str
    message: str
    author: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class Token:
    """A lexical token produced during tokenisation."""

    type: str
    value: str
    line: int
    column: int


@dataclass
class CodeContext:
    """Unified context object shared across all review agents.

    The orchestrator builds this once per file under review.  Fields
    that require expensive computation (graphs, embeddings) are Optional
    and may be None when the corresponding analysis pass is skipped.
    """

    # Raw content
    source_code: str
    language: str
    file_path: str
    framework: Optional[str] = None

    # Parsed representations (computed once, shared)
    ast: Any = None
    token_stream: list[Token] = field(default_factory=list)
    call_graph: Optional[nx.DiGraph] = None
    data_flow_graph: Optional[nx.DiGraph] = None
    control_flow_graph: Optional[nx.DiGraph] = None

    # Git context
    diff: Optional[str] = None
    blame: Optional[dict[str, Any]] = None
    commit_history: Optional[list[Commit]] = None
    pr_metadata: Optional[PRMetadata] = None

    # Codebase context
    file_embeddings: Optional[np.ndarray] = None
    historical_findings: Optional[list[Finding]] = None
    codebase_statistics: Optional[dict[str, Any]] = None

    # Classification results (set by the classifier)
    detected_domains: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    recommended_agents: list[str] = field(default_factory=list)
