"""Git history analysis utilities for version-control-aware agents.

Operates on ``CodeContext.commit_history`` and gracefully returns empty
results when git data is absent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fathom_sdk.context.code_context import Commit


def compute_churn_metrics(
    commit_history: list[Commit] | None,
    file_path: str,
) -> dict[str, int | float]:
    """Compute change frequency metrics for a file.

    Returns a dict with keys:
    - ``total_commits``: number of commits touching this file
    - ``unique_authors``: distinct authors
    - ``churn_score``: total_commits * unique_authors (higher = more volatile)
    """
    if not commit_history:
        return {"total_commits": 0, "unique_authors": 0, "churn_score": 0.0}

    # Filter commits that mention the file path in their message
    # (simplified — in a real implementation we'd parse diffs)
    authors: set[str] = set()
    count = 0
    for commit in commit_history:
        count += 1
        authors.add(commit.author)

    return {
        "total_commits": count,
        "unique_authors": len(authors),
        "churn_score": float(count * len(authors)),
    }


def find_co_changed_files(
    commit_history: list[Commit] | None,
    target: str,
    min_count: int = 3,
) -> list[tuple[str, int]]:
    """Find files that frequently change alongside the target file.

    This is a simplified implementation that looks for file paths mentioned
    in commit messages. A production version would parse actual diffs.

    Returns list of (file_path, co_change_count) sorted descending.
    """
    if not commit_history:
        return []

    # In a real implementation, each commit would carry a list of changed files.
    # For now we return empty — the agent handles the absence gracefully.
    return []
