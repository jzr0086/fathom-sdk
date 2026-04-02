"""BaseReviewAgent — the interface every Fathom agent must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from fathom_sdk.schema import AgentMetadata, CodeFix, Finding

if TYPE_CHECKING:
    from fathom_sdk.context.code_context import CodeContext


class BaseReviewAgent(ABC):
    """Abstract base class for all review agents.

    The orchestrator only knows this interface.  Every agent in the
    repository must subclass this and implement the abstract methods.
    """

    @abstractmethod
    def analyze(self, context: CodeContext) -> list[Finding]:
        """Core analysis logic.  Must be fast and precise."""

    @abstractmethod
    def explain(self, finding: Finding) -> str:
        """Human-readable explanation of why this finding matters."""

    def suggest_fix(self, finding: Finding) -> Optional[CodeFix]:
        """Optional fix suggestion.  Return None if not applicable."""
        return None

    def confidence(self, finding: Finding) -> float:
        """Calibrated confidence score for this finding."""
        return finding.confidence

    @abstractmethod
    def metadata(self) -> AgentMetadata:
        """Agent metadata for routing and documentation."""
