"""Fathom SDK — build code review agents for the Fathom platform.

Convenience re-exports so agent authors can write::

    from fathom_sdk import BaseReviewAgent, Finding, CodeContext
"""

from fathom_sdk.agent.base import BaseReviewAgent
from fathom_sdk.agent.registry import AgentRegistry, registry
from fathom_sdk.context.code_context import CodeContext, Commit, PRMetadata, Token
from fathom_sdk.schema.finding import CodeFix, Finding
from fathom_sdk.schema.metadata import AgentMetadata

__all__ = [
    "AgentMetadata",
    "AgentRegistry",
    "BaseReviewAgent",
    "CodeContext",
    "CodeFix",
    "Commit",
    "Finding",
    "PRMetadata",
    "Token",
    "registry",
]

__version__ = "0.1.0"
