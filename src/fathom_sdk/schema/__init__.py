"""fathom_sdk.schema — Pydantic models shared across the entire system."""

from fathom_sdk.schema.finding import CodeFix, Finding
from fathom_sdk.schema.metadata import AgentMetadata

__all__ = ["AgentMetadata", "CodeFix", "Finding"]
