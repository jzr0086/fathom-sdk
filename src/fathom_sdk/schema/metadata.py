"""AgentMetadata — describes an agent's capabilities and cost profile."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentMetadata(BaseModel):
    """Metadata every agent must provide for routing and documentation."""

    name: str
    version: str
    languages: list[str] = Field(
        description='Language identifiers, or ["*"] for language-agnostic agents'
    )
    domains: list[str] = Field(description="Domain clusters this agent covers")
    methodology: str = Field(description="Methodology cluster")
    axis_type: Literal["agnostic", "aware", "critical"]
    tags: list[str] = Field(default_factory=list)
    model_required: bool = False
    estimated_cost_cents: float = Field(ge=0.0, default=0.0)
