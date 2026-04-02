"""Finding and CodeFix — the core output types of every review agent."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CodeFix(BaseModel):
    """A suggested fix for a finding."""

    description: str
    original_code: str
    fixed_code: str
    explanation: str


class Finding(BaseModel):
    """A single issue discovered by a review agent."""

    agent_name: str
    severity: Literal["critical", "high", "medium", "low", "info"]
    category: str = Field(description="e.g. 'security', 'performance', 'bug'")
    title: str
    description: str
    file_path: str
    line_start: int = Field(ge=0)
    line_end: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    fix_available: bool = False
    tags: list[str] = Field(default_factory=list)
    suggested_fix: Optional[CodeFix] = None
