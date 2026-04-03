"""Review Generator Agent — uses an LLM to produce high-level code review findings.

This agent sends the source code to Claude and asks for the top 3-5 most
important issues.  It is language-agnostic and works across all domains.
"""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, CodeContext, Finding
from fathom_sdk.agent.llm_base import BaseLLMAgent

_SYSTEM_PROMPT = (
    "You are a senior code reviewer. Analyze the following code and identify "
    "the top 3-5 most important issues regarding correctness, security, "
    "performance, maintainability, or best-practice violations.\n\n"
    "Return your findings as a numbered list. Each item MUST follow this "
    "exact format:\n\n"
    "1. **Title of issue**: Description explaining the problem and why it matters.\n"
    "2. **Title of issue**: Description explaining the problem and why it matters.\n\n"
    "Only report genuine concerns. If the code is clean, return an empty list."
)

_ITEM_RE = re.compile(
    r"^\s*\d+\.\s+\*\*(.+?)\*\*:\s*(.+)",
    re.MULTILINE,
)


class ReviewGeneratorAgent(BaseLLMAgent):
    """LLM-backed agent that produces a high-level code review."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="review_generator",
            version="0.1.0",
            languages=["*"],
            domains=["web_development", "api_integration", "enterprise_engineering"],
            methodology="llm_generative",
            axis_type="agnostic",
            tags=["llm", "review", "generative"],
            model_required=True,
            estimated_cost_cents=5.0,
        )

    # -- BaseLLMAgent hooks --------------------------------------------------

    def _build_prompts(
        self,
        context: CodeContext,
        findings: list[Finding] | None = None,
    ) -> tuple[str, str]:
        user_prompt = (
            f"File: {context.file_path}  (language: {context.language})\n\n"
            f"```\n{context.source_code}\n```"
        )
        return _SYSTEM_PROMPT, user_prompt

    def _parse_response(
        self,
        response: str,
        context: CodeContext,
        findings: list[Finding] | None = None,
    ) -> list[Finding]:
        results: list[Finding] = []
        for match in _ITEM_RE.finditer(response):
            title = match.group(1).strip()
            description = match.group(2).strip()
            results.append(
                Finding(
                    agent_name="review_generator",
                    severity="medium",
                    category="quality",
                    title=title,
                    description=description,
                    file_path=context.file_path,
                    line_start=1,
                    line_end=max(1, len(context.source_code.splitlines())),
                    confidence=0.65,
                    tags=["llm", "review", "generative"],
                )
            )
        return results

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """Return a deterministic mock response that ``_parse_response`` can handle."""
        return (
            "1. **Missing input validation**: The function accepts external input "
            "without any sanitization or type checking, which may lead to "
            "unexpected runtime errors.\n"
            "2. **No error handling**: There are no try/except blocks around "
            "operations that could raise exceptions, making the code fragile.\n"
            "3. **Hard-coded configuration values**: Configuration values are "
            "embedded directly in the source instead of being read from "
            "environment variables or a config file."
        )
