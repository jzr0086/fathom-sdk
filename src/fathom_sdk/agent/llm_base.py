"""BaseLLMAgent — base class for agents that use LLM APIs (Anthropic Claude).

Set FATHOM_LLM_MOCK=1 to enable mock mode for testing without API calls.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from fathom_sdk.agent.base import BaseReviewAgent
from fathom_sdk.schema.finding import CodeFix, Finding

if TYPE_CHECKING:
    from fathom_sdk.context.code_context import CodeContext

_MOCK_MODE = os.environ.get("FATHOM_LLM_MOCK", "0") == "1"
_DEFAULT_MODEL = "claude-sonnet-4-20250514"


class BaseLLMAgent(BaseReviewAgent):
    """Base class for LLM-backed review agents.

    Wraps the Anthropic API with mock support for testing.
    Subclasses implement ``_build_prompts`` and ``_parse_response``.
    """

    def __init__(self, *, model: str = _DEFAULT_MODEL, max_tokens: int = 1024) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = None  # lazy

    def _get_client(self):
        """Lazily initialise the Anthropic client."""
        if self._client is not None:
            return self._client
        try:
            import anthropic

            self._client = anthropic.Anthropic()
        except ImportError:
            return None
        return self._client

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str | None:
        """Call the LLM and return the text response.

        Returns a deterministic mock response when FATHOM_LLM_MOCK=1 or when
        the ``anthropic`` package is not installed.
        """
        if _MOCK_MODE:
            return self._mock_response(system_prompt, user_prompt)

        client = self._get_client()
        if client is None:
            return self._mock_response(system_prompt, user_prompt)

        message = client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if message.content and len(message.content) > 0:
            return message.content[0].text
        return None

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """Return a deterministic mock response for testing.

        Subclasses may override for richer mock behaviour.
        """
        return "[mock] LLM analysis not available in mock mode."

    def _build_prompts(
        self, context: "CodeContext", findings: list[Finding] | None = None
    ) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for the LLM call.

        Subclasses MUST override this.
        """
        raise NotImplementedError

    def _parse_response(
        self,
        response: str,
        context: "CodeContext",
        findings: list[Finding] | None = None,
    ) -> list[Finding]:
        """Parse the LLM response into findings.

        Subclasses MUST override this.
        """
        raise NotImplementedError

    def analyze(self, context: "CodeContext") -> list[Finding]:
        if not context.source_code.strip():
            return []
        try:
            system_prompt, user_prompt = self._build_prompts(context)
            response = self._call_llm(system_prompt, user_prompt)
            if response is None:
                return []
            return self._parse_response(response, context)
        except Exception:
            return []

    def explain(self, finding: Finding) -> str:
        return finding.description

    def suggest_fix(self, finding: Finding) -> Optional[CodeFix]:
        return finding.suggested_fix
