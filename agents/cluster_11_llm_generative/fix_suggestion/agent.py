"""Fix Suggestion Generator Agent — uses an LLM to produce concrete code fixes.

This agent takes existing findings (from ``context.historical_findings``) along
with the source code and asks Claude to suggest concrete fixes for each issue.
When no prior findings are available it falls back to a general code review
that looks for fixable issues.

Set FATHOM_LLM_MOCK=1 to enable deterministic mock mode for testing.
"""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, CodeContext, CodeFix, Finding
from fathom_sdk.agent.llm_base import BaseLLMAgent

_SYSTEM_PROMPT = (
    "You are a code repair expert. Given the following code and issues found, "
    "suggest concrete fixes for each issue.\n\n"
    "Return your suggestions as a numbered list. Each item MUST follow this "
    "exact format:\n\n"
    "1. **Title of fix**: Description of what to change and why.\n"
    "ORIGINAL:\n"
    "```\n"
    "<original code snippet>\n"
    "```\n"
    "FIXED:\n"
    "```\n"
    "<fixed code snippet>\n"
    "```\n"
    "EXPLANATION: A brief explanation of why this fix is correct.\n\n"
    "Only suggest fixes you are confident about. If the code is clean, "
    "return an empty list."
)

_FIX_BLOCK_RE = re.compile(
    r"\d+\.\s+\*\*(.+?)\*\*:\s*(.+?)\s*\n"
    r"ORIGINAL:\s*\n```\n?(.*?)\n?```\s*\n"
    r"FIXED:\s*\n```\n?(.*?)\n?```\s*\n"
    r"EXPLANATION:\s*(.+?)(?=\n\d+\.\s+\*\*|\Z)",
    re.DOTALL,
)


class FixSuggestionAgent(BaseLLMAgent):
    """LLM-backed agent that generates concrete fix suggestions for code issues."""

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="fix_suggestion",
            version="0.1.0",
            languages=["*"],
            domains=["web_development", "api_integration", "enterprise_engineering"],
            methodology="llm_generative",
            axis_type="agnostic",
            tags=["llm", "fix", "generative"],
            model_required=True,
            estimated_cost_cents=8.0,
        )

    # -- BaseLLMAgent hooks --------------------------------------------------

    def _build_prompts(
        self,
        context: CodeContext,
        findings: list[Finding] | None = None,
    ) -> tuple[str, str]:
        parts = [
            f"File: {context.file_path}  (language: {context.language})\n",
            f"```\n{context.source_code}\n```\n",
        ]

        if findings:
            parts.append("Known issues to fix:\n")
            for i, f in enumerate(findings, 1):
                parts.append(
                    f"{i}. [{f.severity}] {f.title} (line {f.line_start}): "
                    f"{f.description}\n"
                )
        else:
            parts.append(
                "No prior issues were flagged. Review the code for any fixable "
                "problems related to correctness, security, performance, or "
                "best practices and suggest concrete fixes.\n"
            )

        return _SYSTEM_PROMPT, "".join(parts)

    def _parse_response(
        self,
        response: str,
        context: CodeContext,
        findings: list[Finding] | None = None,
    ) -> list[Finding]:
        results: list[Finding] = []
        for match in _FIX_BLOCK_RE.finditer(response):
            title = match.group(1).strip()
            description = match.group(2).strip()
            original_code = match.group(3).strip()
            fixed_code = match.group(4).strip()
            explanation = match.group(5).strip()

            results.append(
                Finding(
                    agent_name="fix_suggestion",
                    severity="info",
                    category="fix",
                    title=title,
                    description=description,
                    file_path=context.file_path,
                    line_start=1,
                    line_end=max(1, len(context.source_code.splitlines())),
                    confidence=0.60,
                    fix_available=True,
                    tags=["llm", "fix", "generative"],
                    suggested_fix=CodeFix(
                        description=title,
                        original_code=original_code,
                        fixed_code=fixed_code,
                        explanation=explanation,
                    ),
                )
            )
        return results

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        """Return a deterministic mock response that ``_parse_response`` can handle."""
        return (
            "1. **Add input validation**: The function processes user input "
            "without checking its type or bounds.\n"
            "ORIGINAL:\n"
            "```\n"
            "def process(data):\n"
            "    return data * 2\n"
            "```\n"
            "FIXED:\n"
            "```\n"
            "def process(data):\n"
            "    if not isinstance(data, (int, float)):\n"
            '        raise TypeError("data must be numeric")\n'
            "    return data * 2\n"
            "```\n"
            "EXPLANATION: Adding a type check prevents unexpected behaviour when "
            "non-numeric values are passed to the function.\n"
            "2. **Use parameterised query**: Building SQL with string concatenation "
            "is vulnerable to injection.\n"
            "ORIGINAL:\n"
            "```\n"
            'query = "SELECT * FROM users WHERE id=" + user_id\n'
            "```\n"
            "FIXED:\n"
            "```\n"
            'query = "SELECT * FROM users WHERE id=%s"\n'
            "cursor.execute(query, (user_id,))\n"
            "```\n"
            "EXPLANATION: Parameterised queries prevent SQL injection by separating "
            "data from the query structure.\n"
            "3. **Close file handle**: The file is opened but never explicitly closed, "
            "risking resource leaks.\n"
            "ORIGINAL:\n"
            "```\n"
            'f = open("data.txt")\n'
            "contents = f.read()\n"
            "```\n"
            "FIXED:\n"
            "```\n"
            'with open("data.txt") as f:\n'
            "    contents = f.read()\n"
            "```\n"
            "EXPLANATION: Using a context manager ensures the file handle is closed "
            "even if an exception occurs."
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        """Analyze code and generate fix suggestions.

        Uses ``context.historical_findings`` as the input findings when
        available.  Falls back to a general review when none are present.
        """
        if not context.source_code.strip():
            return []
        try:
            prior_findings = context.historical_findings or None
            system_prompt, user_prompt = self._build_prompts(
                context, findings=prior_findings
            )
            response = self._call_llm(system_prompt, user_prompt)
            if response is None:
                return []
            return self._parse_response(response, context, findings=prior_findings)
        except Exception:
            return []
