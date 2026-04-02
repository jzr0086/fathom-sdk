"""Lightweight standalone agent runner.

Allows running agents outside of the full Fathom orchestrator — useful for
development, testing, and quick one-off reviews.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fathom_sdk.agent.base import BaseReviewAgent
from fathom_sdk.agent.registry import AgentRegistry
from fathom_sdk.context.code_context import CodeContext
from fathom_sdk.schema.finding import Finding

logger = logging.getLogger(__name__)


def run_agents(
    source_code: str,
    file_path: str,
    language: str,
    agents: list[BaseReviewAgent] | None = None,
    discover: bool = True,
    filter_languages: list[str] | None = None,
    filter_domains: list[str] | None = None,
    filter_tags: list[str] | None = None,
) -> list[Finding]:
    """Run agents synchronously on source code and return findings.

    Parameters
    ----------
    source_code:
        The full source text to review.
    file_path:
        The file path (used for reporting and language detection).
    language:
        The programming language of the source code.
    agents:
        Specific agent instances to run. If None, discovers agents.
    discover:
        If True and agents is None, discover agents via entry points.
    filter_languages:
        When discovering, only include agents matching these languages.
    filter_domains:
        When discovering, only include agents matching these domains.
    filter_tags:
        When discovering, only include agents matching these tags.

    Returns
    -------
    list[Finding]
        All findings from all agents, sorted by severity then confidence.
    """
    context = CodeContext(
        source_code=source_code,
        language=language,
        file_path=file_path,
    )

    # Optionally parse AST
    try:
        from fathom_sdk.context.ast_parser import parse
        from fathom_sdk.context.graph_builder import (
            build_call_graph,
            build_control_flow_graph,
        )

        context.ast = parse(source_code, language)
        context.call_graph = build_call_graph(context.ast, language)
        context.control_flow_graph = build_control_flow_graph(context.ast, language)
    except (ValueError, Exception):
        logger.debug("AST parsing unavailable for %s", file_path, exc_info=True)

    if agents is None:
        agents = _discover_agents(
            discover=discover,
            languages=filter_languages,
            domains=filter_domains,
            tags=filter_tags,
        )

    all_findings: list[Finding] = []
    for agent in agents:
        meta = agent.metadata()
        try:
            findings = agent.analyze(context)
            all_findings.extend(findings)
            logger.debug("Agent %s produced %d findings", meta.name, len(findings))
        except Exception:
            logger.exception("Agent %s failed", meta.name)

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    all_findings.sort(
        key=lambda f: (severity_order.get(f.severity, 99), -f.confidence),
    )
    return all_findings


def _discover_agents(
    *,
    discover: bool,
    languages: Optional[list[str]],
    domains: Optional[list[str]],
    tags: Optional[list[str]],
) -> list[BaseReviewAgent]:
    """Discover and optionally filter agents."""
    if not discover:
        return []

    reg = AgentRegistry()
    reg.discover()

    if languages or domains or tags:
        classes = reg.filter_by(languages=languages, domains=domains, tags=tags)
    else:
        classes = list(reg.get_all().values())

    return [cls() for cls in classes]
