"""Agent registry — registration, discovery, and filtering."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Optional

from fathom_sdk.agent.base import BaseReviewAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Singleton registry of all available review agents."""

    def __init__(self) -> None:
        self._agents: dict[str, type[BaseReviewAgent]] = {}

    def register(self, agent_class: type[BaseReviewAgent]) -> type[BaseReviewAgent]:
        """Register an agent class.  Can be used as a decorator."""
        meta = agent_class().metadata()
        self._agents[meta.name] = agent_class
        logger.debug("Registered agent: %s (v%s)", meta.name, meta.version)
        return agent_class

    def discover(self) -> None:
        """Discover agents via the ``fathom.agents`` entry-point group."""
        try:
            eps = importlib.metadata.entry_points(group="fathom.agents")
        except TypeError:
            eps = importlib.metadata.entry_points().get("fathom.agents", [])

        for ep in eps:
            try:
                agent_class = ep.load()
                if isinstance(agent_class, type) and issubclass(
                    agent_class, BaseReviewAgent
                ):
                    self.register(agent_class)
            except Exception:
                logger.warning("Failed to load agent entry point: %s", ep.name, exc_info=True)

    def get(self, name: str) -> Optional[type[BaseReviewAgent]]:
        """Retrieve an agent class by name."""
        return self._agents.get(name)

    def get_all(self) -> dict[str, type[BaseReviewAgent]]:
        """Return all registered agents."""
        return dict(self._agents)

    def filter_by(
        self,
        *,
        languages: Optional[list[str]] = None,
        domains: Optional[list[str]] = None,
        methodology: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[type[BaseReviewAgent]]:
        """Filter agents by metadata fields."""
        results: list[type[BaseReviewAgent]] = []
        for agent_class in self._agents.values():
            meta = agent_class().metadata()

            if languages and "*" not in meta.languages:
                if not set(languages) & set(meta.languages):
                    continue
            if domains and not set(domains) & set(meta.domains):
                continue
            if methodology and meta.methodology != methodology:
                continue
            if tags and not set(tags) & set(meta.tags):
                continue

            results.append(agent_class)
        return results


registry = AgentRegistry()
