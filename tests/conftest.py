"""Shared pytest fixtures for Fathom SDK tests."""

from __future__ import annotations

import pytest

from fathom_sdk.agent.registry import AgentRegistry, registry
from fathom_sdk.context.code_context import CodeContext
from fathom_sdk.schema.finding import Finding

SAMPLE_PYTHON_CODE = '''\
def greet(name):
    """Say hello."""
    if name is None:
        return "Hello, stranger!"
    return f"Hello, {name}!"


def add(a, b):
    return a + b


result = add(1, 2)
print(greet(result))
'''


@pytest.fixture
def sample_context() -> CodeContext:
    """A simple CodeContext with Python source for testing."""
    return CodeContext(
        source_code=SAMPLE_PYTHON_CODE,
        language="python",
        file_path="example.py",
    )


@pytest.fixture
def sample_finding() -> Finding:
    """A sample Finding for testing."""
    return Finding(
        agent_name="test_agent",
        severity="medium",
        category="bug",
        title="Possible null dereference",
        description="Variable may be None when accessed.",
        file_path="example.py",
        line_start=3,
        line_end=4,
        confidence=0.85,
        fix_available=False,
        tags=["null", "safety"],
    )


@pytest.fixture
def agent_registry() -> AgentRegistry:
    """A fresh registry with discovered agents."""
    reg = AgentRegistry()
    reg.discover()
    return reg


@pytest.fixture
def all_registered_agents(agent_registry: AgentRegistry) -> list:
    """All currently registered agent classes."""
    return list(agent_registry.get_all().values())
