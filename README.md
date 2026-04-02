# Fathom SDK

SDK for building [Fathom](https://github.com/fathom-team/fathom) code review agents.

Fathom is an ML-powered multi-agent code review system. This package provides
the base interfaces, data models, context utilities, and all open-source agents.

## Installation

```bash
pip install fathom-sdk
```

For LLM-backed agents:

```bash
pip install fathom-sdk[llm]
```

## Quick Start

```python
from fathom_sdk import BaseReviewAgent, Finding, CodeContext, AgentMetadata

class MyAgent(BaseReviewAgent):
    def analyze(self, context: CodeContext) -> list[Finding]:
        findings = []
        # Your analysis logic here
        return findings

    def explain(self, finding: Finding) -> str:
        return f"Found: {finding.title}"

    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="my-agent",
            version="0.1.0",
            languages=["python"],
            domains=["general"],
            methodology="static_analysis",
            axis_type="aware",
            tags=["custom"],
        )
```

## Running Agents Standalone

```python
from fathom_sdk.runner import run_agents

results = run_agents(
    source_code=open("my_file.py").read(),
    file_path="my_file.py",
    language="python",
)
for finding in results:
    print(f"[{finding.severity}] {finding.title} ({finding.file_path}:{finding.line_start})")
```

## Development

```bash
uv sync --dev
uv run pytest
```

## Adding an Agent

1. Create a directory under `agents/` in the appropriate cluster
2. Implement `BaseReviewAgent` in `agent.py`
3. Register via entry points in `pyproject.toml`
4. Write tests with 10+ positive and 10+ negative examples
5. Benchmark against labeled datasets — target precision > 0.85

See [Adding an Agent](docs/adding_an_agent.md) for the full guide.

## License

Apache-2.0
