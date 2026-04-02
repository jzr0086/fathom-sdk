"""Evaluation harness — benchmark agents against labeled datasets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from fathom_sdk.agent.base import BaseReviewAgent
from fathom_sdk.context.code_context import CodeContext

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single labeled sample from a benchmark dataset."""

    source_code: str
    language: str
    file_path: str
    expected_findings: list[dict]


@dataclass
class BenchmarkResult:
    """Aggregated metrics for one agent on one dataset."""

    agent_name: str
    dataset: str
    total_samples: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float


class EvaluationHarness:
    """Runs agents against benchmark datasets and computes metrics."""

    def __init__(self, agents: list[BaseReviewAgent] | None = None) -> None:
        self.agents = agents or []

    def load_dataset(self, path: str) -> list[BenchmarkSample]:
        """Load a benchmark dataset from JSON-lines format.

        Each line is a JSON object with keys: source_code, language,
        file_path, expected_findings.
        """
        samples: list[BenchmarkSample] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                samples.append(
                    BenchmarkSample(
                        source_code=data["source_code"],
                        language=data["language"],
                        file_path=data.get("file_path", "unknown.py"),
                        expected_findings=data.get("expected_findings", []),
                    )
                )
        return samples

    async def evaluate(
        self,
        agent: BaseReviewAgent,
        samples: list[BenchmarkSample],
        dataset_name: str = "unknown",
    ) -> BenchmarkResult:
        """Run agent against all samples and compute precision/recall."""
        tp = fp = fn = 0
        confidences: list[float] = []

        for sample in samples:
            context = CodeContext(
                source_code=sample.source_code,
                language=sample.language,
                file_path=sample.file_path,
            )
            try:
                findings = agent.analyze(context)
            except Exception:
                logger.exception(
                    "Agent %s failed on %s", agent.metadata().name, sample.file_path,
                )
                fn += len(sample.expected_findings)
                continue

            matched_expected: set[int] = set()
            matched_actual: set[int] = set()

            for i, finding in enumerate(findings):
                confidences.append(finding.confidence)
                for j, expected in enumerate(sample.expected_findings):
                    if j in matched_expected:
                        continue
                    if self._matches(finding, expected):
                        matched_expected.add(j)
                        matched_actual.add(i)
                        break

            tp += len(matched_actual)
            fp += len(findings) - len(matched_actual)
            fn += len(sample.expected_findings) - len(matched_expected)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return BenchmarkResult(
            agent_name=agent.metadata().name,
            dataset=dataset_name,
            total_samples=len(samples),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        )

    def passes_quality_gate(
        self, result: BenchmarkResult, min_precision: float = 0.85,
    ) -> bool:
        """Check if agent meets minimum precision threshold."""
        return result.precision >= min_precision

    @staticmethod
    def _matches(finding: object, expected: dict) -> bool:
        """Check if a finding matches an expected finding.

        Match criteria: same category and overlapping line range.
        """
        f_cat = getattr(finding, "category", None)
        f_start = getattr(finding, "line_start", None)
        f_end = getattr(finding, "line_end", None)

        e_cat = expected.get("category")
        e_start = expected.get("line_start", 0)
        e_end = expected.get("line_end", e_start)

        if f_cat != e_cat:
            return False

        if f_start is not None and f_end is not None:
            return f_start <= e_end and f_end >= e_start

        return True
