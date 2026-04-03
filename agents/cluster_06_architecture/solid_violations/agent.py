"""SOLID Violations Detector — flags classes that violate SOLID design principles.

Checks:
- SRP (Single Responsibility Principle): class with >10 methods likely has too
  many responsibilities.
- DIP (Dependency Inversion Principle): constructor with >5 concrete parameters
  suggests tight coupling to implementations rather than abstractions.
- ISP (Interface Segregation Principle): interface/class with >7 methods may
  be a "fat interface" that forces implementers to depend on methods they don't use.
"""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import (
    find_class_definitions,
    find_functions,
    get_node_text,
    walk,
)
from fathom_sdk.context.ast_parser import parse

_SRP_METHOD_THRESHOLD = 10
_DIP_PARAM_THRESHOLD = 5
_ISP_METHOD_THRESHOLD = 7


class SolidViolationsAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="solid_violations",
            version="0.1.0",
            languages=["*"],
            domains=["web_development", "enterprise_engineering", "api_integration"],
            methodology="architecture",
            axis_type="agnostic",
            tags=["architecture", "solid", "design"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []

        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        findings: list[Finding] = []
        classes = find_class_definitions(root, context.language)

        for cls in classes:
            # --- SRP: too many methods ---
            if cls.method_count > _SRP_METHOD_THRESHOLD:
                findings.append(
                    Finding(
                        agent_name="solid_violations",
                        severity="high",
                        category="architecture",
                        title=(
                            f"SRP violation: class '{cls.name}' has "
                            f"{cls.method_count} methods"
                        ),
                        description=(
                            f"Class '{cls.name}' defines {cls.method_count} methods "
                            f"(threshold: {_SRP_METHOD_THRESHOLD}). A class with this "
                            f"many methods likely has multiple responsibilities. "
                            f"Consider splitting it into focused, single-purpose classes."
                        ),
                        file_path=context.file_path,
                        line_start=cls.start_line,
                        line_end=cls.end_line,
                        confidence=0.75,
                        tags=["architecture", "solid", "srp"],
                    )
                )

            # --- ISP: fat interface ---
            if cls.method_count > _ISP_METHOD_THRESHOLD:
                findings.append(
                    Finding(
                        agent_name="solid_violations",
                        severity="medium",
                        category="architecture",
                        title=(
                            f"ISP violation: '{cls.name}' exposes "
                            f"{cls.method_count} methods"
                        ),
                        description=(
                            f"Class/interface '{cls.name}' has {cls.method_count} "
                            f"methods (threshold: {_ISP_METHOD_THRESHOLD}). Clients "
                            f"may be forced to depend on methods they do not use. "
                            f"Consider splitting into smaller, role-specific interfaces."
                        ),
                        file_path=context.file_path,
                        line_start=cls.start_line,
                        line_end=cls.end_line,
                        confidence=0.70,
                        tags=["architecture", "solid", "isp"],
                    )
                )

            # --- DIP: constructor with too many concrete parameters ---
            self._check_dip(cls, context, findings)

        return findings

    def _check_dip(
        self,
        cls: object,
        context: CodeContext,
        findings: list[Finding],
    ) -> None:
        """Check for constructors with too many concrete parameters."""
        lang = context.language.lower()

        for func in find_functions(cls.node, lang):
            if not self._is_constructor(func, lang):
                continue
            if func.param_count > _DIP_PARAM_THRESHOLD:
                findings.append(
                    Finding(
                        agent_name="solid_violations",
                        severity="medium",
                        category="architecture",
                        title=(
                            f"DIP violation: '{cls.name}' constructor has "
                            f"{func.param_count} parameters"
                        ),
                        description=(
                            f"The constructor of '{cls.name}' accepts "
                            f"{func.param_count} parameters "
                            f"(threshold: {_DIP_PARAM_THRESHOLD}). Many concrete "
                            f"constructor parameters suggest tight coupling. "
                            f"Consider depending on abstractions or using a "
                            f"builder/factory pattern."
                        ),
                        file_path=context.file_path,
                        line_start=func.start_line,
                        line_end=func.end_line,
                        confidence=0.70,
                        tags=["architecture", "solid", "dip"],
                    )
                )

    @staticmethod
    def _is_constructor(func: object, language: str) -> bool:
        """Return True if the function represents a constructor."""
        if language == "python":
            return func.name == "__init__"
        if language in ("javascript", "typescript"):
            return func.name == "constructor"
        if language == "java":
            # tree-sitter classifies Java constructors as
            # "constructor_declaration" nodes.
            return func.node.type == "constructor_declaration"
        # Default heuristic: common constructor name patterns
        return func.name in ("__init__", "constructor", "init", "Init", "New")

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, finding: Finding) -> str:
        if "srp" in finding.tags:
            return (
                f"{finding.title}. The Single Responsibility Principle states that a "
                f"class should have only one reason to change. Extract cohesive groups "
                f"of methods into their own classes."
            )
        if "dip" in finding.tags:
            return (
                f"{finding.title}. The Dependency Inversion Principle recommends "
                f"depending on abstractions rather than concrete implementations. "
                f"Use interfaces, protocols, or abstract base classes to decouple "
                f"the constructor from its dependencies."
            )
        if "isp" in finding.tags:
            return (
                f"{finding.title}. The Interface Segregation Principle states that "
                f"clients should not be forced to depend on methods they do not use. "
                f"Split large interfaces into smaller, focused ones."
            )
        return f"{finding.title}. Review the class design for SOLID compliance."
