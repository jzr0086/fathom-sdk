"""Auth Bypass Detector — flags route/endpoint definitions missing auth decorators or middleware."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding

# ---------------------------------------------------------------------------
# Flask / generic Python decorator-based frameworks
# ---------------------------------------------------------------------------

# Matches @app.route(...) or @blueprint.route(...) with mutating methods
_FLASK_ROUTE = re.compile(
    r"""@\w+\.route\s*\(\s*['"][^'"]*['"]\s*,\s*methods\s*=\s*\[([^\]]+)\]""",
    re.IGNORECASE,
)

_MUTATING_METHODS = re.compile(
    r"""['"](?:POST|PUT|DELETE|PATCH)['"]""", re.IGNORECASE
)

# Auth decorators that indicate the route is protected
_PYTHON_AUTH_DECORATORS = re.compile(
    r"@(?:login_required|auth_required|jwt_required|requires_auth|"
    r"permission_required|roles_required|authenticated|"
    r"token_required|api_key_required|require_login)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Django URL patterns
# ---------------------------------------------------------------------------

# Matches path(..., SomeView.as_view()) or path(..., some_view)
_DJANGO_URL_PATTERN = re.compile(
    r"""(?:path|re_path|url)\s*\(\s*r?['"][^'"]*['"]""",
    re.IGNORECASE,
)

# Django auth indicators within surrounding context
_DJANGO_AUTH_INDICATORS = re.compile(
    r"(?:LoginRequiredMixin|login_required|permission_required|"
    r"PermissionRequiredMixin|UserPassesTestMixin|"
    r"IsAuthenticated|IsAdminUser|AllowAny)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Express.js patterns
# ---------------------------------------------------------------------------

# Matches router.post(...), app.put(...), router.delete(...), app.patch(...)
_EXPRESS_MUTATING = re.compile(
    r"""(?:router|app)\s*\.\s*(?:post|put|delete|patch)\s*\(""",
    re.IGNORECASE,
)

# Auth middleware keywords that would appear in the Express call chain
_EXPRESS_AUTH_MIDDLEWARE = re.compile(
    r"(?:auth|authenticate|isAuthenticated|requireAuth|ensureAuth|"
    r"passport\.\w+|verifyToken|checkAuth|requireLogin|"
    r"isLoggedIn|protect|authorize|jwt|authMiddleware|"
    r"requirePermission|checkPermission|isAuthorized)",
    re.IGNORECASE,
)

# How many lines above a route definition to search for auth decorators
_LOOKBACK_LINES = 5


class AuthBypassAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="auth_bypass",
            version="0.1.0",
            languages=["*"],
            domains=["security_engineering", "web_development", "api_integration"],
            methodology="security",
            axis_type="agnostic",
            tags=["security", "auth", "bypass"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []

        findings: list[Finding] = []
        lines = context.source_code.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            line_num = i + 1

            # Skip comments
            if stripped.startswith(("#", "//", "/*", "*")):
                continue

            # --- Flask / Python decorator-based routes ---
            route_match = _FLASK_ROUTE.search(line)
            if route_match:
                methods_str = route_match.group(1)
                if _MUTATING_METHODS.search(methods_str):
                    if not self._has_python_auth_decorator(lines, i):
                        findings.append(
                            self._make_finding(
                                title="Route with mutating method missing auth decorator",
                                description=(
                                    f"Route definition at line {line_num} accepts mutating "
                                    f"HTTP methods ({methods_str.strip()}) but has no "
                                    f"authentication decorator (e.g. @login_required, "
                                    f"@auth_required, @jwt_required)."
                                ),
                                file_path=context.file_path,
                                line_start=line_num,
                                line_end=line_num,
                            )
                        )
                continue

            # --- Django URL patterns ---
            if _DJANGO_URL_PATTERN.search(line):
                if not self._has_django_auth(lines, i):
                    findings.append(
                        self._make_finding(
                            title="Django URL pattern potentially missing auth protection",
                            description=(
                                f"URL pattern at line {line_num} does not appear to be "
                                f"protected by LoginRequiredMixin, @login_required, or "
                                f"a permissions class."
                            ),
                            file_path=context.file_path,
                            line_start=line_num,
                            line_end=line_num,
                        )
                    )
                continue

            # --- Express.js mutating routes ---
            if _EXPRESS_MUTATING.search(line):
                if not self._has_express_auth(line, lines, i):
                    findings.append(
                        self._make_finding(
                            title="Express route with mutating method missing auth middleware",
                            description=(
                                f"Express route at line {line_num} handles a mutating "
                                f"HTTP method but does not include auth middleware in "
                                f"the handler chain."
                            ),
                            file_path=context.file_path,
                            line_start=line_num,
                            line_end=line_num,
                        )
                    )

        return findings

    # ------------------------------------------------------------------
    # Helper: check for Python auth decorators above a route line
    # ------------------------------------------------------------------

    @staticmethod
    def _has_python_auth_decorator(lines: list[str], route_idx: int) -> bool:
        # Check lines above the route decorator, stopping at a function def
        # or blank line (which indicates a separate block)
        for j in range(route_idx - 1, max(-1, route_idx - _LOOKBACK_LINES - 1), -1):
            line = lines[j].strip()
            if not line or line.startswith("def ") or line.startswith("class "):
                break
            if _PYTHON_AUTH_DECORATORS.search(lines[j]):
                return True
        # Check lines below (auth decorator may be between @app.route and def)
        end = min(len(lines), route_idx + _LOOKBACK_LINES + 1)
        for j in range(route_idx + 1, end):
            line = lines[j].strip()
            if line.startswith("def ") or line.startswith("class "):
                break
            if _PYTHON_AUTH_DECORATORS.search(lines[j]):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper: check for Django auth indicators in surrounding context
    # ------------------------------------------------------------------

    @staticmethod
    def _has_django_auth(lines: list[str], url_idx: int) -> bool:
        # Check the line itself and surrounding context
        start = max(0, url_idx - _LOOKBACK_LINES)
        end = min(len(lines), url_idx + _LOOKBACK_LINES + 1)
        for j in range(start, end):
            if _DJANGO_AUTH_INDICATORS.search(lines[j]):
                return True
        return False

    # ------------------------------------------------------------------
    # Helper: check for Express auth middleware on the same line or above
    # ------------------------------------------------------------------

    @staticmethod
    def _has_express_auth(line: str, lines: list[str], route_idx: int) -> bool:
        # Check the route line itself for inline auth middleware
        if _EXPRESS_AUTH_MIDDLEWARE.search(line):
            return True
        # Check lines above for middleware declarations applied to router/app
        start = max(0, route_idx - _LOOKBACK_LINES)
        for j in range(start, route_idx):
            if _EXPRESS_AUTH_MIDDLEWARE.search(lines[j]):
                return True
        return False

    # ------------------------------------------------------------------
    # Finding factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_finding(
        title: str,
        description: str,
        file_path: str,
        line_start: int,
        line_end: int,
    ) -> Finding:
        return Finding(
            agent_name="auth_bypass",
            severity="high",
            category="security",
            title=title,
            description=description,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            confidence=0.72,
            tags=["security", "auth", "bypass"],
        )

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Routes that handle mutating HTTP methods "
            f"(POST, PUT, DELETE, PATCH) without authentication middleware "
            f"may allow unauthorized users to modify data. Add an appropriate "
            f"auth decorator or middleware to protect this endpoint."
        )
