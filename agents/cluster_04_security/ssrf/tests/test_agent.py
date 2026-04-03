"""Tests for SsrfAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.ssrf.agent import SsrfAgent


@pytest.fixture
def agent() -> SsrfAgent:
    return SsrfAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# -----------------------------------------------------------------------
# Positive cases (10+)
# -----------------------------------------------------------------------


class TestPositiveCases:
    def test_python_requests_get_fstring(self, agent: SsrfAgent) -> None:
        src = 'requests.get(f"http://{user_input}/api")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_requests_post_fstring(self, agent: SsrfAgent) -> None:
        src = 'requests.post(f"https://{host}/endpoint")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_requests_get_format(self, agent: SsrfAgent) -> None:
        src = 'requests.get("http://{}/api".format(user_url))\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_requests_get_concat(self, agent: SsrfAgent) -> None:
        src = 'requests.get("http://" + user_input + "/api")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_urlopen_fstring(self, agent: SsrfAgent) -> None:
        src = 'urllib.request.urlopen(f"http://{target}/path")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_python_httpx_get_fstring(self, agent: SsrfAgent) -> None:
        src = 'httpx.get(f"https://{url}/data")\n'
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_fetch_template_literal(self, agent: SsrfAgent) -> None:
        src = "fetch(`http://${userInput}/api`);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "test.js"))) >= 1

    def test_js_axios_get_template_literal(self, agent: SsrfAgent) -> None:
        src = "axios.get(`https://${host}/endpoint`);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "test.js"))) >= 1

    def test_go_http_get_concat(self, agent: SsrfAgent) -> None:
        src = 'package main\n\nfunc f() {\n    http.Get("http://" + userInput)\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "test.go"))) >= 1

    def test_go_http_new_request_concat(self, agent: SsrfAgent) -> None:
        src = 'package main\n\nfunc f() {\n    http.NewRequest("GET", "http://" + target, nil)\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "test.go"))) >= 1

    def test_severity_is_high(self, agent: SsrfAgent) -> None:
        src = 'requests.get(f"http://{user_input}/api")\n'
        f = agent.analyze(_ctx(src))[0]
        assert f.severity == "high"

    def test_confidence_is_correct(self, agent: SsrfAgent) -> None:
        src = 'requests.get(f"http://{user_input}/api")\n'
        f = agent.analyze(_ctx(src))[0]
        assert f.confidence == pytest.approx(0.82)

    def test_js_fetch_concat(self, agent: SsrfAgent) -> None:
        src = 'fetch("http://" + userUrl);\n'
        assert len(agent.analyze(_ctx(src, "javascript", "test.js"))) >= 1


# -----------------------------------------------------------------------
# Negative cases (10+)
# -----------------------------------------------------------------------


class TestNegativeCases:
    def test_hardcoded_url(self, agent: SsrfAgent) -> None:
        src = 'requests.get("https://api.example.com/users")\n'
        assert agent.analyze(_ctx(src)) == []

    def test_no_http_calls(self, agent: SsrfAgent) -> None:
        src = "x = 1\ny = x + 2\nprint(y)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: SsrfAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: SsrfAgent) -> None:
        assert agent.analyze(_ctx("   \n\n  ")) == []

    def test_comment_line(self, agent: SsrfAgent) -> None:
        src = '# requests.get(f"http://{user_input}/api")\n'
        assert agent.analyze(_ctx(src)) == []

    def test_js_comment_line(self, agent: SsrfAgent) -> None:
        src = '// fetch(`http://${userInput}/api`);\n'
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_hardcoded_url_js(self, agent: SsrfAgent) -> None:
        src = 'fetch("https://api.example.com/data");\n'
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_no_interpolation_python(self, agent: SsrfAgent) -> None:
        src = 'requests.get(API_URL)\n'
        assert agent.analyze(_ctx(src)) == []

    def test_non_http_fstring(self, agent: SsrfAgent) -> None:
        src = 'print(f"Hello {name}")\n'
        assert agent.analyze(_ctx(src)) == []

    def test_hardcoded_url_go(self, agent: SsrfAgent) -> None:
        src = 'package main\n\nfunc f() {\n    http.Get("https://example.com/api")\n}\n'
        assert agent.analyze(_ctx(src, "go", "test.go")) == []

    def test_non_http_concat(self, agent: SsrfAgent) -> None:
        src = 'result = "prefix" + some_var\n'
        assert agent.analyze(_ctx(src)) == []

    def test_hardcoded_axios(self, agent: SsrfAgent) -> None:
        src = 'axios.get("https://api.example.com/items");\n'
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []


# -----------------------------------------------------------------------
# Edge cases & metadata
# -----------------------------------------------------------------------


class TestEdgeCases:
    def test_metadata_name(self, agent: SsrfAgent) -> None:
        m = agent.metadata()
        assert m.name == "ssrf"

    def test_metadata_version(self, agent: SsrfAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_tags(self, agent: SsrfAgent) -> None:
        m = agent.metadata()
        assert "ssrf" in m.tags
        assert "security" in m.tags

    def test_metadata_languages(self, agent: SsrfAgent) -> None:
        m = agent.metadata()
        assert "python" in m.languages
        assert "javascript" in m.languages
        assert "go" in m.languages

    def test_metadata_model_not_required(self, agent: SsrfAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False

    def test_metadata_zero_cost(self, agent: SsrfAgent) -> None:
        m = agent.metadata()
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: SsrfAgent) -> None:
        src = 'requests.get(f"http://{user_input}/api")\n'
        f = agent.analyze(_ctx(src))[0]
        explanation = agent.explain(f)
        assert isinstance(explanation, str)
        assert "SSRF" in explanation

    def test_unsupported_language(self, agent: SsrfAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_no_duplicate_findings(self, agent: SsrfAgent) -> None:
        src = 'requests.get(f"http://{user_input}/api")\n'
        findings = agent.analyze(_ctx(src))
        lines = [f.line_start for f in findings]
        assert len(lines) == len(set(lines))
