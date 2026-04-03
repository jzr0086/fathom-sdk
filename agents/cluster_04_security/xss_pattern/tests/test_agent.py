"""Tests for XssPatternAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.xss_pattern.agent import XssPatternAgent


@pytest.fixture
def agent() -> XssPatternAgent:
    return XssPatternAgent()


def _ctx(source: str, language: str = "javascript", fp: str = "test.js") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_innerhtml(self, agent: XssPatternAgent) -> None:
        assert len(agent.analyze(_ctx("el.innerHTML = userInput;\n"))) >= 1

    def test_dangerously_set(self, agent: XssPatternAgent) -> None:
        src = "<div dangerouslySetInnerHTML={{__html: data}} />\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_document_write(self, agent: XssPatternAgent) -> None:
        assert len(agent.analyze(_ctx("document.write(userInput);\n"))) >= 1

    def test_jinja_safe(self, agent: XssPatternAgent) -> None:
        src = "{{ user_input | safe }}\n"
        ctx = CodeContext(source_code=src, language="python", file_path="t.html")
        assert len(agent.analyze(ctx)) >= 1

    def test_vue_vhtml(self, agent: XssPatternAgent) -> None:
        src = '<div v-html="userContent"></div>\n'
        ctx = CodeContext(source_code=src, language="javascript", file_path="t.vue")
        assert len(agent.analyze(ctx)) >= 1

    def test_angular_innerhtml(self, agent: XssPatternAgent) -> None:
        src = '<div [innerHTML]="content"></div>\n'
        ctx = CodeContext(source_code=src, language="typescript", file_path="t.ts")
        assert len(agent.analyze(ctx)) >= 1

    def test_mark_safe(self, agent: XssPatternAgent) -> None:
        src = "output = mark_safe(user_input)\n"
        ctx = CodeContext(source_code=src, language="python", file_path="t.py")
        assert len(agent.analyze(ctx)) >= 1

    def test_outerhtml(self, agent: XssPatternAgent) -> None:
        assert len(agent.analyze(_ctx("el.outerHTML = data;\n"))) >= 1

    def test_insert_adjacent_html(self, agent: XssPatternAgent) -> None:
        assert len(agent.analyze(_ctx("el.insertAdjacentHTML('beforeend', data);\n"))) >= 1

    def test_go_template_html(self, agent: XssPatternAgent) -> None:
        src = "package main\n\nfunc f() {\n    t := template.HTML(input)\n}\n"
        ctx = CodeContext(source_code=src, language="go", file_path="t.go")
        assert len(agent.analyze(ctx)) >= 1


class TestNegativeCases:
    def test_textcontent(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("el.textContent = userInput;\n")) == []

    def test_innertext(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("el.innerText = data;\n")) == []

    def test_react_jsx(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("return <div>{userInput}</div>;\n")) == []

    def test_escaped_output(self, agent: XssPatternAgent) -> None:
        src = "{{ user_input }}\n"  # no |safe, auto-escaped
        ctx = CodeContext(source_code=src, language="python", file_path="t.html")
        assert agent.analyze(ctx) == []

    def test_no_dom(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("let x = 1;\n")) == []

    def test_comment(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("// el.innerHTML = data;\n")) == []

    def test_empty(self, agent: XssPatternAgent) -> None:
        ctx = CodeContext(source_code="", language="javascript", file_path="t.js")
        assert agent.analyze(ctx) == []

    def test_createelement(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("const el = document.createElement('div');\n")) == []

    def test_appendchild(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("parent.appendChild(child);\n")) == []

    def test_classname(self, agent: XssPatternAgent) -> None:
        assert agent.analyze(_ctx("el.className = 'active';\n")) == []


class TestEdgeCases:
    def test_metadata(self, agent: XssPatternAgent) -> None:
        m = agent.metadata()
        assert m.name == "xss_pattern"
        assert "xss" in m.tags
