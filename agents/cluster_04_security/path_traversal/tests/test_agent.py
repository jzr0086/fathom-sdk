"""Tests for PathTraversalAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.path_traversal.agent import PathTraversalAgent


@pytest.fixture
def agent() -> PathTraversalAgent:
    return PathTraversalAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ---- Positive cases (should produce at least one finding) --------------------


class TestPositiveCases:
    def test_open_with_user_input(self, agent: PathTraversalAgent) -> None:
        src = "filename = request.args.get('file')\nf = open(filename)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_path_join_with_request_param(self, agent: PathTraversalAgent) -> None:
        src = "name = request.form.get('name')\npath = os.path.join('/uploads', name)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_pathlib_with_user_input(self, agent: PathTraversalAgent) -> None:
        src = "fname = request.args.get('f')\np = pathlib.Path(fname)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_literal_traversal_in_open(self, agent: PathTraversalAgent) -> None:
        src = "f = open('../../etc/passwd')\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_literal_traversal_in_path_join(self, agent: PathTraversalAgent) -> None:
        src = "path = os.path.join(base, '../secret')\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_js_readfile_with_req_params(self, agent: PathTraversalAgent) -> None:
        src = "const file = req.params.filename;\nfs.readFile(file, cb);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "test.js"))) >= 1

    def test_js_createreadstream_with_req_query(self, agent: PathTraversalAgent) -> None:
        src = "const p = req.query.path;\nfs.createReadStream(p);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "test.js"))) >= 1

    def test_java_fileinputstream_with_getparameter(self, agent: PathTraversalAgent) -> None:
        src = (
            "public class T {\n"
            "    void f(HttpServletRequest req) {\n"
            "        String name = req.getParameter(\"file\");\n"
            "        FileInputStream fis = new FileInputStream(name);\n"
            "    }\n"
            "}\n"
        )
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_open_with_formvalue(self, agent: PathTraversalAgent) -> None:
        src = (
            "package main\n\n"
            "func handler(w http.ResponseWriter, r *http.Request) {\n"
            "    name := r.FormValue(\"file\")\n"
            "    f, _ := os.Open(name)\n"
            "}\n"
        )
        assert len(agent.analyze(_ctx(src, "go", "main.go"))) >= 1

    def test_severity_high(self, agent: PathTraversalAgent) -> None:
        src = "filename = request.args.get('file')\nf = open(filename)\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].severity == "high"

    def test_sys_argv_in_open(self, agent: PathTraversalAgent) -> None:
        src = "f = open(sys.argv[1])\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_input_in_pathlib(self, agent: PathTraversalAgent) -> None:
        src = "name = input('filename: ')\np = Path(name)\n"
        assert len(agent.analyze(_ctx(src))) >= 1


# ---- Negative cases (should produce zero findings) --------------------------


class TestNegativeCases:
    def test_hardcoded_path(self, agent: PathTraversalAgent) -> None:
        src = "f = open('/etc/config.json')\n"
        assert agent.analyze(_ctx(src)) == []

    def test_sanitized_with_basename(self, agent: PathTraversalAgent) -> None:
        src = (
            "filename = request.args.get('file')\n"
            "safe = os.path.basename(filename)\n"
            "f = open(safe)\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_sanitized_with_realpath(self, agent: PathTraversalAgent) -> None:
        src = (
            "name = request.form.get('name')\n"
            "safe = os.path.realpath(name)\n"
            "f = open(safe)\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_no_file_io(self, agent: PathTraversalAgent) -> None:
        src = "x = 1\ny = x + 2\nprint(y)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: PathTraversalAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_comment_only(self, agent: PathTraversalAgent) -> None:
        src = "# f = open(request.args.get('file'))\n"
        assert agent.analyze(_ctx(src)) == []

    def test_no_user_input(self, agent: PathTraversalAgent) -> None:
        src = "import json\nwith open('data.json') as f:\n    data = json.load(f)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_constant_path_join(self, agent: PathTraversalAgent) -> None:
        src = "path = os.path.join('/app', 'static', 'logo.png')\n"
        assert agent.analyze(_ctx(src)) == []

    def test_sanitized_node_resolve(self, agent: PathTraversalAgent) -> None:
        src = (
            "const p = req.query.path;\n"
            "const safe = path.resolve('/uploads', p);\n"
        )
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_sanitized_go_filepath_clean(self, agent: PathTraversalAgent) -> None:
        src = (
            "package main\n\n"
            "func handler(w http.ResponseWriter, r *http.Request) {\n"
            "    name := r.FormValue(\"file\")\n"
            "    safe := filepath.Clean(name)\n"
            "    f, _ := os.Open(safe)\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src, "go", "main.go")) == []

    def test_print_user_input(self, agent: PathTraversalAgent) -> None:
        src = "name = request.args.get('name')\nprint(name)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_whitespace_only(self, agent: PathTraversalAgent) -> None:
        assert agent.analyze(_ctx("   \n  \n")) == []


# ---- Edge cases --------------------------------------------------------------


class TestEdgeCases:
    def test_metadata_name(self, agent: PathTraversalAgent) -> None:
        m = agent.metadata()
        assert m.name == "path_traversal"

    def test_metadata_tags(self, agent: PathTraversalAgent) -> None:
        m = agent.metadata()
        assert "path-traversal" in m.tags
        assert "security" in m.tags

    def test_metadata_languages(self, agent: PathTraversalAgent) -> None:
        m = agent.metadata()
        assert "python" in m.languages
        assert "javascript" in m.languages

    def test_unsupported_language(self, agent: PathTraversalAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_explain_returns_string(self, agent: PathTraversalAgent) -> None:
        src = "filename = request.args.get('file')\nf = open(filename)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert "path traversal" in explanation.lower()

    def test_no_duplicate_findings(self, agent: PathTraversalAgent) -> None:
        src = "filename = request.args.get('file')\nf = open(filename)\n"
        findings = agent.analyze(_ctx(src))
        lines = [f.line_start for f in findings]
        assert len(lines) == len(set(lines))
