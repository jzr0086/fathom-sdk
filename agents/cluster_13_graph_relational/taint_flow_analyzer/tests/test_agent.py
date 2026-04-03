"""Tests for TaintFlowAnalyzerAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_13_graph_relational.taint_flow_analyzer.agent import (
    TaintFlowAnalyzerAgent,
)


@pytest.fixture
def agent() -> TaintFlowAnalyzerAgent:
    return TaintFlowAnalyzerAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


# ── Positive cases ─────────────────────────────────────────────────────

class TestPositiveCases:
    def test_python_input_to_execute(self, agent: TaintFlowAnalyzerAgent) -> None:
        """input() flowing into cursor.execute() => sql_injection."""
        src = (
            'user_input = input("Enter ID: ")\n'
            'query = "SELECT * FROM users WHERE id = " + user_input\n'
            "cursor.execute(query)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any(f.severity == "critical" for f in findings)
        assert any("sql_injection" in f.title for f in findings)

    def test_python_input_to_os_system(self, agent: TaintFlowAnalyzerAgent) -> None:
        """input() flowing into os.system() => command_injection."""
        src = (
            'cmd = input("Enter command: ")\n'
            "os.system(cmd)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("command_injection" in f.title for f in findings)

    def test_python_request_args_to_open(self, agent: TaintFlowAnalyzerAgent) -> None:
        """request.args.get flowing into open() => path_traversal."""
        src = (
            'filename = request.args.get("file")\n'
            "f = open(filename)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("path_traversal" in f.title for f in findings)

    def test_python_request_to_requests_get(self, agent: TaintFlowAnalyzerAgent) -> None:
        """request.args.get flowing into requests.get() => ssrf."""
        src = (
            'url = request.args.get("url")\n'
            "resp = requests.get(url)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("ssrf" in f.title for f in findings)

    def test_python_input_to_pickle_loads(self, agent: TaintFlowAnalyzerAgent) -> None:
        """input() flowing into pickle.loads() => deserialization."""
        src = (
            'data = input("data: ")\n'
            "obj = pickle.loads(data)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("deserialization" in f.title for f in findings)
        assert any(f.severity == "critical" for f in findings)

    def test_python_input_to_render_template_string(self, agent: TaintFlowAnalyzerAgent) -> None:
        """input() flowing into render_template_string() => xss."""
        src = (
            'name = input("name: ")\n'
            "html = render_template_string(name)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("xss" in f.title for f in findings)

    def test_python_input_to_subprocess(self, agent: TaintFlowAnalyzerAgent) -> None:
        """input() flowing into subprocess.run() => command_injection."""
        src = (
            'user_cmd = input("cmd: ")\n'
            "subprocess.run(user_cmd)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert any("command_injection" in f.title for f in findings)
        assert any(f.severity == "critical" for f in findings)

    def test_python_propagation_through_intermediate(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Taint should propagate through intermediate variable assignments."""
        src = (
            'raw = input("val: ")\n'
            'built = "SELECT * FROM t WHERE x = " + raw\n'
            "cursor.execute(built)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_python_env_var_to_execute(self, agent: TaintFlowAnalyzerAgent) -> None:
        """os.environ.get flowing into execute() => sql_injection."""
        src = (
            'db_filter = os.environ.get("FILTER")\n'
            'query = "SELECT * FROM t WHERE col = " + db_filter\n'
            "cursor.execute(query)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_js_req_body_to_query(self, agent: TaintFlowAnalyzerAgent) -> None:
        """JS req.body flowing into db.query() => sql_injection."""
        src = (
            "const userId = req.body.id;\n"
            'db.query("SELECT * FROM users WHERE id = " + userId);\n'
        )
        findings = agent.analyze(_ctx(src, "javascript", "app.js"))
        assert len(findings) >= 1
        assert any("sql_injection" in f.title for f in findings)

    def test_js_req_params_to_exec(self, agent: TaintFlowAnalyzerAgent) -> None:
        """JS req.params flowing into exec() => command_injection."""
        src = (
            "const cmd = req.params.cmd;\n"
            "exec(cmd);\n"
        )
        findings = agent.analyze(_ctx(src, "javascript", "handler.js"))
        assert len(findings) >= 1
        assert any("command_injection" in f.title for f in findings)

    def test_finding_fields_populated(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Verify Finding fields are all correctly set."""
        src = (
            'user_input = input("Enter ID: ")\n'
            'query = "SELECT * FROM users WHERE id = " + user_input\n'
            "cursor.execute(query)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        f = findings[0]
        assert f.agent_name == "taint_flow_analyzer"
        assert f.category == "security"
        assert f.file_path == "test.py"
        assert f.line_start >= 1
        assert f.line_end >= f.line_start
        assert 0.0 <= f.confidence <= 1.0
        assert len(f.tags) >= 1


# ── Negative cases ─────────────────────────────────────────────────────

class TestNegativeCases:
    def test_no_sources(self, agent: TaintFlowAnalyzerAgent) -> None:
        """No taint sources => no findings."""
        src = (
            'query = "SELECT count(*) FROM users"\n'
            "cursor.execute(query)\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_no_sinks(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Sources present but no sinks => no findings."""
        src = (
            'name = input("name: ")\n'
            "print(name)\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_hardcoded_values_only(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Hardcoded constant strings with no taint => no findings."""
        src = (
            'query = "SELECT 1"\n'
            "cursor.execute(query)\n"
        )
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Empty source code => no findings."""
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Whitespace-only source => no findings."""
        assert agent.analyze(_ctx("   \n\n  \n")) == []

    def test_pure_arithmetic(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Pure arithmetic code with no sources or sinks."""
        src = "x = 1 + 2\ny = x * 3\n"
        assert agent.analyze(_ctx(src)) == []

    def test_parameterized_query_no_taint(self, agent: TaintFlowAnalyzerAgent) -> None:
        """A parameterized query with no tainted input should not flag."""
        src = (
            'name = "Alice"\n'
            'cursor.execute("SELECT * FROM users WHERE name = %s", (name,))\n'
        )
        assert agent.analyze(_ctx(src)) == []

    def test_function_def_only(self, agent: TaintFlowAnalyzerAgent) -> None:
        """A function definition with no body executing sinks."""
        src = (
            "def greet(name):\n"
            '    return "Hello, " + name\n'
        )
        assert agent.analyze(_ctx(src)) == []

    def test_import_statements_only(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Only import statements => no findings."""
        src = "import os\nimport sys\n"
        assert agent.analyze(_ctx(src)) == []

    def test_unsupported_language(self, agent: TaintFlowAnalyzerAgent) -> None:
        """An unsupported language returns no findings."""
        src = 'MOVE user-input TO ws-query.\nEXEC SQL SELECT * FROM users END-EXEC.\n'
        assert agent.analyze(_ctx(src, "cobol", "prog.cob")) == []

    def test_comment_only(self, agent: TaintFlowAnalyzerAgent) -> None:
        """Source code that is only comments."""
        src = "# user_input = input('x')\n# cursor.execute(query)\n"
        assert agent.analyze(_ctx(src)) == []


# ── Edge cases & metadata ──────────────────────────────────────────────

class TestEdgeCases:
    def test_metadata_name(self, agent: TaintFlowAnalyzerAgent) -> None:
        m = agent.metadata()
        assert m.name == "taint_flow_analyzer"

    def test_metadata_version(self, agent: TaintFlowAnalyzerAgent) -> None:
        m = agent.metadata()
        assert m.version == "0.1.0"

    def test_metadata_languages(self, agent: TaintFlowAnalyzerAgent) -> None:
        m = agent.metadata()
        assert "python" in m.languages
        assert "javascript" in m.languages
        assert "go" in m.languages

    def test_metadata_methodology(self, agent: TaintFlowAnalyzerAgent) -> None:
        m = agent.metadata()
        assert m.methodology == "graph_relational"

    def test_metadata_axis_type(self, agent: TaintFlowAnalyzerAgent) -> None:
        m = agent.metadata()
        assert m.axis_type == "aware"

    def test_metadata_tags(self, agent: TaintFlowAnalyzerAgent) -> None:
        m = agent.metadata()
        assert "security" in m.tags
        assert "taint" in m.tags

    def test_metadata_model_not_required(self, agent: TaintFlowAnalyzerAgent) -> None:
        m = agent.metadata()
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: TaintFlowAnalyzerAgent) -> None:
        src = (
            'user_input = input("Enter ID: ")\n'
            'query = "SELECT * FROM users WHERE id = " + user_input\n'
            "cursor.execute(query)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_suggest_fix_returns_none(self, agent: TaintFlowAnalyzerAgent) -> None:
        src = (
            'user_input = input("Enter ID: ")\n'
            'query = "SELECT * FROM users WHERE id = " + user_input\n'
            "cursor.execute(query)\n"
        )
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert agent.suggest_fix(findings[0]) is None

    def test_severity_mapping_critical(self, agent: TaintFlowAnalyzerAgent) -> None:
        """sql_injection and command_injection should both be critical."""
        src_sql = (
            'uid = input("id: ")\n'
            'q = "SELECT * FROM t WHERE id = " + uid\n'
            "cursor.execute(q)\n"
        )
        src_cmd = (
            'cmd = input("cmd: ")\n'
            "os.system(cmd)\n"
        )
        sql_findings = agent.analyze(_ctx(src_sql))
        cmd_findings = agent.analyze(_ctx(src_cmd))
        assert all(f.severity == "critical" for f in sql_findings)
        assert all(f.severity == "critical" for f in cmd_findings)

    def test_severity_mapping_high(self, agent: TaintFlowAnalyzerAgent) -> None:
        """path_traversal, ssrf, and xss should be high."""
        src_path = (
            'filename = request.args.get("f")\n'
            "open(filename)\n"
        )
        findings = agent.analyze(_ctx(src_path))
        assert all(f.severity == "high" for f in findings)
