"""Tests for CommandInjectionAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.command_injection.agent import CommandInjectionAgent


@pytest.fixture
def agent() -> CommandInjectionAgent:
    return CommandInjectionAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    """At least 10 positive cases — code that SHOULD trigger findings."""

    def test_os_system_fstring(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system(f"rm -rf {user_dir}")\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "critical"

    def test_os_system_variable(self, agent: CommandInjectionAgent) -> None:
        src = "os.system(cmd)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_os_system_concat(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system("ping " + host)\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_os_system_format(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system("ls {}".format(dirname))\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_subprocess_run_shell_true_fstring(self, agent: CommandInjectionAgent) -> None:
        src = 'subprocess.run(f"grep {pattern} file.txt", shell=True)\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_subprocess_call_shell_true_variable(self, agent: CommandInjectionAgent) -> None:
        src = "subprocess.call(cmd, shell=True)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_subprocess_popen_shell_true_concat(self, agent: CommandInjectionAgent) -> None:
        src = 'subprocess.Popen("cat " + filename, shell=True)\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_js_exec_variable(self, agent: CommandInjectionAgent) -> None:
        src = "const { exec } = require('child_process');\nexec(userInput);\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert len(findings) >= 1

    def test_js_exec_template_literal(self, agent: CommandInjectionAgent) -> None:
        src = "exec(`ls ${dir}`);\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert len(findings) >= 1

    def test_js_execsync_concat(self, agent: CommandInjectionAgent) -> None:
        src = "execSync('rm -rf ' + path);\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert len(findings) >= 1

    def test_go_exec_command_variable(self, agent: CommandInjectionAgent) -> None:
        src = 'package main\n\nimport "os/exec"\n\nfunc run(cmd string) {\n\texec.Command(cmd)\n}\n'
        findings = agent.analyze(_ctx(src, "go", "main.go"))
        assert len(findings) >= 1

    def test_subprocess_check_output_shell_true(self, agent: CommandInjectionAgent) -> None:
        src = 'subprocess.check_output(f"whoami {arg}", shell=True)\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_severity_is_critical(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system(f"echo {data}")\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "critical"

    def test_confidence_is_correct(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system(f"echo {data}")\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].confidence == 0.88


class TestNegativeCases:
    """At least 10 negative cases — code that should NOT trigger findings."""

    def test_subprocess_list_args(self, agent: CommandInjectionAgent) -> None:
        src = 'subprocess.run(["ls", "-la", dirname])\n'
        assert agent.analyze(_ctx(src)) == []

    def test_subprocess_list_no_shell(self, agent: CommandInjectionAgent) -> None:
        src = 'subprocess.call(["echo", "hello"])\n'
        assert agent.analyze(_ctx(src)) == []

    def test_hardcoded_os_system(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system("ls -la")\n'
        assert agent.analyze(_ctx(src)) == []

    def test_hardcoded_subprocess_shell(self, agent: CommandInjectionAgent) -> None:
        src = 'subprocess.run("echo hello", shell=True)\n'
        assert agent.analyze(_ctx(src)) == []

    def test_no_shell_calls(self, agent: CommandInjectionAgent) -> None:
        src = "x = 1\ny = x + 2\nprint(y)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_empty_source(self, agent: CommandInjectionAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_whitespace_only(self, agent: CommandInjectionAgent) -> None:
        assert agent.analyze(_ctx("   \n  \n")) == []

    def test_comment_os_system(self, agent: CommandInjectionAgent) -> None:
        src = '# os.system(f"rm {path}")\n'
        assert agent.analyze(_ctx(src)) == []

    def test_js_comment(self, agent: CommandInjectionAgent) -> None:
        src = "// exec(userInput);\n"
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_subprocess_popen_list_args(self, agent: CommandInjectionAgent) -> None:
        src = 'subprocess.Popen(["git", "status"])\n'
        assert agent.analyze(_ctx(src)) == []

    def test_non_shell_exec_call(self, agent: CommandInjectionAgent) -> None:
        """A function named exec that is not a shell call (e.g. DB exec)."""
        src = 'cursor.exec("SELECT 1")\n'
        # This should not flag because there is no interpolation or variable arg
        assert agent.analyze(_ctx(src)) == []

    def test_print_not_flagged(self, agent: CommandInjectionAgent) -> None:
        src = 'print(f"command is {cmd}")\n'
        assert agent.analyze(_ctx(src)) == []


class TestEdgeCases:
    def test_metadata(self, agent: CommandInjectionAgent) -> None:
        m = agent.metadata()
        assert m.name == "command_injection"
        assert m.version == "0.1.0"
        assert "python" in m.languages
        assert "command-injection" in m.tags
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_explain_returns_string(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system(f"echo {data}")\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        explanation = agent.explain(findings[0])
        assert isinstance(explanation, str)
        assert "command injection" in explanation.lower()

    def test_unsupported_language(self, agent: CommandInjectionAgent) -> None:
        assert agent.analyze(_ctx("system(cmd)", "cobol", "t.cob")) == []

    def test_category_is_security(self, agent: CommandInjectionAgent) -> None:
        src = 'os.system(f"echo {data}")\n'
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].category == "security"

    def test_no_duplicate_findings(self, agent: CommandInjectionAgent) -> None:
        """Same line should not produce duplicate findings from AST + source passes."""
        src = 'os.system(f"rm {path}")\n'
        findings = agent.analyze(_ctx(src))
        lines = [f.line_start for f in findings]
        assert len(lines) == len(set(lines))
