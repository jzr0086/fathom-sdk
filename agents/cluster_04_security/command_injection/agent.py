"""Command Injection Detector — flags shell execution with unsanitized input."""

from __future__ import annotations

import re

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, get_node_text
from fathom_sdk.context.ast_parser import parse

# ---------------------------------------------------------------------------
# Dangerous shell execution sinks (callee names)
# ---------------------------------------------------------------------------

# Calls that are ALWAYS dangerous when arguments contain user input
_ALWAYS_DANGEROUS_CALLS: set[str] = {
    "system",       # os.system() — Python, C
    "exec",         # child_process.exec() — JS/TS
    "execSync",     # child_process.execSync() — JS/TS
}

# Calls that are dangerous when combined with shell=True or variable arguments
_SHELL_CALLS: set[str] = {
    "run",          # subprocess.run()
    "call",         # subprocess.call()
    "check_output", # subprocess.check_output()
    "check_call",   # subprocess.check_call()
    "Popen",        # subprocess.Popen()
}

# Java Runtime.exec() and ProcessBuilder
_JAVA_EXEC_CALLS: set[str] = {
    "exec",         # Runtime.getRuntime().exec()
}

# Go os/exec calls
_GO_EXEC_CALLS: set[str] = {
    "Command",      # exec.Command()
}

# ---------------------------------------------------------------------------
# Interpolation / concatenation signals (shared with SQL injection logic)
# ---------------------------------------------------------------------------

_INTERPOLATION_SIGNALS = [
    re.compile(r'f["\']'),          # Python f-string
    re.compile(r"\$\{"),            # JS template literal
    re.compile(r"\.format\s*\("),   # .format()
    re.compile(r'\+\s*["\']|\+\s*\w'),  # string concatenation (var + ...)
    re.compile(r'["\'\`]\s*\+'),    # string + variable
    re.compile(r"%\s"),             # Python % formatting: "cmd %s" % var
]

_PERCENT_FORMAT = re.compile(r"%[sd]")

# shell=True detection
_SHELL_TRUE = re.compile(r"shell\s*=\s*True")


def _has_interpolation(text: str) -> bool:
    """Check if text contains string interpolation or concatenation signals."""
    for signal in _INTERPOLATION_SIGNALS:
        if signal.search(text):
            return True
    if _PERCENT_FORMAT.search(text):
        return True
    return False


def _has_variable_arg(text: str) -> bool:
    """Check if a call has a variable (non-literal) argument.

    Returns True when the first argument is not a plain string literal,
    indicating potential taint from user input.
    """
    # If it contains interpolation, it is tainted
    if _has_interpolation(text):
        return True
    # Check for variable names passed directly (not string literals)
    # e.g. os.system(cmd) or exec.Command(userInput)
    # Look for a function call where the first arg is an identifier
    paren = text.find("(")
    if paren == -1:
        return False
    args_text = text[paren + 1:]
    args_text = args_text.strip()
    # If args start with a quote, it is a literal string
    if args_text and args_text[0] in ('"', "'", "`"):
        return False
    # If args start with an identifier character, it is likely a variable
    if args_text and (args_text[0].isalpha() or args_text[0] == "_"):
        return True
    return False


# ---------------------------------------------------------------------------
# Source-level patterns for broader detection
# ---------------------------------------------------------------------------

_SOURCE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"os\.system\s*\((?!.*['\"][^'\"]*['\"])"), "os.system() with variable argument"),
    (re.compile(r"os\.system\s*\(.*f['\"]"), "os.system() with f-string"),
    (re.compile(r"os\.system\s*\(.*\.format\s*\("), "os.system() with .format()"),
    (re.compile(r"os\.system\s*\(.*\+"), "os.system() with string concatenation"),
    (re.compile(r"os\.popen\s*\(.*f['\"]"), "os.popen() with f-string"),
    (re.compile(r"os\.popen\s*\((?!.*['\"][^'\"]*['\"])"), "os.popen() with variable argument"),
    (re.compile(r"subprocess\.\w+\(.*shell\s*=\s*True.*f['\"]"), "subprocess with shell=True and f-string"),
    (re.compile(r"commands\.getoutput\s*\("), "commands.getoutput() — deprecated and unsafe"),
]


class CommandInjectionAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="command_injection",
            version="0.1.0",
            languages=["python", "javascript", "typescript", "java", "go"],
            domains=["security_engineering", "web_development"],
            methodology="security",
            axis_type="aware",
            tags=["security", "command-injection", "injection"],
            model_required=False,
            estimated_cost_cents=0.0,
        )

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
        seen_lines: set[int] = set()

        # --- AST-based detection ---
        for call in find_calls(root, context.language):
            text = call.full_text
            callee = call.callee_name

            flagged = False
            title = ""

            # Always-dangerous sinks (os.system, exec, execSync)
            if callee in _ALWAYS_DANGEROUS_CALLS:
                if _has_variable_arg(text):
                    flagged = True
                    title = f"Command injection via {callee}() with dynamic argument"

            # subprocess calls with shell=True and variable/interpolated args
            elif callee in _SHELL_CALLS:
                if _SHELL_TRUE.search(text) and _has_variable_arg(text):
                    flagged = True
                    title = f"Command injection via subprocess.{callee}() with shell=True"
                elif _SHELL_TRUE.search(text) and _has_interpolation(text):
                    flagged = True
                    title = f"Command injection via subprocess.{callee}() with shell=True"

            # Java Runtime.exec() with string concatenation
            elif callee in _JAVA_EXEC_CALLS and context.language == "java":
                if _has_interpolation(text):
                    flagged = True
                    title = "Command injection via Runtime.exec() with dynamic argument"

            # Go exec.Command() with variable arguments
            elif callee in _GO_EXEC_CALLS and context.language == "go":
                if _has_variable_arg(text):
                    flagged = True
                    title = "Command injection via exec.Command() with dynamic argument"

            if flagged and call.start_line not in seen_lines:
                seen_lines.add(call.start_line)
                findings.append(
                    Finding(
                        agent_name="command_injection",
                        severity="critical",
                        category="security",
                        title=title,
                        description=(
                            f"Shell command at line {call.start_line} uses dynamic input. "
                            f"An attacker who controls the input can execute arbitrary system "
                            f"commands. Use parameterized APIs (e.g. subprocess with a list "
                            f"argument) or validate/sanitize input strictly."
                        ),
                        file_path=context.file_path,
                        line_start=call.start_line,
                        line_end=call.end_line,
                        confidence=0.88,
                        tags=["security", "command-injection", "injection"],
                    )
                )

        # --- Source-level fallback for patterns not caught by AST ---
        lines = context.source_code.splitlines()
        for i, line in enumerate(lines, 1):
            if i in seen_lines:
                continue
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                continue
            for pattern, desc in _SOURCE_PATTERNS:
                if pattern.search(line):
                    seen_lines.add(i)
                    findings.append(
                        Finding(
                            agent_name="command_injection",
                            severity="critical",
                            category="security",
                            title=f"Potential command injection: {desc}",
                            description=(
                                f"Line {i} executes a shell command with dynamic input. "
                                f"Use parameterized APIs or validate input strictly."
                            ),
                            file_path=context.file_path,
                            line_start=i,
                            line_end=i,
                            confidence=0.85,
                            tags=["security", "command-injection", "injection"],
                        )
                    )
                    break

        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Command injection allows attackers to execute arbitrary "
            f"system commands on the server. Never pass unsanitized user input to shell "
            f"execution functions. Use subprocess with a list of arguments (no shell=True), "
            f"shlex.quote() for escaping, or avoid shell execution entirely."
        )
