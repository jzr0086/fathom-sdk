"""Blocking I/O in Async Context — flags sync I/O inside async functions."""

from __future__ import annotations

from fathom_sdk import AgentMetadata, BaseReviewAgent, CodeContext, Finding
from fathom_sdk.context.ast_helpers import find_calls, find_functions
from fathom_sdk.context.ast_parser import parse

_BLOCKING_CALLS: dict[str, set[str]] = {
    "python": {
        "open",
        "sleep",
        "get",
        "post",
        "put",
        "delete",
        "patch",
        "head",
        "urlopen",
        "connect",
        "run",
        "call",
        "check_output",
        "read",
        "write",
        "recv",
        "send",
        "input",
        "getaddrinfo",
    },
    "javascript": {
        "readFileSync",
        "writeFileSync",
        "appendFileSync",
        "existsSync",
        "mkdirSync",
        "readdirSync",
        "statSync",
        "execSync",
        "spawnSync",
    },
    "typescript": {
        "readFileSync",
        "writeFileSync",
        "appendFileSync",
        "existsSync",
        "mkdirSync",
        "readdirSync",
        "statSync",
        "execSync",
        "spawnSync",
    },
}

# Full-text patterns for more precise matching
_BLOCKING_FULL_TEXT: dict[str, list[str]] = {
    "python": [
        "time.sleep(",
        "requests.get(",
        "requests.post(",
        "requests.put(",
        "requests.delete(",
        "requests.patch(",
        "urllib.request.urlopen(",
        "socket.connect(",
        "subprocess.run(",
        "subprocess.call(",
        "subprocess.check_output(",
        "os.read(",
        "os.write(",
    ],
    "javascript": [],
    "typescript": [],
}


class BlockingIoAsyncAgent(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="blocking_io_async",
            version="0.1.0",
            languages=["python", "javascript", "typescript"],
            domains=["web_development", "api_integration"],
            methodology="performance",
            axis_type="aware",
            tags=["performance", "async", "blocking-io"],
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        if not context.source_code.strip():
            return []
        lang = context.language.lower()
        if lang not in _BLOCKING_CALLS:
            return []
        root = context.ast
        if root is None:
            try:
                root = parse(context.source_code, context.language)
            except (ValueError, Exception):
                return []

        # Find async function names
        async_funcs = {f.name for f in find_functions(root, context.language) if f.is_async}
        if not async_funcs:
            return []

        blocking_names = _BLOCKING_CALLS[lang]
        blocking_texts = _BLOCKING_FULL_TEXT.get(lang, [])
        findings: list[Finding] = []

        for call in find_calls(root, context.language):
            if call.enclosing_function not in async_funcs:
                continue

            is_blocking = False
            if call.callee_name in blocking_names:
                # For Python, verify with full_text for common false positives
                if lang == "python":
                    if any(bt in call.full_text for bt in blocking_texts):
                        is_blocking = True
                    elif call.callee_name == "open":
                        is_blocking = True
                else:
                    is_blocking = True

            if is_blocking:
                findings.append(
                    Finding(
                        agent_name="blocking_io_async",
                        severity="high",
                        category="performance",
                        title=(
                            f"Blocking call '{call.callee_name}' "
                            f"in async function '{call.enclosing_function}'"
                        ),
                        description=(
                            f"Synchronous I/O call '{call.full_text.split(chr(10))[0][:60]}' "
                            f"at line {call.start_line} inside async function "
                            f"'{call.enclosing_function}' blocks the event loop."
                        ),
                        file_path=context.file_path,
                        line_start=call.start_line,
                        line_end=call.end_line,
                        confidence=0.90,
                        tags=["performance", "async", "blocking-io"],
                    )
                )
        return findings

    def explain(self, finding: Finding) -> str:
        return (
            f"{finding.title}. Blocking calls in async functions prevent the event "
            f"loop from processing other tasks. Use async alternatives (aiohttp, "
            f"aiofiles, asyncio.sleep) or run in a thread executor."
        )
