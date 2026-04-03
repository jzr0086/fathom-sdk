"""Tests for InsecureRandomnessAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.insecure_randomness.agent import InsecureRandomnessAgent


@pytest.fixture
def agent() -> InsecureRandomnessAgent:
    return InsecureRandomnessAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_python_random_random(self, agent: InsecureRandomnessAgent) -> None:
        assert len(agent.analyze(_ctx("import random\nx = random.random()\n"))) >= 1

    def test_python_random_randint(self, agent: InsecureRandomnessAgent) -> None:
        assert len(agent.analyze(_ctx("import random\nx = random.randint(1, 100)\n"))) >= 1

    def test_python_random_choice(self, agent: InsecureRandomnessAgent) -> None:
        assert len(agent.analyze(_ctx("import random\nx = random.choice([1, 2])\n"))) >= 1

    def test_js_math_random(self, agent: InsecureRandomnessAgent) -> None:
        assert len(agent.analyze(_ctx("let x = Math.random();\n", "javascript", "t.js"))) >= 1

    def test_ts_math_random(self, agent: InsecureRandomnessAgent) -> None:
        assert len(agent.analyze(_ctx("const x = Math.random();\n", "typescript", "t.ts"))) >= 1

    def test_java_random_nextint(self, agent: InsecureRandomnessAgent) -> None:
        src = "public class T {\n    void f() {\n        Random r = new Random();\n        int x = r.nextInt(100);\n    }\n}\n"
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_rand_intn(self, agent: InsecureRandomnessAgent) -> None:
        src = 'package main\n\nimport "math/rand"\n\nfunc f() {\n    x := rand.Intn(100)\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_severity_medium(self, agent: InsecureRandomnessAgent) -> None:
        f = agent.analyze(_ctx("import random\nx = random.random()\n"))[0]
        assert f.severity == "medium"

    def test_python_random_shuffle(self, agent: InsecureRandomnessAgent) -> None:
        assert len(agent.analyze(_ctx("import random\nrandom.shuffle(items)\n"))) >= 1

    def test_go_rand_float64(self, agent: InsecureRandomnessAgent) -> None:
        src = 'package main\n\nimport "math/rand"\n\nfunc f() {\n    x := rand.Float64()\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1


class TestNegativeCases:
    def test_python_secrets(self, agent: InsecureRandomnessAgent) -> None:
        assert agent.analyze(_ctx("import secrets\nt = secrets.token_hex(32)\n")) == []

    def test_python_os_urandom(self, agent: InsecureRandomnessAgent) -> None:
        assert agent.analyze(_ctx("import os\nx = os.urandom(32)\n")) == []

    def test_js_crypto(self, agent: InsecureRandomnessAgent) -> None:
        src = "const buf = crypto.getRandomValues(new Uint8Array(32));\n"
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_java_secure_random(self, agent: InsecureRandomnessAgent) -> None:
        src = "public class T {\n    void f() {\n        SecureRandom sr = new SecureRandom();\n    }\n}\n"
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_no_random(self, agent: InsecureRandomnessAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_go_crypto_rand(self, agent: InsecureRandomnessAgent) -> None:
        src = 'package main\n\nimport "crypto/rand"\n\nfunc f() {\n    n, _ := rand.Int(rand.Reader, big.NewInt(100))\n}\n'
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_comment_mention(self, agent: InsecureRandomnessAgent) -> None:
        assert agent.analyze(_ctx("# random.random() is not secure\nx = 1\n")) == []

    def test_simple_math(self, agent: InsecureRandomnessAgent) -> None:
        assert agent.analyze(_ctx("x = 1 + 2\n")) == []

    def test_empty(self, agent: InsecureRandomnessAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_class_method(self, agent: InsecureRandomnessAgent) -> None:
        src = "class Foo:\n    def method(self):\n        return self.value\n"
        assert agent.analyze(_ctx(src)) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: InsecureRandomnessAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: InsecureRandomnessAgent) -> None:
        m = agent.metadata()
        assert m.name == "insecure_randomness"
