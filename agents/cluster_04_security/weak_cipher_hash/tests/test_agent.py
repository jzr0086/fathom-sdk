"""Tests for WeakCipherHashAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.weak_cipher_hash.agent import WeakCipherHashAgent


@pytest.fixture
def agent() -> WeakCipherHashAgent:
    return WeakCipherHashAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_python_hashlib_md5(self, agent: WeakCipherHashAgent) -> None:
        assert len(agent.analyze(_ctx("import hashlib\nh = hashlib.md5(data)\n"))) >= 1

    def test_python_hashlib_sha1(self, agent: WeakCipherHashAgent) -> None:
        assert len(agent.analyze(_ctx("import hashlib\nh = hashlib.sha1(data)\n"))) >= 1

    def test_js_create_hash_md5(self, agent: WeakCipherHashAgent) -> None:
        src = 'const h = crypto.createHash("md5");\n'
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_js_create_hash_sha1(self, agent: WeakCipherHashAgent) -> None:
        src = "const h = crypto.createHash('sha1');\n"
        assert len(agent.analyze(_ctx(src, "javascript", "t.js"))) >= 1

    def test_java_get_instance_md5(self, agent: WeakCipherHashAgent) -> None:
        src = 'public class T {\n    void f() {\n        MessageDigest.getInstance("MD5");\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_java_get_instance_des(self, agent: WeakCipherHashAgent) -> None:
        src = 'public class T {\n    void f() {\n        Cipher.getInstance("DES");\n    }\n}\n'
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_md5_new(self, agent: WeakCipherHashAgent) -> None:
        src = 'package main\n\nimport "crypto/md5"\n\nfunc f() {\n    h := md5.New()\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_go_sha1_new(self, agent: WeakCipherHashAgent) -> None:
        src = 'package main\n\nimport "crypto/sha1"\n\nfunc f() {\n    h := sha1.New()\n}\n'
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_severity_is_high(self, agent: WeakCipherHashAgent) -> None:
        f = agent.analyze(_ctx("import hashlib\nh = hashlib.md5(data)\n"))[0]
        assert f.severity == "high"
        assert f.category == "security"

    def test_python_des_import(self, agent: WeakCipherHashAgent) -> None:
        src = "from Crypto.Cipher import DES\n"
        assert len(agent.analyze(_ctx(src))) >= 1


class TestNegativeCases:
    def test_sha256(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("import hashlib\nh = hashlib.sha256(data)\n")) == []

    def test_sha512(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("h = hashlib.sha512(data)\n")) == []

    def test_js_sha256(self, agent: WeakCipherHashAgent) -> None:
        src = 'const h = crypto.createHash("sha256");\n'
        assert agent.analyze(_ctx(src, "javascript", "t.js")) == []

    def test_java_aes(self, agent: WeakCipherHashAgent) -> None:
        src = 'public class T {\n    void f() {\n        Cipher.getInstance("AES");\n    }\n}\n'
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_no_crypto(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("x = 1\ny = 2\n")) == []

    def test_comment_with_md5(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("# hashlib.md5 is deprecated\nx = 1\n")) == []

    def test_string_mention(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("msg = 'use sha256 instead of md5'\n")) == []

    def test_go_sha256(self, agent: WeakCipherHashAgent) -> None:
        src = 'package main\n\nimport "crypto/sha256"\n\nfunc f() {\n    h := sha256.New()\n}\n'
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_bcrypt(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("import bcrypt\nh = bcrypt.hashpw(pw, salt)\n")) == []

    def test_empty_file(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("")) == []


class TestEdgeCases:
    def test_unsupported_lang(self, agent: WeakCipherHashAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_metadata(self, agent: WeakCipherHashAgent) -> None:
        m = agent.metadata()
        assert m.name == "weak_cipher_hash"
        assert "security" in m.tags
