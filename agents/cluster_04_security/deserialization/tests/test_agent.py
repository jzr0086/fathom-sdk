"""Tests for DeserializationAgent."""

from __future__ import annotations

import pytest

from fathom_sdk import CodeContext
from fathom_sdk.context.ast_parser import parse

from agents.cluster_04_security.deserialization.agent import DeserializationAgent


@pytest.fixture
def agent() -> DeserializationAgent:
    return DeserializationAgent()


def _ctx(source: str, language: str = "python", fp: str = "test.py") -> CodeContext:
    try:
        ast = parse(source, language)
    except (ValueError, Exception):
        ast = None
    return CodeContext(source_code=source, language=language, file_path=fp, ast=ast)


class TestPositiveCases:
    def test_pickle_loads(self, agent: DeserializationAgent) -> None:
        src = "import pickle\ndata = pickle.loads(payload)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "critical"

    def test_pickle_load(self, agent: DeserializationAgent) -> None:
        src = "import pickle\nwith open('f', 'rb') as f:\n    data = pickle.load(f)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_cpickle_loads(self, agent: DeserializationAgent) -> None:
        src = "import cPickle\ndata = cPickle.loads(payload)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_marshal_loads(self, agent: DeserializationAgent) -> None:
        src = "import marshal\ncode = marshal.loads(data)\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_shelve_open(self, agent: DeserializationAgent) -> None:
        src = "import shelve\ndb = shelve.open('mydb')\n"
        assert len(agent.analyze(_ctx(src))) >= 1

    def test_yaml_load_without_safeloader(self, agent: DeserializationAgent) -> None:
        src = "import yaml\ndata = yaml.load(content)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_yaml_load_with_fullloader(self, agent: DeserializationAgent) -> None:
        src = "import yaml\ndata = yaml.load(content, Loader=FullLoader)\n"
        findings = agent.analyze(_ctx(src))
        assert len(findings) >= 1

    def test_js_eval(self, agent: DeserializationAgent) -> None:
        src = "const result = eval(userInput);\n"
        findings = agent.analyze(_ctx(src, "javascript", "test.js"))
        assert len(findings) >= 1
        assert findings[0].severity == "critical"

    def test_js_function_constructor(self, agent: DeserializationAgent) -> None:
        src = "const fn = Function('return ' + data);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "test.js"))) >= 1

    def test_js_unserialize(self, agent: DeserializationAgent) -> None:
        src = "const obj = unserialize(cookie);\n"
        assert len(agent.analyze(_ctx(src, "javascript", "test.js"))) >= 1

    def test_java_readobject(self, agent: DeserializationAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        ObjectInputStream ois = new ObjectInputStream(is);\n"
            "        Object obj = ois.readObject();\n"
            "    }\n"
            "}\n"
        )
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_java_readunshared(self, agent: DeserializationAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        Object obj = ois.readUnshared();\n"
            "    }\n"
            "}\n"
        )
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_java_xmldecoder(self, agent: DeserializationAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        XMLDecoder decoder = new XMLDecoder(is);\n"
            "    }\n"
            "}\n"
        )
        assert len(agent.analyze(_ctx(src, "java", "T.java"))) >= 1

    def test_go_gob_decode(self, agent: DeserializationAgent) -> None:
        src = (
            'package main\n\nimport "encoding/gob"\n\n'
            "func f() {\n    err := gob.Decode(buf)\n}\n"
        )
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_go_gob_newdecoder(self, agent: DeserializationAgent) -> None:
        src = (
            'package main\n\nimport "encoding/gob"\n\n'
            "func f() {\n    dec := gob.NewDecoder(r)\n}\n"
        )
        assert len(agent.analyze(_ctx(src, "go", "t.go"))) >= 1

    def test_confidence_090(self, agent: DeserializationAgent) -> None:
        src = "import pickle\ndata = pickle.loads(payload)\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].confidence == 0.90

    def test_category_security(self, agent: DeserializationAgent) -> None:
        src = "import pickle\ndata = pickle.loads(payload)\n"
        findings = agent.analyze(_ctx(src))
        assert findings[0].category == "security"


class TestNegativeCases:
    def test_yaml_safe_load(self, agent: DeserializationAgent) -> None:
        src = "import yaml\ndata = yaml.safe_load(content)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_yaml_load_with_safeloader(self, agent: DeserializationAgent) -> None:
        src = "import yaml\ndata = yaml.load(content, Loader=SafeLoader)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_yaml_load_with_yaml_safeloader(self, agent: DeserializationAgent) -> None:
        src = "import yaml\ndata = yaml.load(content, Loader=yaml.SafeLoader)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_json_loads(self, agent: DeserializationAgent) -> None:
        src = "import json\ndata = json.loads(payload)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_json_load(self, agent: DeserializationAgent) -> None:
        src = "import json\nwith open('f') as f:\n    data = json.load(f)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_js_json_parse(self, agent: DeserializationAgent) -> None:
        src = "const data = JSON.parse(text);\n"
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_java_jackson(self, agent: DeserializationAgent) -> None:
        src = (
            "public class T {\n"
            "    void f() {\n"
            "        ObjectMapper mapper = new ObjectMapper();\n"
            "        MyObj obj = mapper.readValue(json, MyObj.class);\n"
            "    }\n"
            "}\n"
        )
        assert agent.analyze(_ctx(src, "java", "T.java")) == []

    def test_python_comment(self, agent: DeserializationAgent) -> None:
        src = "# data = pickle.loads(payload)\nx = 1\n"
        assert agent.analyze(_ctx(src)) == []

    def test_js_comment(self, agent: DeserializationAgent) -> None:
        src = "// const result = eval(data);\nlet x = 1;\n"
        assert agent.analyze(_ctx(src, "javascript", "test.js")) == []

    def test_empty_source(self, agent: DeserializationAgent) -> None:
        assert agent.analyze(_ctx("")) == []

    def test_no_deserialization(self, agent: DeserializationAgent) -> None:
        src = "x = 1\ny = 2\nprint(x + y)\n"
        assert agent.analyze(_ctx(src)) == []

    def test_go_json_unmarshal(self, agent: DeserializationAgent) -> None:
        src = (
            'package main\n\nimport "encoding/json"\n\n'
            "func f() {\n    json.Unmarshal(data, &obj)\n}\n"
        )
        assert agent.analyze(_ctx(src, "go", "t.go")) == []

    def test_unsupported_language(self, agent: DeserializationAgent) -> None:
        assert agent.analyze(_ctx("x = 1", "cobol", "t.cob")) == []

    def test_ts_json_parse(self, agent: DeserializationAgent) -> None:
        src = "const data: MyType = JSON.parse(text);\n"
        assert agent.analyze(_ctx(src, "typescript", "test.ts")) == []


class TestEdgeCases:
    def test_metadata(self, agent: DeserializationAgent) -> None:
        m = agent.metadata()
        assert m.name == "deserialization"
        assert m.version == "0.1.0"
        assert "deserialization" in m.tags
        assert m.model_required is False
        assert m.estimated_cost_cents == 0.0

    def test_metadata_languages(self, agent: DeserializationAgent) -> None:
        m = agent.metadata()
        assert "python" in m.languages
        assert "javascript" in m.languages
        assert "typescript" in m.languages
        assert "java" in m.languages
        assert "go" in m.languages

    def test_metadata_domains(self, agent: DeserializationAgent) -> None:
        m = agent.metadata()
        assert "security_engineering" in m.domains
        assert "web_development" in m.domains

    def test_explain(self, agent: DeserializationAgent) -> None:
        src = "import pickle\ndata = pickle.loads(payload)\n"
        findings = agent.analyze(_ctx(src))
        explanation = agent.explain(findings[0])
        assert "deserialization" in explanation.lower()

    def test_no_duplicate_findings_same_line(self, agent: DeserializationAgent) -> None:
        src = "import pickle\ndata = pickle.loads(payload)\n"
        findings = agent.analyze(_ctx(src))
        lines = [f.line_start for f in findings]
        assert len(lines) == len(set(lines))
