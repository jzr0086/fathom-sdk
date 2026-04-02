# Fathom Agent Manifest

> Canonical reference for every planned agent in the Fathom review system.
> All agents are numbered 1–75 with unique sequential IDs.

---

## Summary

- **75 base agents** across 14 active methodology clusters (cluster 15 is orchestration infrastructure)
- **~1,080 concrete instances** when expanded via language adapters across all tiers
- **Phase breakdown:** 20 (Phase 1) → 50 (Phase 2) → 75 (Phase 3)

### Distribution by Agent Type

| Type | Count | % | Description |
|---|---|---|---|
| Agnostic | 19 | 25% | Build once, language irrelevant |
| Aware | 43 | 57% | Base agent + per-language adapters |
| Critical | 13 | 17% | Full per-language implementation required |
| **Total** | **75** | **100%** | |

### Distribution by Cluster

| Cluster | # Agents | Agnostic | Aware | Critical |
|---|---|---|---|---|
| C01 Static Analysis & Syntax | 6 | 0 | 6 | 0 |
| C02 Semantic Understanding | 4 | 1 | 3 | 0 |
| C03 Bug Detection | 10 | 0 | 7 | 3 |
| C04 Security & Vulnerability | 12 | 2 | 10 | 0 |
| C05 Performance & Efficiency | 8 | 1 | 4 | 3 |
| C06 Architecture & Design | 5 | 5 | 0 | 0 |
| C07 Code Quality & Maintainability | 6 | 2 | 4 | 0 |
| C08 Testing | 5 | 0 | 5 | 0 |
| C09 Git Intelligence | 4 | 4 | 0 | 0 |
| C10 Language-Specific Deep | 7 | 0 | 0 | 7 |
| C11 LLM & Generative | 3 | 2 | 1 | 0 |
| C12 Observability | 2 | 0 | 2 | 0 |
| C13 Graph & Relational | 2 | 1 | 1 | 0 |
| C14 Learning & Adaptation | 1 | 1 | 0 | 0 |
| **Total** | **75** | **19** | **43** | **13** |

---

## All 75 Agents

### C01 — Static Analysis & Syntax (6 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 1 | Dead Code Reachability | aware | All | 1 | Detects unreachable code via AST control flow analysis |
| 2 | AST Pattern Matching | aware | All | 2 | Matches known anti-pattern AST subtrees against configurable rule sets |
| 3 | Token Sequence Classifier | aware | All | 2 | Classifies suspicious token sequences using trained sequence models |
| 4 | Identifier Naming Conventions | aware | All | 3 | Flags naming violations against language-specific conventions |
| 5 | Comment-Code Alignment | aware | All | 3 | Detects stale or misleading comments that contradict adjacent code |
| 6 | Import/Dependency Order | aware | All | 3 | Enforces canonical import ordering and flags unused/duplicate imports |

### C02 — Semantic Understanding (4 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 7 | Duplicate Code Detector | aware | All | 1 | Finds semantic code clones using code embeddings and similarity thresholds |
| 8 | Semantic Clone Detection | aware | All | 2 | Identifies Type-3/4 clones via CodeBERT embeddings and contrastive learning |
| 9 | Code Embedding Similarity | agnostic | All | 3 | Indexes codebase embeddings to surface unexpected similarity clusters |
| 10 | Intent-Implementation Mismatch | aware | All | 3 | Compares function names/docs against implementation semantics using LLM |

### C03 — Bug Detection (10 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 11 | Null Dereference Predictor | aware | All | 1 | Predicts null/None/nil dereferences via data flow analysis and ML |
| 12 | Resource Leak Detector | aware | All | 1 | Finds unclosed files, connections, locks via resource lifecycle tracking |
| 13 | Exception Swallowing Detector | aware | All | 1 | Flags empty catch blocks and silently ignored exceptions |
| 14 | Off-by-One Error Classifier | aware | All | 1 | Classifies loop bounds and index arithmetic for off-by-one patterns |
| 15 | Race Condition Detector | critical | Distributed, Systems | 1 | Detects data races via happens-before analysis (Go channels, Java synchronized) |
| 16 | Integer Overflow Detector | aware | Systems, Security | 2 | Flags arithmetic on untrusted integers without overflow checks |
| 17 | Infinite Loop/Recursion Detector | aware | All | 2 | Identifies missing termination conditions in loops and recursive calls |
| 18 | Uninitialized Variable Detector | aware | Systems | 2 | Finds variables used before assignment via reaching definitions analysis |
| 19 | Type Confusion Detector | critical | All | 2 | Detects type mismatches across function boundaries and casts |
| 20 | Memory Safety Violation Detector | critical | Systems | 3 | Identifies use-after-free, double-free, buffer overflows in C/C++/Rust |

### C04 — Security & Vulnerability (12 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 21 | Hardcoded Secrets Detector | aware | All | 1 | Detects API keys, passwords, tokens via entropy analysis + regex + ML |
| 22 | SQL Injection Detector | aware | Web, API, Database | 1 | Traces user input to SQL queries; flags unsanitized concatenation |
| 23 | XSS Pattern Detector | aware | Web | 1 | Identifies unescaped user input rendered in HTML/DOM contexts |
| 24 | Weak Cipher/Hash Detector | aware | Security, All | 1 | Flags MD5, SHA1, DES, RC4, and other deprecated cryptographic primitives |
| 25 | Insecure Randomness Detector | aware | Security, All | 1 | Detects use of non-cryptographic PRNGs in security-sensitive contexts |
| 26 | Path Traversal Detector | aware | Web, API | 2 | Traces file path construction for directory traversal vulnerabilities |
| 27 | Command Injection Detector | aware | Web, API, Systems | 2 | Flags shell command construction from untrusted input |
| 28 | SSRF Detector | aware | Web, API, Cloud | 2 | Identifies server-side request forgery via URL construction analysis |
| 29 | Deserialization Vulnerability Detector | aware | Web, API | 2 | Flags unsafe deserialization of untrusted data (pickle, yaml.load, etc.) |
| 30 | Authentication Bypass Detector | agnostic | Web, API, Security | 2 | Finds missing auth checks on protected routes and endpoints |
| 31 | CSRF Vulnerability Detector | aware | Web | 3 | Detects missing CSRF tokens on state-mutating endpoints |
| 32 | Dependency Vulnerability Scanner | agnostic | All | 3 | Cross-references dependency versions against CVE databases |

### C05 — Performance & Efficiency (8 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 33 | N+1 Query Detector | aware | Web, API, Database | 1 | Identifies ORM loops that produce N+1 database queries |
| 34 | Unnecessary Recomputation Detector | aware | All | 1 | Finds repeated expensive computations that should be cached or memoized |
| 35 | Blocking I/O in Async Context | aware | Web, API, Distributed | 1 | Flags synchronous I/O calls inside async functions/event loops |
| 36 | Nested Loop Complexity Classifier | aware | All | 1 | Classifies nested loop structures by computational complexity class |
| 37 | GC Pressure Predictor | critical | All | 1 | Predicts garbage collection pressure from allocation patterns (Java, Python, Go) |
| 38 | Memory Allocation Hotspot | critical | Systems, HPC | 2 | Identifies excessive heap allocations in hot paths |
| 39 | Cache Efficiency Analyzer | agnostic | Web, API, Distributed | 3 | Evaluates caching strategy effectiveness and cache invalidation correctness |
| 40 | Algorithmic Complexity Detector | critical | All | 3 | Infers big-O complexity from code structure and flags suboptimal algorithms |

### C06 — Architecture & Design (5 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 41 | SOLID Violations Detector | agnostic | Enterprise, All | 2 | Flags violations of single-responsibility, open-closed, and dependency inversion |
| 42 | Circular Dependency Detector | agnostic | All | 2 | Identifies circular imports/dependencies via module dependency graph |
| 43 | Layer Violation Detector | agnostic | Enterprise, Web | 2 | Detects architectural layer bypasses (e.g., UI calling database directly) |
| 44 | God Class/Module Detector | agnostic | All | 3 | Flags classes/modules with too many responsibilities via coupling metrics |
| 45 | API Contract Consistency | agnostic | API, Web | 3 | Verifies API implementations match declared contracts (OpenAPI, GraphQL schema) |

### C07 — Code Quality & Maintainability (6 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 46 | Cyclomatic Complexity Agent | aware | All | 1 | Computes cyclomatic complexity per function; flags above configurable threshold |
| 47 | Cognitive Complexity Agent | aware | All | 1 | Measures cognitive complexity (nesting, breaks in flow) per function |
| 48 | Long Method Detector | aware | All | 1 | Flags methods exceeding configurable LOC and complexity thresholds |
| 49 | Magic Number Detector | aware | All | 2 | Identifies unexplained numeric/string literals that should be named constants |
| 50 | Code Smell Classifier | agnostic | All | 3 | ML-based classifier for common code smells (feature envy, data clumps, etc.) |
| 51 | Technical Debt Estimator | agnostic | All | 3 | Estimates remediation cost of accumulated quality issues per file/module |

### C08 — Testing (5 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 52 | Test Coverage Gap Analyzer | aware | Quality, All | 2 | Identifies untested code paths via coverage data + changed-line analysis |
| 53 | Assertion Quality Checker | aware | Quality, All | 2 | Flags weak assertions (assertTrue vs assertEquals) and missing assertions |
| 54 | Test Smell Detector | aware | Quality, All | 2 | Detects test anti-patterns: eager test, mystery guest, conditional logic in tests |
| 55 | Mutation Testing Agent | aware | Quality, All | 3 | Runs lightweight mutation testing to find tests that never fail |
| 56 | Flaky Test Predictor | aware | Quality, DevOps | 3 | Predicts test flakiness from code patterns (sleep, network, shared state) |

### C09 — Git & Version Control Intelligence (4 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 57 | Hotspot Detector | agnostic | All | 2 | Identifies files with high churn rate and defect correlation from git history |
| 58 | Co-Change Pattern Miner | agnostic | All | 2 | Mines change coupling: files that always change together but shouldn't |
| 59 | Commit Message Quality | agnostic | DevOps, All | 3 | Evaluates commit messages for clarity, scope, and conventional format |
| 60 | Change Risk Predictor | agnostic | All | 3 | Predicts defect probability for a changeset based on historical patterns |

### C10 — Language-Specific Deep Agents (7 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 61 | Python Type Annotation Checker | critical | All | 2 | Validates type annotation correctness, Optional usage, and mypy compatibility |
| 62 | JavaScript Async/Await Correctness | critical | Web, API | 2 | Detects missing awaits, unhandled rejections, and Promise anti-patterns |
| 63 | TypeScript Type Safety Deep Check | critical | Web, API | 2 | Flags `any` abuse, unsafe casts, and type narrowing gaps |
| 64 | Java Concurrency Correctness | critical | Enterprise, Distributed | 2 | Validates synchronized blocks, volatile usage, and java.util.concurrent patterns |
| 65 | Go Goroutine Safety | critical | Cloud, Distributed, Systems | 2 | Detects goroutine leaks, channel deadlocks, and race-prone patterns |
| 66 | Rust Ownership/Lifetime Validator | critical | Systems | 3 | Flags ownership anti-patterns and unnecessary clones beyond borrow checker |
| 67 | C/C++ Memory Management Auditor | critical | Systems, HPC | 3 | Identifies manual memory management bugs: leaks, dangling pointers, double-free |

### C11 — LLM & Generative AI (3 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 68 | Natural Language Review Generator | agnostic | All | 2 | Generates human-readable review summaries from aggregated findings |
| 69 | Fix Suggestion Generator | agnostic | All | 2 | Produces concrete code fix suggestions using LLM with finding context |
| 70 | Code Documentation Generator | aware | All | 3 | Generates missing docstrings and inline documentation for complex logic |

### C12 — Metrics, Observability & Monitoring (2 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 71 | Logging Best Practices | aware | Cloud, DevOps, All | 3 | Flags missing error logging, PII in logs, inconsistent log levels |
| 72 | Metrics Instrumentation Checker | aware | Cloud, DevOps, Distributed | 3 | Detects missing metrics on critical paths (latency, error rates, throughput) |

### C13 — Graph & Relational Intelligence (2 agents)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 73 | Taint Flow Analyzer | aware | Security, Web, API | 2 | Full taint propagation from sources to sinks via interprocedural data flow |
| 74 | Call Graph Anomaly Detector | agnostic | All | 3 | Identifies unusual call patterns: unreachable functions, unexpected dependencies |

### C14 — Learning & Adaptation (1 agent)

| ID | Agent Name | Type | Domains | Phase | Description |
|---|---|---|---|---|---|
| 75 | Confidence Calibrator | agnostic | All | 3 | RL-based agent that adjusts confidence scores from developer feedback signals |

---

## Phase Roadmap

### Phase 1 — 20 Agents (Foundation)

High signal, low false-positive agents. Ship these first.

| ID | Agent | Cluster | Type |
|---|---|---|---|
| 1 | Dead Code Reachability | C01 Static Analysis | aware |
| 7 | Duplicate Code Detector | C02 Semantic | aware |
| 11 | Null Dereference Predictor | C03 Bug Detection | aware |
| 12 | Resource Leak Detector | C03 Bug Detection | aware |
| 13 | Exception Swallowing Detector | C03 Bug Detection | aware |
| 14 | Off-by-One Error Classifier | C03 Bug Detection | aware |
| 15 | Race Condition Detector | C03 Bug Detection | critical |
| 21 | Hardcoded Secrets Detector | C04 Security | aware |
| 22 | SQL Injection Detector | C04 Security | aware |
| 23 | XSS Pattern Detector | C04 Security | aware |
| 24 | Weak Cipher/Hash Detector | C04 Security | aware |
| 25 | Insecure Randomness Detector | C04 Security | aware |
| 33 | N+1 Query Detector | C05 Performance | aware |
| 34 | Unnecessary Recomputation Detector | C05 Performance | aware |
| 35 | Blocking I/O in Async Context | C05 Performance | aware |
| 36 | Nested Loop Complexity Classifier | C05 Performance | aware |
| 37 | GC Pressure Predictor | C05 Performance | critical |
| 46 | Cyclomatic Complexity Agent | C07 Code Quality | aware |
| 47 | Cognitive Complexity Agent | C07 Code Quality | aware |
| 48 | Long Method Detector | C07 Code Quality | aware |

**Breakdown:** Security (5), Bug Detection (5), Performance (5), Code Quality (3), Static Analysis (1), Semantic (1)

### Phase 2 — +30 Agents (= 50 total)

Expand security coverage, add testing/architecture/git intelligence, language-specific deep agents.

| ID | Agent | Cluster |
|---|---|---|
| 2 | AST Pattern Matching | C01 |
| 3 | Token Sequence Classifier | C01 |
| 8 | Semantic Clone Detection | C02 |
| 16 | Integer Overflow Detector | C03 |
| 17 | Infinite Loop/Recursion Detector | C03 |
| 18 | Uninitialized Variable Detector | C03 |
| 19 | Type Confusion Detector | C03 |
| 26 | Path Traversal Detector | C04 |
| 27 | Command Injection Detector | C04 |
| 28 | SSRF Detector | C04 |
| 29 | Deserialization Vulnerability Detector | C04 |
| 30 | Authentication Bypass Detector | C04 |
| 38 | Memory Allocation Hotspot | C05 |
| 41 | SOLID Violations Detector | C06 |
| 42 | Circular Dependency Detector | C06 |
| 43 | Layer Violation Detector | C06 |
| 49 | Magic Number Detector | C07 |
| 52 | Test Coverage Gap Analyzer | C08 |
| 53 | Assertion Quality Checker | C08 |
| 54 | Test Smell Detector | C08 |
| 57 | Hotspot Detector | C09 |
| 58 | Co-Change Pattern Miner | C09 |
| 61 | Python Type Annotation Checker | C10 |
| 62 | JavaScript Async/Await Correctness | C10 |
| 63 | TypeScript Type Safety Deep Check | C10 |
| 64 | Java Concurrency Correctness | C10 |
| 65 | Go Goroutine Safety | C10 |
| 68 | Natural Language Review Generator | C11 |
| 69 | Fix Suggestion Generator | C11 |
| 73 | Taint Flow Analyzer | C13 |

### Phase 3 — +25 Agents (= 75 total)

Full suite: advanced ML, learning/adaptation, observability, remaining specialized agents.

| ID | Agent | Cluster |
|---|---|---|
| 4 | Identifier Naming Conventions | C01 |
| 5 | Comment-Code Alignment | C01 |
| 6 | Import/Dependency Order | C01 |
| 9 | Code Embedding Similarity | C02 |
| 10 | Intent-Implementation Mismatch | C02 |
| 20 | Memory Safety Violation Detector | C03 |
| 31 | CSRF Vulnerability Detector | C04 |
| 32 | Dependency Vulnerability Scanner | C04 |
| 39 | Cache Efficiency Analyzer | C05 |
| 40 | Algorithmic Complexity Detector | C05 |
| 44 | God Class/Module Detector | C06 |
| 45 | API Contract Consistency | C06 |
| 50 | Code Smell Classifier | C07 |
| 51 | Technical Debt Estimator | C07 |
| 55 | Mutation Testing Agent | C08 |
| 56 | Flaky Test Predictor | C08 |
| 59 | Commit Message Quality | C09 |
| 60 | Change Risk Predictor | C09 |
| 66 | Rust Ownership/Lifetime Validator | C10 |
| 67 | C/C++ Memory Management Auditor | C10 |
| 70 | Code Documentation Generator | C11 |
| 71 | Logging Best Practices | C12 |
| 72 | Metrics Instrumentation Checker | C12 |
| 74 | Call Graph Anomaly Detector | C13 |
| 75 | Confidence Calibrator | C14 |

---

## Language Expansion

### Language Tiers

| Tier | Languages | Priority |
|---|---|---|
| Tier 1 | Python, JavaScript, TypeScript, Java, Go | Build first |
| Tier 2 | Rust, C#, C/C++, Swift, Kotlin | Build second |
| Tier 3 | Ruby, PHP, Scala, R, Dart, Elixir, Haskell, OCaml, Clojure | Community-driven |

### Expansion from 75 Base Agents to ~1,080 Instances

| Agent Type | Base Count | Tier 1 (5 langs) | Tier 2 (5 langs) | Tier 3 (9 langs) | Total Instances |
|---|---|---|---|---|---|
| Agnostic | 19 | — | — | — | 19 |
| Aware | 43 | 215 | 215 | 387 | 817 |
| Critical | 13 | 65 | 65 | 117 | 247 |
| **Total** | **75** | **280** | **280** | **504** | **1,083** |

Agnostic agents are built once. Aware agents have one base implementation plus a language adapter per target language. Critical agents require full per-language implementations.

### Phased Language Rollout

| Phase | Languages | Aware Instances | Critical Instances | Agnostic | Total |
|---|---|---|---|---|---|
| Phase 1 (Tier 1 only) | Py, JS, TS, Java, Go | 100 | 5 | 19 | 124 |
| Phase 2 (+ Tier 1 complete) | Py, JS, TS, Java, Go | 215 | 65 | 19 | 299 |
| Phase 3 (+ Tier 2) | + Rust, C#, C/C++, Swift, Kotlin | 430 | 130 | 19 | 579 |
| Full (+ Tier 3) | All 19 languages | 817 | 247 | 19 | 1,083 |

---

## Implementation Template

### Directory Structure per Agent

```
agents/cluster_XX_<name>/<agent_name>/
├── agent.py           # BaseReviewAgent implementation
├── patterns/          # Language-specific pattern files (aware/critical agents)
│   ├── python.py
│   ├── javascript.py
│   └── ...
├── tests/
│   ├── test_agent.py  # Min 10 positive + 10 negative cases
│   ├── fixtures/      # Test input files
│   └── ...
├── benchmarks/        # Benchmark dataset results
│   └── results.json
└── README.md          # What this agent catches and why
```

### Required `agent.py` Structure

```python
from core.agent.base import BaseReviewAgent
from core.schema.metadata import AgentMetadata
from core.schema.finding import Finding, CodeFix
from core.context.code_context import CodeContext


class <AgentName>(BaseReviewAgent):
    def metadata(self) -> AgentMetadata:
        return AgentMetadata(
            name="<agent_name>",
            version="0.1.0",
            languages=["python", "javascript", ...],  # or ["*"] for agnostic
            domains=["web", "api", ...],
            methodology="cluster_XX",
            axis_type="aware",  # "agnostic" | "aware" | "critical"
            tags=["security", "bug", ...],
            model_required=False,
            estimated_cost_cents=0.0,
        )

    def analyze(self, context: CodeContext) -> list[Finding]:
        ...

    def explain(self, finding: Finding) -> str:
        ...
```

### AgentMetadata Field Guide

| Field | Description | Examples |
|---|---|---|
| `name` | Snake_case unique identifier | `hardcoded_secrets_detector` |
| `version` | Semver starting at 0.1.0 | `0.1.0` |
| `languages` | Target languages or `["*"]` | `["python", "javascript"]` |
| `domains` | Domain clusters this agent covers | `["web", "api", "security"]` |
| `methodology` | Parent methodology cluster | `cluster_04` |
| `axis_type` | Agent type classification | `"aware"` |
| `tags` | Searchable tags for routing | `["security", "injection"]` |
| `model_required` | Whether ML model inference is needed | `False` |
| `estimated_cost_cents` | Estimated cost per invocation | `0.0` (static), `5.0` (LLM) |

### Cost Budget Guidelines

| Tier | Agent Types | Max Cost/Invocation | Budget Ceiling/Review |
|---|---|---|---|
| Free | Static analysis only (no `model_required`) | 0.0 cents | $0.00 |
| Pro | Static + lightweight ML | 1.0 cents | $0.10 |
| Team | Static + ML + selective LLM | 5.0 cents | $0.50 |
| Enterprise | All agents including LLM-backed | 10.0 cents | $2.00 |

---

## Verification Checklist

- [x] 75 agents with unique sequential IDs (1–75)
- [x] 14 active clusters, each with at least 1 agent
- [x] Phase 1 matches the 20 agents from CLAUDE.md
- [x] Type distribution: 19 agnostic (25%), 43 aware (57%), 13 critical (17%)
- [x] Phase breakdown: 20 → 50 → 75
- [x] Language expansion: 75 base → 1,083 instances
