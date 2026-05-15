---
name: gitnexus-impact-analysis
description: "Use when modifying code to understand what depends on it. Examples: 'What breaks if I change X?', 'Show dependencies', 'Is it safe to refactor this?'"
---

# Impact Analysis with GitNexus

## When to Use

- Before modifying any function, class, or module
- "What depends on this?"
- "What will break if I change X?"
- Before committing — to understand blast radius

## Workflow

```
1. gitnexus_impact({target: "ClassName.method", direction: "upstream"})  → What depends on this
2. READ gitnexus://repo/legalmind/processes                              → Check affected flows
3. gitnexus_detect_changes()                                             → Map staged changes to affected code
4. Assess risk and report
```

> If "Index is stale" → run `npx gitnexus analyze` in terminal first.

## Understanding Output

| Depth | Risk Level       | Meaning                  |
| ----- | ---------------- | ------------------------ |
| d=1   | **WILL BREAK**   | Direct callers/importers |
| d=2   | LIKELY AFFECTED  | Indirect dependencies    |
| d=3   | MAY NEED TESTING | Transitive effects       |

## Key Dependency Chains in LegalMind

```
ModelManager → InferenceEngine → OCREngine, DocumentStructurer, DraftGenerator, PatternExtractor
CorrectionStore → ExemplarRetriever → DraftGenerator
EvidenceSearcher → DraftGenerator
FirewallRunner → SourceAnchorVerifier, ConfidenceScorer
DocumentIngester → TextExtractor, OCREngine, DocumentStructurer, TextChunker
```

If modifying ModelManager or InferenceEngine: HIGH risk — nearly everything depends on them.
If modifying a single draft prompt: LOW risk — only affects that draft type's output.
