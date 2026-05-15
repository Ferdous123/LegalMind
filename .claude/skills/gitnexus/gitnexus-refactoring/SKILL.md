---
name: gitnexus-refactoring
description: "Use when renaming, moving, or restructuring code. Ensures all references are updated. Examples: 'Rename this class', 'Move this module', 'Extract this into a separate file'"
---

# Safe Refactoring with GitNexus

## When to Use

- Renaming a function, class, or module
- Moving code between files
- Extracting code into a new module
- Changing function signatures

## Workflow

```
1. gitnexus_impact({target: "old_name", direction: "upstream"})  → Find all callers
2. gitnexus_impact({target: "old_name", direction: "downstream"}) → Find all dependencies
3. Make the change
4. Update ALL d=1 references
5. gitnexus_detect_changes() → Verify no broken links
6. Run: python -c "from code.{module} import {Class}; print('OK')"
```

## LegalMind Module Import Map

```python
# Core (everything imports from these)
from config.paths import *
from code.llm_interface import ModelManager, InferenceEngine

# Pipeline (imports llm_interface)
from code.pipeline import DocumentIngester, ProcessedDocument

# Retrieval (imports llm_interface for embeddings)
from code.retrieval import DocumentIndexer, EvidenceSearcher, EvidencePackager

# Generation (imports retrieval + learning)
from code.generation import DraftGenerator, DraftOutput

# Learning (imports llm_interface + retrieval/chromadb)
from code.learning import CorrectionStore, ExemplarRetriever, PatternExtractor, PromptConsolidator

# Firewall (imports llm_interface for embeddings)
from code.firewall import FirewallRunner, VerificationResult

# Webapp (imports everything above)
from webapp.main import app
```

## Safety Rules

- NEVER rename without checking gitnexus_impact first
- ALWAYS update __init__.py __all__ lists after moving exports
- ALWAYS run import verification after refactoring
- If >10 files affected: do it in multiple commits
