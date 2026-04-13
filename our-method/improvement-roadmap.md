# Improvement Roadmap

Prioritized by impact/effort ratio based on paper survey findings.

## Phase 1: Quick Wins (1-2 weeks each)

### 1.1 Input Token Reduction — PersonaX Clustering
- **Problem**: Implicit extraction averages 73K tokens/user (expensive, slow)
- **Solution**: Cluster similar click/save events before LLM call
  - Group by category → representative summary per cluster
  - Expected: 73K → ~10K tokens (7x reduction)
- **Reference**: PersonaX (behavior clustering)
- **Impact**: Cost/latency reduction, no quality loss expected
- **Files to modify**: `src/user/memory/async_run.py`, `src/user/memory/prompt_extract.py`

### 1.2 Evaluation Framework
- **Problem**: No automated quality measurement
- **Solution**: PersonaGym-style evaluation adapted for POI recommendation
  - Generate persona-conditioned questions ("Given this user's memory, would they prefer A or B?")
  - Compare predictions against actual future behavior
- **Reference**: PersonaGym, PersonaBench
- **Impact**: Enables data-driven iteration on all subsequent improvements
- **New files**: `src/user/memory/evaluation/`

## Phase 2: Architecture Improvements (2-4 weeks each)

### 2.1 Incremental Memory Update — Mem0-style Operations
- **Problem**: Currently rebuilds full memory on each run (for changed users)
- **Solution**: Per-note ADD/UPDATE/DELETE operations instead of full rebuild
  - Compare new extraction against existing notes
  - Only modify changed notes
- **Reference**: Mem0 (ADD/UPDATE/DELETE/NO_CHANGE)
- **Impact**: Faster updates, preserves stable memory content
- **Files to modify**: `src/user/memory/memory_agent/agent.py`

### 2.2 Cross-Category Linking — A-Mem Activation
- **Problem**: No connections between related preferences across categories (e.g., "morning person" + "cafe lover" → "morning cafe recommendations")
- **Solution**: After note update, check activation against all other notes. Create bidirectional links above threshold
- **Reference**: A-Mem (Aha Moment)
- **Impact**: Richer persona, better recommendation context
- **Files to modify**: `src/user/memory/memory_agent/agent.py`, `storage.py`

### 2.3 Hybrid Retrieval for Note Selection
- **Problem**: Current keyword-based (BM25) retrieval may miss semantic matches
- **Solution**: Dense embedding (E5/ModernBERT) + BM25 hybrid with RRF
- **Reference**: E5, RRF, ReaRANK
- **Impact**: Better note selection for persona generation and downstream retrieval
- **New files**: `src/user/memory/memory_agent/retrieval.py`

## Phase 3: Advanced (1-2 months)

### 3.1 Knowledge Distillation — 4B Fine-tuned Model
- **Problem**: qwen3-235b is expensive ($15/M tokens), high latency
- **Solution**: Distill memory operations to 4B parameter model
  - Teacher: qwen3-235b generates training pairs
  - Student: 4B model fine-tuned on (input, memory_output) pairs
- **Reference**: PersonaMem-v2 (55% win rate with 4B vs GPT-4)
- **Impact**: 50x cost reduction, 10x latency improvement
- **Prerequisites**: Evaluation framework (Phase 1.2) must exist first

### 3.2 Expectation-Guided Storage — EMem
- **Problem**: Currently stores all extracted preferences equally
- **Solution**: Score each preference by surprise/utility before storing
  - Expected preferences (consistent with existing memory) → lower priority
  - Surprising preferences (contradicts or novel) → higher priority
- **Reference**: EMem
- **Impact**: Higher signal-to-noise in memory content

### 3.3 Full-Context Validation Baseline
- **Problem**: No ground truth quality reference
- **Solution**: Run full-context on sample users as teacher
  - Compare memory-based persona vs full-context persona
  - Identify systematic gaps in extraction/routing
- **Reference**: Beyond the Context Window
- **Impact**: Continuous quality calibration
