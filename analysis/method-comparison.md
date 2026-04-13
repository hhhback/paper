# Method Comparison: Our Pipeline vs. Existing Approaches

## Architecture Comparison Table

| Feature | **Ours** | PersonaMem-v2 | Mem0 | A-Mem | EMem |
|---------|----------|---------------|------|-------|------|
| Memory Structure | 3-level hierarchy | 3-level hierarchy | Flat list | Hierarchy + links | Structured |
| Implicit Extraction | O (click/save logs) | O (behavior signals) | X | X | X |
| Explicit Extraction | O (review text) | O (dialogue text) | O (conversation) | O (conversation) | O (conversation) |
| Dual Extraction | **O** | **O** | X | X | X |
| Forward Pass | extract → match → update | extract → classify → store | ADD/UPDATE | insert → activate | expect → store |
| Backward Pass | merge + lift + prune | consolidation | DELETE | merge + link + prune | forget |
| Dynamic Path Creation | **O (LLM-based)** | X (predefined slots) | N/A (flat) | Partial | X |
| Cross-Node Links | X | X | Graph memory | **O (Aha Moment)** | X |
| Common Trait Lifting | **O (lift_common_traits)** | X | X | X | X |
| Hash-based Dedup | O (hash_tracker) | X | O | X | X |
| 3-axis Separation | **O (user_info/explicit/implicit)** | 2-axis (explicit/implicit) | 1-axis | 1-axis | 1-axis |
| Incremental Update | Partial (hash check) | X (full rebuild) | O (per-memory ops) | O | O |
| Evaluation | Manual | PersonaGym-style | LoCoMo, LOCOMO | Custom | LoCoMo |

## Forward Pass Detail Comparison

### Our Pipeline (3-step LLM chain)
```
interaction_text
  → [1] extract_preferences (LLM)
      Output: {user_info, explicit_preferences, implicit_preferences}
  → [2] match_paths (LLM)
      Input: extracted + base_hierarchy + existing_paths
      Output: [{target_path, user_info, explicit, implicit}]
  → [3] update_note (LLM, per path)
      Input: current_note + new_data_package
      Output: {user_info, summary, keywords}
```

### PersonaMem-v2 (2-step)
```
interaction_text
  → [1] extract + classify (single LLM call)
      Output: {aspect: sub-aspect: preference}
  → [2] store to predefined tree slot
```

### Mem0 (2-step)
```
conversation
  → [1] extract facts/preferences
  → [2] compare with existing → ADD/UPDATE/DELETE/NO_CHANGE
```

### A-Mem (multi-step)
```
interaction
  → [1] extract key info
  → [2] insert into hierarchy
  → [3] activate related nodes (Aha Moment)
  → [4] create cross-links if activation threshold met
```

## Backward Pass Comparison

### Our Pipeline
1. **merge_siblings** — LLM judges semantic similarity, merges leaf nodes
2. **lift_common_traits** — Extract shared traits from children → parent (NOVEL)
3. **prune_stale_leaves** — Remove low-update + old nodes

### A-Mem
1. **merge** — Combine similar nodes
2. **link** — Cross-category connections
3. **prune** — Remove stale

### Key Difference
Our `lift_common_traits` is structurally unique — it enforces information-theoretic efficiency by deduplicating shared attributes upward. No surveyed paper implements this specific operation.

## What We Lack (Gap Analysis)

| Gap | Source Paper | Impact | Effort |
|-----|-------------|--------|--------|
| Cross-node activation/linking | A-Mem | Captures "coffee lover + morning person → cafe recommendations" connections | Medium |
| Incremental per-memory UPDATE/DELETE | Mem0 | Avoids full rebuild on each interaction | Medium |
| Input token reduction via clustering | PersonaX | 73K → ~10K tokens per user | Low-Medium |
| Automated evaluation framework | PersonaGym | Quantitative quality measurement | Medium |
| KD fine-tuning (4B model) | PersonaMem-v2 | 50x cost reduction | High |
| Expectation-guided selective storage | EMem | Stores only surprising/useful info | Medium |

## Our Novel Contributions

1. **Dynamic Hierarchical Routing (match_paths)**: LLM dynamically routes preferences to existing paths OR creates new paths while maintaining level consistency — unlike PersonaMem-v2's fixed slots
2. **Common Trait Elevation (lift_common_traits)**: Backward pass operation that promotes shared attributes from sibling children to parent node — novel structural maintenance
3. **3-axis Preference Separation**: Explicit separation of demographics (user_info, root-only) from explicit and implicit preferences — cleaner information architecture than 2-axis approaches
4. **Production Pipeline Integration**: Full Spark + Argo + LLM async pipeline — not just an algorithm but a deployable daily system with hash-based incremental processing
