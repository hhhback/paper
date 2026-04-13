# Our Method: Hierarchical Agentic Memory for POI Recommendation

## System Overview

Production-grade daily pipeline for building and maintaining hierarchical user memory from implicit (click/save) and explicit (review) interaction signals, designed for Naver Maps POI recommendation.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Daily Pipeline (Argo)                        │
│                                                                     │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐             │
│  │ Scoring  │───▶│ Hash Check│───▶│ Changed User IDs │             │
│  │ (Spark)  │    │  (Spark)  │    └────────┬─────────┘             │
│  └────┬─────┘    └───────────┘             │                       │
│       │                                     │                       │
│       ▼                                     ▼                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ Implicit Preproc│  │ Explicit Preproc│  │   Memory Update  │   │
│  │   (Spark)       │  │   (Spark)       │  │    (K8s Pod)     │   │
│  └───────┬─────────┘  └───────┬─────────┘  │                  │   │
│          │                     │             │ extract_prefs    │   │
│          ▼                     ▼             │    ↓             │   │
│  ┌───────────────┐  ┌───────────────┐       │ match_paths      │   │
│  │Implicit Extract│  │Explicit Extract│      │    ↓             │   │
│  │  (Async LLM)  │  │  (Async LLM)  │──────▶ update_note      │   │
│  └───────────────┘  └───────────────┘       └──────────────────┘   │
│                                                                     │
│  Weekly: merge_siblings → lift_common_traits → prune_stale_leaves  │
│  Monthly: generate_persona                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Memory Structure (Per User)

```
사용자 (Root, L1)
├── user_info: [age, gender, region, job, ...]    ← Root-only demographics
├── summary: "전반적 선호 요약"
├── keywords: [...]
│
├── 음식 (L2)
│   ├── summary: "음식 카테고리 공통 선호"
│   ├── 한식 (L3)
│   │   ├── summary: "한식 세부 선호"
│   │   └── keywords: [...]
│   └── 카페/디저트 (L3)
│       ├── summary: "카페 세부 선호"
│       └── keywords: [...]
│
├── 쇼핑 (L2)
│   └── ...
└── 여행 (L2)
    └── ...
```

### Storage Format
- `notes.parquet`: idhash, path, level, description, summary, keywords, user_info, updated_count, recently_updated
- `hierarchy.json`: tree structure with nodes {path, level, parent_id, children[]}
- `hierarchy_tree.json`: nested tree for visualization

## Forward Pass: 3-Step LLM Chain

### Step 1: Extract Preferences
- **Input**: Raw interaction text (implicit clicks/saves + explicit reviews)
- **Output**: `{user_info: [], explicit_preferences: [], implicit_preferences: []}`
- **Key Design**: 3-axis separation — demographics vs stated preferences vs behavioral signals

### Step 2: Match Paths (Hierarchical Routing)
- **Input**: Extracted preferences + Base Hierarchy + Existing paths
- **Output**: `[{target_path, user_info, explicit_preferences, implicit_preferences}]`
- **Key Design**: Dynamic path creation with level-consistency enforcement
  - Reuse existing paths when semantically compatible
  - Create new paths matching sibling abstraction level
  - Strict 3-level max depth
  - Root-only user_info routing

### Step 3: Update Note
- **Input**: Current note content + new data package + path context
- **Output**: `{user_info, summary, keywords}`
- **Key Design**: Edit, not append — LLM merges/resolves conflicts with existing content

## Backward Pass: 3-Stage Cleanup (Weekly)

### Stage 1: Merge Siblings
- LLM identifies semantically overlapping leaf nodes under same parent
- Merges with content integration (not concatenation)
- Supports partial merge (split content across groups)

### Stage 2: Lift Common Traits (Novel)
- Identifies traits shared by 2+ children
- Promotes to parent note, removes from children
- Enforces information-theoretic efficiency

### Stage 3: Prune Stale Leaves
- Removes nodes with: low update count (<=2) AND old timestamp (>90 days)
- Deepest-first traversal to handle cascading empty parents

## Differentiation from PersonaMem-v2

| Aspect | PersonaMem-v2 | Ours |
|--------|---------------|------|
| Path management | Fixed slots | Dynamic creation + level enforcement |
| Backward cleanup | Generic consolidation | 3-stage: merge + lift + prune |
| Input source | Dialogue text | Click/save logs + review text (dual) |
| Demographics | Mixed with preferences | Isolated to root (3-axis) |
| Scale | Research prototype | Production pipeline (Spark + Argo + async LLM) |
| Incremental | Full rebuild | Hash-based change detection |
