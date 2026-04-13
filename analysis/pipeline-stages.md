# Daily Pipeline Stage Analysis

## Stage-by-Stage Breakdown

### Stage 1: User Scoring (`scoring.py`)
- **Type**: PySpark (32 executors x 8G)
- **Duration**: ~15 min
- **Function**: Composite engagement scoring
  ```
  S(u) = 0.2×norm(click) + 0.3×norm(review) + 0.15×norm(save) + 0.2×D(u) + 0.15×R(u)
  ```
  - D(u): category diversity score
  - R(u): recency score
- **Output**: Top 10K users by score → `hdfs://.../scored_users/{datestr}`
- **Bottleneck**: None (fast)

### Stage 2: Hash Check (`hash_tracker.py`)
- **Type**: PySpark (16 executors x 4G)
- **Duration**: ~5 min
- **Function**: SHA256 hash of (etimestamp, poi_id) tuples per user. Compare vs previous day's hash
- **Output**: changed_ids + skipped_ids → JFS
- **Design**: Mem0-inspired dedup — skip users with no behavioral change
- **Bottleneck**: None

### Stage 3a: Implicit Preprocessing (`data.py`)
- **Type**: PySpark (32 executors x 8G)
- **Duration**: ~10 min
- **Function**: Join click/save logs with POI metadata + user demographics
  - 21-day sliding window
  - LEFT SEMI JOIN on scored user IDs
- **Output**: Parquet → `jfs://.../implicit_data/{datestr}`

### Stage 3b: Explicit Preprocessing (`explicit_preprocessing.py`)
- **Type**: PySpark (16 executors x 4G)
- **Duration**: ~10 min
- **Function**: Review text quality scoring + sampling
  - Quality weights: TEXT(0.3), ENGAGEMENT(0.2), SOURCE(0.15), OPINION(0.15), MENU(0.1), DECAY(0.1)
  - Dedup by review_group_id
  - Min filters: 3 reviews, 30 chars, 2 categories
- **Output**: Parquet → `jfs://.../explicit_data/{datestr}`

### Stage 4a: Implicit Extraction (`async_run.py`)
- **Type**: K8s Pod (16Gi mem)
- **Duration**: ~25 min (LLM bottleneck)
- **Function**: Async LLM batch calls — raw click/save logs → structured preferences
- **LLM Model**: qwen3-235b-a22b-instruct-2507
- **Token Usage**: Avg 73K prompt tokens/user
- **Output**: JSONL → `jfs://.../output/implicit_data/{datestr}`
- **Bottleneck**: **Primary bottleneck** — LLM API throughput + token cost

### Stage 4b: Explicit Extraction (`explicit_extract.py`)
- **Type**: K8s Pod (4Gi mem)
- **Duration**: ~20 min
- **Function**: Async LLM batch calls — review text → structured insights
- **Token Usage**: Avg 5.3 chunks/user, ~5.3M total tokens
- **Output**: JSONL → `jfs://.../output/explicit_data/{datestr}`

### Stage 5: Memory Update (`build_memories.py`)
- **Type**: K8s Pod (4Gi mem), ProcessPoolExecutor (concurrency=4)
- **Duration**: ~15 min
- **Function**: 3-step LLM chain per user
  1. extract_preferences → 3-axis separation
  2. match_paths → hierarchical routing
  3. update_note → per-path note editing
- **LLM Calls**: 3 + N calls per user (N = number of assigned paths)
- **Output**: notes.parquet + hierarchy.json per user

## Resource Summary

| Stage | Compute | Memory | Duration | Cost Driver |
|-------|---------|--------|----------|-------------|
| Scoring | Spark 32x8G | 256G total | 15m | Spark cluster |
| Hash Check | Spark 16x4G | 64G total | 5m | Spark cluster |
| Implicit Preproc | Spark 32x8G | 256G total | 10m | Spark cluster |
| Explicit Preproc | Spark 16x4G | 64G total | 10m | Spark cluster |
| Implicit Extract | K8s 16Gi | 16Gi | 25m | **LLM API tokens** |
| Explicit Extract | K8s 4Gi | 4Gi | 20m | **LLM API tokens** |
| Memory Update | K8s 4Gi | 4Gi | 15m | **LLM API calls** |
| **Total** | - | - | **~1h 10m** | - |

## Scaling Bottleneck Analysis

Current: 100 target users → ~1h 10m
Target: 10K users/day, eventually 1M users

| Stage | 100 users | 10K users | Scaling Strategy |
|-------|-----------|-----------|------------------|
| Scoring | 15m | ~15m (Spark scales) | Already parallelized |
| Hash Check | 5m | ~5m (Spark scales) | Already parallelized |
| Implicit Extract | 25m | ~42h (!!) | PersonaX clustering (7x), batching, KD model |
| Explicit Extract | 20m | ~33h (!!) | Batching, KD model |
| Memory Update | 15m | ~25h (!!) | Reduce LLM calls, incremental update |

**Critical path for scaling**: LLM stages (4a, 4b, 5) — must reduce token usage and call count.
