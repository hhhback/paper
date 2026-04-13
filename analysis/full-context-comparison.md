# Full-Context vs. Structured Memory: Analysis

## What is Full-Context?

Full-context means passing the **entire raw interaction history** (clicks, reviews, saves) directly into LLM context window with no extraction, summarization, or memory structure. The LLM processes everything at once.

## Key Benchmarks

### Factual QA Tasks (LoCoMo Benchmark)
| Method | Accuracy | Tokens Used | Source |
|--------|----------|-------------|--------|
| Long-context GPT-5-mini | **92.85%** | Full history | Beyond the Context Window |
| EMem | **78.0%** | ~1/32 of full | EMem paper |
| ReadAgent | 72.3% | Compressed | EMem paper |
| Mem0 | 57.68% | Memory only | Beyond the Context Window |
| MemGPT | 48.23% | Paged memory | Beyond the Context Window |

**Verdict**: For factual recall ("what restaurant did I mention?"), full-context wins decisively.

### Implicit Personalization Tasks (PersonaMem-v2)
| Method | Win Rate | Tokens Used | Source |
|--------|----------|-------------|--------|
| PersonaMem-v2 (agentic memory) | **55.2%** | ~2K | PersonaMem-v2 |
| Full-context (32K window) | 45.6% | ~32K | PersonaMem-v2 |

**Verdict**: For implicit preference modeling, structured memory beats full-context — LLM can't reliably extract scattered signals from raw logs.

## Why Full-Context Loses for Our Use Case

Our implicit extraction averages **73K tokens/user** of raw click/save logs. Problems with full-context:

1. **Signal Dilution**: Preferences scattered across 1000+ click events. LLM attention dilutes over noise
2. **Cost**: 73K input tokens × 10K users/day × $15/M tokens = **$10,950/day** (vs ~$200 with structured memory)
3. **Latency**: 73K token processing = 30-60s per user. Structured memory retrieval = <1s
4. **No Persistence**: Full-context requires reprocessing entire history each time. Memory accumulates incrementally
5. **Context Window Limits**: 73K already pushes limits; users with 6+ months of history could exceed 200K+

## Where Full-Context Wins

1. **Quality upper bound for factual tasks** — "Did I save this restaurant?" → full-context is 35% better than best memory system
2. **Zero-engineering baseline** — No extraction pipeline needed, just dump and query
3. **One-shot analysis** — First-time user profiling where no memory exists yet

## Recommendation for Our Pipeline

```
                        Full-Context          Structured Memory
                        ─────────────         ──────────────────
Quality (factual QA)    ████████████  92%     ██████████  78%
Quality (persona)       ██████████  46%       ████████████  55%
Cost efficiency         ██  ~$11K/day         ████████████  ~$200/day
Latency                 ██  30-60s/user       ████████████  <1s
Scalability             ██  1K users max      ████████████  1M+ users
Incrementality          X                     ████████████

→ Structured memory is the right choice for production POI recommendation.
→ Full-context as quality validation baseline only.
```

## Hybrid Opportunity

EMem's insight: use full-context as a **teacher** to train structured memory:
1. Run full-context on sample users → get ground truth preferences
2. Run structured memory on same users → get memory-based preferences
3. Measure gap → improve extraction/routing prompts
4. Iterate until memory quality matches full-context quality
