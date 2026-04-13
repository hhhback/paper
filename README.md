# Hierarchical Agentic Memory for POI Recommendation

POI 추천을 위한 계층적 에이전트 메모리 시스템 — 연구 자료 및 분석 정리

## Directory Structure

```
paper/
├── README.md                              ← 이 파일
├── references/
│   └── persona-memory.md                  ← 논문 15편 + 검색/리랭킹 6편 레퍼런스 정리
├── analysis/
│   ├── method-comparison.md               ← Our method vs PersonaMem-v2/Mem0/A-Mem/EMem 비교표
│   ├── full-context-comparison.md         ← Full-context vs Structured memory 분석
│   └── pipeline-stages.md                 ← Daily pipeline 스테이지별 분석 + 스케일링 병목
├── our-method/
│   ├── architecture.md                    ← 시스템 아키텍처 + Forward/Backward pass 상세
│   └── improvement-roadmap.md             ← 논문 기반 개선 로드맵 (Phase 1-3)
└── figures/                               ← 다이어그램, 실험 결과 그래프 (TBD)
```

## Research Summary

### Position
PersonaMem-v2 (AAAI 2025) 계열의 **계층적 메모리 시스템**을 POI 추천 도메인에 적용한 production pipeline. 기존 연구 대비 3가지 차별점:

1. **Dynamic Hierarchical Routing** — LLM이 기존 경로를 재사용하거나 level-consistency를 유지하며 새 경로를 동적 생성
2. **Common Trait Elevation** — Backward pass에서 자식 노드의 공통 특성을 부모로 승격하는 연산 (기존 논문에 없음)
3. **3-axis Preference Separation** — 인구통계(user_info) / 명시적 선호 / 암묵적 선호를 분리하고, 인구통계는 Root 전용

### Pipeline
```
Scoring → Hash Check → Implicit/Explicit Preprocess → LLM Extract → Memory Update
(Spark)   (Spark)       (Spark)                       (Async LLM)   (3-step LLM chain)
```
- Daily 2AM, ~1h 10m for 100 users
- Weekly: merge_siblings → lift_common_traits → prune_stale_leaves
- Monthly: persona generation

### Key Papers

| Paper | Relevance | Key Takeaway |
|-------|-----------|-------------|
| **PersonaMem-v2** | Most similar architecture | 3-level tree + dual extraction, our baseline framework |
| **Mem0** | Hash-based dedup, incremental ops | ADD/UPDATE/DELETE per memory item |
| **A-Mem** | Cross-category linking | Aha Moment for inter-node connections (we lack this) |
| **EMem** | Quality benchmark | First memory to beat full-context (78% vs 72.3%) |
| **PersonaX** | Token reduction | Clustering reduces input 7x — top priority optimization |
| **PersonaGym** | Evaluation framework | 200 personas x 10K questions, 6 evaluation dimensions |
| **ZeroPOIRec** | Domain validation | LLM personas for POI recommendation — validates our direction |
| **Beyond Context Window** | Full-context baseline | Full-context wins factual QA (92.85%) but loses persona (45.6% vs 55.2%) |

전체 15편 + 검색/리랭킹 6편은 `references/persona-memory.md` 참조.

### Improvement Priority

| # | Task | Reference | Impact | Effort |
|---|------|-----------|--------|--------|
| 1 | Input token 축소 (PersonaX clustering) | PersonaX | 7x cost/latency reduction | Low |
| 2 | 평가 프레임워크 구축 | PersonaGym | Enables data-driven iteration | Medium |
| 3 | Incremental update (per-note ops) | Mem0 | Faster updates | Medium |
| 4 | Cross-category linking | A-Mem | Richer persona | Medium |
| 5 | 4B KD fine-tuning | PersonaMem-v2 | 50x cost reduction | High |

상세 로드맵은 `our-method/improvement-roadmap.md` 참조.
