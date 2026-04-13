# Persona & Memory Systems — Paper References

## Tier 1: Core Architecture (Hierarchical / Structured Memory)

### PersonaMem-v2
- **Title**: PersonaMem-v2: Hierarchical Agentic Memory for Personalized LLM Agents
- **Authors**: Xinyuan Hu, Haolun Wu, Yujia Hu, et al.
- **Venue**: AAAI 2025
- **Links**: [arXiv](https://arxiv.org/abs/2502.12068)
- **Key Contribution**: 3-level hierarchical memory tree (User > Aspect > Sub-aspect), implicit/explicit dual extraction, forward+backward pass, 4B KD fine-tuning beats GPT-4 at 1/50 cost
- **Relevance**: **Most similar to our method** — same paradigm (hierarchical tree + dual extraction + forward/backward pass). Our differences: dynamic path creation, lift_common_traits, 3-axis separation (user_info/explicit/implicit)

### PersonaMem
- **Title**: PersonaMem: A Structured Memory Approach for Personalized Language Model Agents
- **Authors**: Xinyuan Hu, et al.
- **Venue**: Preprint 2024
- **Links**: [arXiv](https://arxiv.org/abs/2410.XXXXX)
- **Key Contribution**: First version of hierarchical persona memory with aspect-level organization
- **Relevance**: Predecessor to PersonaMem-v2; established the core framework we build upon

### A-Mem (Agentic Memory)
- **Title**: A-Mem: Agentic Memory for LLM Agents
- **Authors**: Wujiang Xu, Zujie Liang, et al.
- **Venue**: Preprint 2025
- **Links**: [arXiv](https://arxiv.org/abs/2502.12110)
- **Key Contribution**: Aha Moment — cross-category linking via activation-based interconnection. Hierarchical notes + backward cleanup (merge, link, prune)
- **Relevance**: Cross-category linking is a potential extension for our method. Currently we lack inter-node connections

### Mem0
- **Title**: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
- **Authors**: Mem0.ai Team (Deshraj Yadav, et al.)
- **Venue**: Preprint 2025
- **Links**: [arXiv](https://arxiv.org/abs/2504.19413)
- **Key Contribution**: Flat memory list with ADD/UPDATE/DELETE/NO_CHANGE operations, hash-based dedup, graph memory layer
- **Relevance**: Our hash_tracker uses similar hash-based change detection. Their UPDATE/DELETE operations inform incremental memory management

### EMem
- **Title**: EMem: Expectation-Guided Memory for Long-Horizon Agents
- **Authors**: Various
- **Venue**: Preprint 2025
- **Links**: [arXiv](https://arxiv.org/abs/2511.17208)
- **Key Contribution**: First memory system to beat full-context on quality (78% vs 72.3% on LoCoMo) with 32x fewer tokens. Expectation-guided selective storage
- **Relevance**: Validates that structured memory can outperform full-context. Benchmark reference for quality evaluation

---

## Tier 2: Persona Generation & Evaluation

### PersonaHub
- **Title**: PersonaHub: A Large-Scale World Simulator for Persona-Driven LLM Agents
- **Authors**: Ge Liu, et al.
- **Venue**: Preprint 2024
- **Links**: [arXiv](https://arxiv.org/abs/2407.18899)
- **Key Contribution**: 1B+ diverse persona dataset from web text via persona-driven synthesis. Text-to-Persona and Persona-to-Persona methods
- **Relevance**: Scalable persona generation methodology; could validate diversity of our generated personas

### PersonaGym
- **Title**: PersonaGym: A Dynamic Evaluation Framework for Persona Agents
- **Authors**: Vinay Samuel, et al.
- **Venue**: Preprint 2024
- **Links**: [arXiv](https://arxiv.org/abs/2407.18416)
- **Key Contribution**: 200 personas x 10K questions, 6 evaluation dimensions, dynamic behavior prediction eval
- **Relevance**: Evaluation framework for our persona generation quality. Could adapt their metrics for POI recommendation persona evaluation

### PersonaBench
- **Title**: PersonaBench: Benchmarking Persona-Based Language Agents
- **Authors**: Various
- **Venue**: Preprint 2024
- **Links**: [arXiv](https://arxiv.org/abs/2407.17960)
- **Key Contribution**: Multi-turn persona consistency evaluation, role-playing quality metrics
- **Relevance**: Supplementary evaluation benchmark; tests persona consistency over multi-turn interactions

### SynthesizeMe
- **Title**: SynthesizeMe: Personalized Language Model Adaptation via Profile-Conditioned Synthesis
- **Authors**: Various
- **Venue**: Preprint 2024
- **Key Contribution**: Profile-conditioned synthetic data generation for personalization
- **Relevance**: Alternative approach to persona-conditioned generation; useful for synthetic training data

---

## Tier 3: Domain-Specific & Behavioral

### DEEPER
- **Title**: DEEPER: Dense Extraction of Personalized Expressions for Recommendations
- **Authors**: Various
- **Venue**: Preprint 2024
- **Key Contribution**: Dense persona extraction from user reviews for recommendation systems
- **Relevance**: Review-based preference extraction methodology; comparable to our explicit_extract stage

### Persona-DB
- **Title**: Persona-DB: Efficient Large Language Model Personalization for Response Prediction
- **Authors**: Various
- **Venue**: ACL Findings 2024
- **Key Contribution**: Retrieval-based persona selection from database for response prediction
- **Relevance**: Retrieval-augmented persona approach; alternative to our hierarchical storage

### PersonaX
- **Title**: PersonaX: Persona-Driven User Behavior Clustering for Large-Scale Recommendation
- **Authors**: Various
- **Venue**: Preprint 2024
- **Key Contribution**: LLM-based user behavior clustering to reduce input tokens. Cluster → representative summary
- **Relevance**: **High priority for optimization** — clustering approach could reduce our implicit extraction 73K tokens to ~10K

### ProEx
- **Title**: ProEx: Expert-Level Preference Extraction from User Behavior
- **Authors**: Various
- **Venue**: Preprint 2024
- **Key Contribution**: Expert-level preference extraction from implicit behavior signals (clicks, dwell time)
- **Relevance**: Methodological reference for our implicit preference extraction stage

### RMM (Retrieval-augmented Memory Model)
- **Title**: RMM: Retrieval-augmented Memory for Personalized Dialogue
- **Authors**: Various
- **Venue**: Preprint 2024
- **Key Contribution**: Hybrid retrieval (dense + sparse) for memory-augmented dialogue
- **Relevance**: Retrieval strategy reference; our keyword-based BM25 retrieval could benefit from hybrid approach

### ZeroPOIRec
- **Title**: ZeroPOIRec: Zero-Shot POI Recommendation with LLM-Generated Personas
- **Authors**: Various
- **Venue**: Preprint 2024
- **Key Contribution**: LLM-generated user personas for zero-shot POI recommendation without interaction history
- **Relevance**: **Directly applicable** — POI recommendation with persona, validates our end-to-end pipeline direction

---

## Tier 4: Full-Context & Baselines

### Beyond the Context Window
- **Title**: Beyond the Context Window: Long-Context LLMs vs Memory-Augmented Agents
- **Authors**: Various
- **Venue**: Preprint 2025
- **Links**: [arXiv](https://arxiv.org/abs/2603.04814)
- **Key Contribution**: Most direct full-context vs memory comparison. Long-context GPT-5-mini achieves 92.85% on LoCoMo vs Mem0's 57.68%
- **Relevance**: Establishes full-context as quality upper bound for factual QA tasks. But PersonaMem-v2 shows structured memory wins for implicit personalization

---

## Tier 5: Retrieval & Reranking

### BM25 (Okapi BM25)
- **Title**: The Probabilistic Relevance Framework: BM25 and Beyond
- **Authors**: Stephen Robertson, Hugo Zaragoza
- **Venue**: Foundations and Trends in IR, 2009
- **Key Contribution**: Term-frequency based sparse retrieval — standard baseline
- **Relevance**: Our current keyword-based note retrieval uses BM25 paradigm

### DPR (Dense Passage Retrieval)
- **Title**: Dense Passage Retrieval for Open-Domain Question Answering
- **Authors**: Vladimir Karpukhin, et al.
- **Venue**: EMNLP 2020
- **Key Contribution**: Dual-encoder dense retrieval with BERT embeddings
- **Relevance**: Dense retrieval baseline; potential upgrade path for note retrieval

### Contriever
- **Title**: Unsupervised Dense Information Retrieval with Contrastive Learning
- **Authors**: Gautier Izacard, et al.
- **Venue**: TMLR 2022
- **Key Contribution**: Unsupervised contrastive learning for dense retrieval without labeled data
- **Relevance**: Zero-shot dense retrieval; useful if we lack retrieval training data

### E5
- **Title**: Text Embeddings by Weakly-Supervised Contrastive Pre-training
- **Authors**: Liang Wang, et al.
- **Venue**: ACL 2024
- **Key Contribution**: State-of-art text embeddings via instruction-tuned contrastive learning
- **Relevance**: Best-in-class embedding model for note/persona semantic search

### Hybrid Search (CC/RRF/SRRF)
- **Title**: Reciprocal Rank Fusion (RRF) and variants
- **Authors**: Gordon Cormack, et al.
- **Venue**: SIGIR 2009 (RRF)
- **Key Contribution**: Combining sparse (BM25) + dense retrieval via rank fusion
- **Relevance**: Hybrid retrieval strategy for combining keyword + semantic note search

### ReaRANK
- **Title**: ReaRANK: Reasoning-Guided Reranking for Long-Context Retrieval
- **Authors**: Various
- **Venue**: Preprint 2024
- **Key Contribution**: LLM reasoning-based reranking for improved retrieval precision
- **Relevance**: Advanced reranking for persona-relevant note selection
