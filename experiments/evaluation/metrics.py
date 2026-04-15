import math


def precision_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    """P@k = |Predicted_k ∩ GT| / k"""
    if k <= 0:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / k


def recall_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    """R@k = |Predicted_k ∩ GT| / |GT|"""
    if not ground_truth:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return hits / len(ground_truth)


def ndcg_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    """DCG@k = sum(rel_i / log2(i+1)), NDCG = DCG/IDCG where rel_i=1 if in GT"""
    if k <= 0 or not ground_truth:
        return 0.0
    top_k = predicted[:k]

    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-based: log2(rank+1)

    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(predicted: list[str], ground_truth: set[str], k: int) -> float:
    """HR@k = 1 if any predicted_k in GT, else 0"""
    if k <= 0:
        return 0.0
    top_k = predicted[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def mrr(predicted: list[str], ground_truth: set[str]) -> float:
    """MRR = 1/rank of first hit. 0 if no hit."""
    for i, item in enumerate(predicted):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def compute_all_metrics(predicted: list[str], ground_truth: set[str]) -> dict[str, float]:
    """Compute all metrics. Returns dict with keys:
    precision@1, precision@3, precision@5, recall@5, ndcg@10, hr@10, mrr"""
    return {
        "precision@1": precision_at_k(predicted, ground_truth, 1),
        "precision@3": precision_at_k(predicted, ground_truth, 3),
        "precision@5": precision_at_k(predicted, ground_truth, 5),
        "recall@5": recall_at_k(predicted, ground_truth, 5),
        "ndcg@10": ndcg_at_k(predicted, ground_truth, 10),
        "hr@10": hit_rate_at_k(predicted, ground_truth, 10),
        "mrr": mrr(predicted, ground_truth),
    }
