import math

def precision_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    top = retrieved[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / k

def dcg_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            score += 1 / math.log2(i + 2)
    return score

def ndcg_at_k(retrieved: list[int], relevant: list[int], k: int) -> float:
    ideal_hits = min(len(relevant), k)
    ideal = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg_at_k(retrieved, relevant, k) / ideal