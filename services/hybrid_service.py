from collections import defaultdict


# ── Reciprocal Rank Fusion ──────────────────────────────────────────────────

def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(ranked_lists: list, top_k: int = 1000) -> list:
    """
    Merge multiple ranked lists using RRF.
    ranked_lists: list of lists, each inner list has dicts with 'doc_id' and 'rank'.
    """
    fusion_scores = defaultdict(float)

    for ranked in ranked_lists:
        for item in ranked:
            fusion_scores[item['doc_id']] += rrf_score(item['rank'])

    sorted_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], start=1):
        results.append({'doc_id': doc_id, 'score': round(score, 6), 'rank': rank})
    return results


# ── Hybrid Parallel (BM25 + TF-IDF + Embedding via RRF) ────────────────────

def hybrid_parallel(bm25_results: list, tfidf_results: list,
                    embedding_results: list, top_k: int = 1000) -> list:
    return reciprocal_rank_fusion([bm25_results, tfidf_results, embedding_results], top_k)


def hybrid_parallel_two(bm25_results: list, embedding_results: list,
                         top_k: int = 1000) -> list:
    return reciprocal_rank_fusion([bm25_results, embedding_results], top_k)


# ── Hybrid Serial (BM25 first-stage → Embedding re-rank) ───────────────────

def hybrid_serial(query_text: str, bm25_results: list, model,
                  doc_ids: list, doc_embeddings,
                  first_stage_k: int = 100, top_k: int = 1000) -> list:
    """
    Stage 1: take top first_stage_k docs from BM25.
    Stage 2: re-rank them using embedding similarity.
    """
    import numpy as np

    candidates = [r['doc_id'] for r in bm25_results[:first_stage_k]]

    q_emb = model.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)

    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    candidate_indices = [doc_id_to_idx[did] for did in candidates if did in doc_id_to_idx]
    candidate_doc_ids = [doc_ids[i] for i in candidate_indices]
    candidate_embs = doc_embeddings[candidate_indices]

    scores = (candidate_embs @ q_emb.T).squeeze()
    if len(scores.shape) == 0:
        scores = scores.reshape(1)

    order = np.argsort(scores)[::-1]
    results = []
    for rank, idx in enumerate(order[:top_k], start=1):
        results.append({
            'doc_id': candidate_doc_ids[idx],
            'score': round(float(scores[idx]), 6),
            'rank': rank
        })
    return results


# ── Score Normalization (helper for weighted fusion) ───────────────────────

def min_max_normalize(results: list) -> list:
    if not results:
        return results
    scores = [r['score'] for r in results]
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [{**r, 'score': 1.0} for r in results]
    return [{**r, 'score': round((r['score'] - mn) / (mx - mn), 6)} for r in results]


def weighted_fusion(results_list: list, weights: list, top_k: int = 1000) -> list:
    """
    Weighted sum of normalized scores from multiple ranked lists.
    weights: list of floats, one per result list, must sum to 1.
    """
    normalized = [min_max_normalize(r) for r in results_list]
    combined = defaultdict(float)
    for res_list, w in zip(normalized, weights):
        for item in res_list:
            combined[item['doc_id']] += w * item['score']
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [{'doc_id': did, 'score': round(score, 6), 'rank': i + 1}
            for i, (did, score) in enumerate(ranked[:top_k])]
