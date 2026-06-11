import math
import pickle
import os
from collections import defaultdict


# BM25 hyperparameters
K1 = 1.5
B = 0.75


def bm25_score(tf: int, df: int, num_docs: int, doc_len: int,
               avg_dl: float, k1: float = K1, b: float = B) -> float:
    idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_dl)))
    return idf * tf_norm


def retrieve_bm25(query_tokens: list, inverted_index: dict,
                  doc_lengths: dict, avg_dl: float,
                  top_k: int = 1000, k1: float = K1, b: float = B) -> list:
    """k1 and b are tunable per query (exposed in the UI)."""
    num_docs = len(doc_lengths)
    scores = defaultdict(float)

    for term in query_tokens:
        if term not in inverted_index:
            continue
        doc_dict = inverted_index[term]
        df = len(doc_dict)
        for doc_id, tf in doc_dict.items():
            dl = doc_lengths.get(doc_id, 1)
            scores[doc_id] += bm25_score(tf, df, num_docs, dl, avg_dl, k1=k1, b=b)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
        results.append({'doc_id': doc_id, 'score': round(score, 6), 'rank': rank})
    return results


def save_bm25_params(avg_dl: float, path: str):
    with open(path, 'wb') as f:
        pickle.dump({'avg_dl': avg_dl, 'k1': K1, 'b': B}, f)


def load_bm25_params(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)
