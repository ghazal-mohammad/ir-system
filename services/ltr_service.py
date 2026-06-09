import numpy as np
import pickle
from collections import defaultdict


def build_feature_matrix(query_ids: list, all_results: dict,
                          qrels: dict, n_features: int = 3) -> tuple:
    X, y, groups = [], [], []

    for qid in query_ids:
        if qid not in all_results:
            continue
        relevant = qrels.get(qid, {})
        results = all_results[qid]
        if not results:
            continue

        score_map = {r['doc_id']: r['score'] for r in results}
        max_score = max(score_map.values()) if score_map else 1.0

        for r in results:
            doc_id = r['doc_id']
            raw_score = r['score']
            norm_score = raw_score / max_score if max_score > 0 else 0.0
            rank = r['rank']

            features = [
                raw_score,
                norm_score,
                1.0 / rank,
            ]
            X.append(features)
            y.append(int(doc_id in relevant))

        groups.append(len(results))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), groups


def build_feature_matrix_multi(query_ids: list, results_per_model: dict,
                                 qrels: dict) -> tuple:
    X, y, groups = [], [], []

    model_names = list(results_per_model.keys())

    for qid in query_ids:
        relevant = qrels.get(qid, {})
        all_docs = set()
        for model_res in results_per_model.values():
            if qid in model_res:
                all_docs.update(r['doc_id'] for r in model_res[qid])

        if not all_docs:
            continue

        for doc_id in all_docs:
            features = []
            for model_name in model_names:
                model_res = results_per_model[model_name]
                score_map = {r['doc_id']: (r['score'], r['rank'])
                             for r in model_res.get(qid, [])}
                if doc_id in score_map:
                    score, rank = score_map[doc_id]
                    features.extend([score, 1.0 / rank])
                else:
                    features.extend([0.0, 0.0])

            X.append(features)
            y.append(min(relevant.get(doc_id, 0), 1))

        groups.append(len(all_docs))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), groups


def train_ltr_model(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        ))
    ])
    clf.fit(X, y)
    return clf


def rerank_with_ltr(query_id: str, results_per_model: dict,
                     ltr_model, model_names: list,
                     top_k: int = 1000) -> list:
    all_docs = set()
    for model_name in model_names:
        model_res = results_per_model[model_name]
        if query_id in model_res:
            all_docs.update(r['doc_id'] for r in model_res[query_id])

    doc_list = list(all_docs)
    if not doc_list:
        return []

    X = []
    for doc_id in doc_list:
        features = []
        for model_name in model_names:
            model_res = results_per_model[model_name]
            score_map = {r['doc_id']: (r['score'], r['rank'])
                         for r in model_res.get(query_id, [])}
            if doc_id in score_map:
                score, rank = score_map[doc_id]
                features.extend([score, 1.0 / rank])
            else:
                features.extend([0.0, 0.0])
        X.append(features)

    X_arr = np.array(X, dtype=np.float32)
    proba = ltr_model.predict_proba(X_arr)[:, 1]

    order = np.argsort(proba)[::-1]
    results = []
    for rank, idx in enumerate(order[:top_k], start=1):
        results.append({
            'doc_id': doc_list[idx],
            'score': round(float(proba[idx]), 6),
            'rank': rank,
        })
    return results


def evaluate_ltr(X: np.ndarray, y: np.ndarray, ltr_model) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    y_pred = ltr_model.predict(X)
    y_proba = ltr_model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_proba)
    except Exception:
        auc = 0.0

    return {'accuracy': round(acc, 4), 'auc': round(auc, 4)}


def save_ltr_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_ltr_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
