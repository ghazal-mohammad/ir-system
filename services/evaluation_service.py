import math
from collections import defaultdict


def load_qrels(dataset) -> dict:
    """
    Load qrels from an ir_datasets dataset object.
    Returns: { query_id: { doc_id: relevance_grade } }
    Keeps only relevance >= 1.
    """
    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        if qrel.relevance >= 1:
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
    return dict(qrels)


def precision_at_k(results: list, relevant: dict, k: int = 10) -> float:
    top_k = [r['doc_id'] for r in results[:k]]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(results: list, relevant: dict, k: int = 1000) -> float:
    if not relevant:
        return 0.0
    top_k = {r['doc_id'] for r in results[:k]}
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def average_precision(results: list, relevant: dict) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total_precision = 0.0
    for i, item in enumerate(results, start=1):
        if item['doc_id'] in relevant:
            hits += 1
            total_precision += hits / i
    return total_precision / len(relevant)


def ndcg_at_k(results: list, relevant: dict, k: int = 10) -> float:
    def dcg(ranked_docs, k):
        score = 0.0
        for i, doc_id in enumerate(ranked_docs[:k], start=1):
            grade = relevant.get(doc_id, 0)
            score += (2 ** grade - 1) / math.log2(i + 1)
        return score

    ranked = [r['doc_id'] for r in results[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    ideal_docs = list(relevant.keys())[:k]

    actual_dcg = dcg(ranked, k)
    ideal_dcg = sum((2 ** grade - 1) / math.log2(i + 2)
                    for i, grade in enumerate(ideal))

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_run(all_results: dict, qrels: dict,
                 k_p: int = 10, k_ndcg: int = 10) -> dict:
    """
    Evaluate a full retrieval run over all queries.
    Returns per-query metrics and macro-averages.
    """
    per_query = {}
    map_scores = []
    recall_scores = []
    p_at_k_scores = []
    ndcg_scores = []

    for qid, results in all_results.items():
        relevant = qrels.get(qid, {})
        if not relevant:
            continue

        ap = average_precision(results, relevant)
        rec = recall_at_k(results, relevant)
        p_k = precision_at_k(results, relevant, k=k_p)
        nd = ndcg_at_k(results, relevant, k=k_ndcg)

        per_query[qid] = {
            'AP': round(ap, 4),
            'Recall': round(rec, 4),
            f'P@{k_p}': round(p_k, 4),
            f'nDCG@{k_ndcg}': round(nd, 4),
        }
        map_scores.append(ap)
        recall_scores.append(rec)
        p_at_k_scores.append(p_k)
        ndcg_scores.append(nd)

    n = len(map_scores)
    aggregated = {
        'num_queries': n,
        'MAP': round(sum(map_scores) / n, 4) if n else 0.0,
        'Recall': round(sum(recall_scores) / n, 4) if n else 0.0,
        f'P@{k_p}': round(sum(p_at_k_scores) / n, 4) if n else 0.0,
        f'nDCG@{k_ndcg}': round(sum(ndcg_scores) / n, 4) if n else 0.0,
    }
    return {'per_query': per_query, 'aggregated': aggregated}


# ── Charts (used in the evaluation notebook and the report) ─────────────────

def plot_model_comparison(model_scores: dict, title: str = 'Model Comparison',
                          save_path: str = None):
    """
    Grouped bar chart of evaluation metrics per model.
    model_scores: { model_name: aggregated_metrics_dict }
    """
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(model_scores.keys())
    metrics = [k for k in model_scores[models[0]] if k != 'num_queries']

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        values = [model_scores[model].get(m, 0.0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model)
        ax.bar_label(bars, fmt='%.3f', fontsize=7)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Chart saved: {save_path}')
        plt.close(fig)
    else:
        plt.show()


def plot_before_after(before: dict, after: dict, feature_name: str,
                      save_path: str = None):
    """
    Bar chart comparing metrics before vs after enabling an additional feature.
    before/after: aggregated metrics dicts from evaluate_run.
    """
    plot_model_comparison(
        {'Before (baseline)': before, f'After ({feature_name})': after},
        title=f'Evaluation Before vs After — {feature_name}',
        save_path=save_path)


def print_results_table(model_scores: dict):
    """
    model_scores: { model_name: aggregated_metrics_dict }
    Prints a formatted comparison table.
    """
    models = list(model_scores.keys())
    if not models:
        return

    metrics = [k for k in model_scores[models[0]] if k != 'num_queries']
    col_w = 14

    header = f'{"Model":<20}' + ''.join(f'{m:>{col_w}}' for m in metrics)
    print(header)
    print('-' * len(header))
    for model_name, scores in model_scores.items():
        row = f'{model_name:<20}' + ''.join(f'{scores.get(m, 0.0):>{col_w}.4f}' for m in metrics)
        print(row)
