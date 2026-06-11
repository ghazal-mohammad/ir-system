"""
Evaluate the system before and after each additional feature,
and save comparison charts (MAP / nDCG focus).

Baseline = BM25. Each feature is evaluated independently:
  - Query refinement (spell correction + synonym expansion)
  - Clustering filter
  - LTR re-ranking

Usage:
    python scripts/evaluate_before_after.py --dataset ct2021
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.indexing_service import load_index
from services.bm25_service import retrieve_bm25, load_bm25_params
from services.tfidf_service import load_tfidf_matrix, retrieve_tfidf_fast
from services.embedding_service import load_model, load_embeddings, retrieve_embedding
from services.query_refinement_service import build_vocab_from_index, refine_query
from services.clustering_service import load_clustering, assign_query_to_cluster
from services.ltr_service import load_ltr_model, rerank_with_ltr
from services.evaluation_service import (
    evaluate_run, print_results_table, plot_before_after, plot_model_comparison)

DATA_DIR = os.environ.get('IR_DATA_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

LTR_MODEL_NAMES = ['bm25', 'tfidf', 'embedding']  # must match training order


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main(dataset):
    p = f'{DATA_DIR}/{dataset}'
    charts_dir = f'{DATA_DIR}/charts'
    os.makedirs(charts_dir, exist_ok=True)

    print('Loading models and data...')
    index = load_index(f'{p}_index.pkl')
    doc_lengths = load_json(f'{p}_doc_lengths.json')
    avg_dl = load_bm25_params(f'{p}_bm25_params.pkl')['avg_dl']
    vocab = build_vocab_from_index(index)
    queries = load_json(f'{p}_queries_processed.json')
    qrels = {qid: {d: r for d, r in docs.items() if r >= 1}
             for qid, docs in load_json(f'{p}_qrels.json').items()}

    vectorizer, doc_matrix, tfidf_doc_ids = load_tfidf_matrix(p)
    emb_model = load_model()
    emb_doc_ids, embeddings = load_embeddings(p, use_memmap=(dataset == 'msmarco'))

    # ── Baseline: BM25 ──────────────────────────────────────────────────────
    print('Running baseline (BM25)...')
    baseline_runs = {}
    for qid, q_text in queries.items():
        tokens = q_text.split()
        baseline_runs[qid] = retrieve_bm25(tokens, index, doc_lengths, avg_dl, top_k=1000)
    baseline = evaluate_run(baseline_runs, qrels)['aggregated']

    comparisons = {'Baseline (BM25)': baseline}

    # ── Feature 1: Query Refinement ─────────────────────────────────────────
    print('Evaluating: query refinement...')
    refined_runs = {}
    for qid, q_text in queries.items():
        tokens = q_text.split()
        refined = refine_query(tokens, vocab, index, use_spell=True, use_expand=True)
        refined_runs[qid] = retrieve_bm25(refined['expanded_tokens'], index,
                                          doc_lengths, avg_dl, top_k=1000)
    after_refinement = evaluate_run(refined_runs, qrels)['aggregated']
    comparisons['+ Refinement'] = after_refinement
    plot_before_after(baseline, after_refinement, 'Query Refinement',
                      save_path=f'{charts_dir}/{dataset}_before_after_refinement.png')

    # ── Feature 2: Clustering filter ────────────────────────────────────────
    print('Evaluating: clustering filter...')
    labels, km_model = load_clustering(p)
    doc_to_cluster = {did: int(lbl) for did, lbl in zip(emb_doc_ids, labels)}
    cluster_runs = {}
    for qid, q_text in queries.items():
        results = baseline_runs[qid]
        q_emb = emb_model.encode([q_text], normalize_embeddings=True,
                                 convert_to_numpy=True)[0]
        q_cluster = assign_query_to_cluster(q_emb, km_model)
        filtered = [r for r in results if doc_to_cluster.get(r['doc_id']) == q_cluster]
        cluster_runs[qid] = ([{**r, 'rank': i + 1} for i, r in enumerate(filtered)]
                             if filtered else results)
    after_clustering = evaluate_run(cluster_runs, qrels)['aggregated']
    comparisons['+ Clustering'] = after_clustering
    plot_before_after(baseline, after_clustering, 'Clustering Filter',
                      save_path=f'{charts_dir}/{dataset}_before_after_clustering.png')

    # ── Feature 3: LTR re-ranking ───────────────────────────────────────────
    print('Evaluating: LTR re-ranking...')
    ltr_model = load_ltr_model(f'{p}_ltr_model.pkl')
    ltr_runs = {}
    for qid, q_text in queries.items():
        tokens = q_text.split()
        results_per_model = {
            'bm25': {qid: baseline_runs[qid][:100]},
            'tfidf': {qid: retrieve_tfidf_fast(q_text, vectorizer, doc_matrix,
                                               tfidf_doc_ids, top_k=100)},
            'embedding': {qid: retrieve_embedding(q_text, emb_model, emb_doc_ids,
                                                  embeddings, top_k=100)},
        }
        ltr_runs[qid] = rerank_with_ltr(qid, results_per_model, ltr_model,
                                        LTR_MODEL_NAMES, top_k=1000)
    after_ltr = evaluate_run(ltr_runs, qrels)['aggregated']
    comparisons['+ LTR'] = after_ltr
    plot_before_after(baseline, after_ltr, 'LTR Re-ranking',
                      save_path=f'{charts_dir}/{dataset}_before_after_ltr.png')

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f'\n=== Before/After Summary — {dataset} ===')
    print_results_table(comparisons)
    plot_model_comparison(comparisons,
                          title=f'Before vs After Additional Features — {dataset}',
                          save_path=f'{charts_dir}/{dataset}_before_after_all.png')

    with open(f'{p}_before_after_eval.json', 'w') as f:
        json.dump(comparisons, f, indent=2)
    print(f'Saved: {p}_before_after_eval.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ct2021', 'msmarco'], default='ct2021')
    args = parser.parse_args()
    main(args.dataset)
