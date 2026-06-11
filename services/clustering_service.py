import numpy as np
import pickle
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict


def find_optimal_k(embeddings: np.ndarray, k_range: range = range(5, 21),
                   sample_size: int = 10000) -> dict:
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample = embeddings[indices]
    else:
        sample = embeddings

    inertias = {}
    sil_scores = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(sample)
        inertias[k] = km.inertia_
        if k > 1:
            sil_scores[k] = silhouette_score(sample, labels, sample_size=min(2000, len(sample)))

    best_k = max(sil_scores, key=sil_scores.get) if sil_scores else k_range[0]
    return {'inertias': inertias, 'silhouette_scores': sil_scores, 'best_k': best_k}


def cluster_documents(embeddings: np.ndarray, n_clusters: int,
                       batch_size: int = 10000) -> np.ndarray:
    if len(embeddings) > 100000:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                              batch_size=batch_size, n_init=5)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    labels = km.fit_predict(embeddings)
    return labels, km


def get_cluster_top_terms(cluster_doc_ids: list, inverted_index: dict,
                           top_k: int = 10) -> list:
    term_freq = defaultdict(int)
    doc_set = set(cluster_doc_ids)
    for term, posting in inverted_index.items():
        for doc_id in posting:
            if doc_id in doc_set:
                term_freq[term] += posting[doc_id]
    return sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]


def get_cluster_summary(doc_ids: list, labels: np.ndarray,
                         inverted_index: dict, n_clusters: int) -> dict:
    cluster_docs = defaultdict(list)
    for doc_id, label in zip(doc_ids, labels):
        cluster_docs[int(label)].append(doc_id)

    summary = {}
    for cid in range(n_clusters):
        docs = cluster_docs[cid]
        top_terms = get_cluster_top_terms(docs, inverted_index, top_k=8)
        summary[cid] = {
            'size': len(docs),
            'top_terms': [t for t, _ in top_terms],
            'doc_ids': docs[:5],
        }
    return summary


def assign_query_to_cluster(query_embedding: np.ndarray, km_model) -> int:
    return int(km_model.predict(query_embedding.reshape(1, -1))[0])


# ── Charts (clustering evaluation for the report) ───────────────────────────

def plot_elbow_and_silhouette(opt_result: dict, save_path: str = None):
    """Plot elbow (inertia) and silhouette score curves from find_optimal_k."""
    import matplotlib.pyplot as plt

    inertias = opt_result['inertias']
    sils = opt_result['silhouette_scores']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(inertias.keys()), list(inertias.values()), 'o-')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(alpha=0.3)

    ax2.plot(list(sils.keys()), list(sils.values()), 'o-', color='green')
    ax2.axvline(opt_result['best_k'], color='red', linestyle='--',
                label=f"best K = {opt_result['best_k']}")
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Chart saved: {save_path}')
        plt.close(fig)
    else:
        plt.show()


def plot_cluster_scatter(embeddings: np.ndarray, labels: np.ndarray,
                         sample_size: int = 5000, save_path: str = None):
    """2D PCA projection of the clusters."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        emb_sample, lbl_sample = embeddings[idx], labels[idx]
    else:
        emb_sample, lbl_sample = embeddings, labels

    points = PCA(n_components=2, random_state=42).fit_transform(emb_sample)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=lbl_sample,
                          cmap='tab10', s=4, alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Document Clusters (PCA 2D projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Chart saved: {save_path}')
        plt.close()
    else:
        plt.show()


def plot_cluster_sizes(labels: np.ndarray, save_path: str = None):
    """Bar chart of how many documents fall in each cluster."""
    import matplotlib.pyplot as plt
    from collections import Counter

    counts = Counter(labels.tolist())
    cids = sorted(counts)
    plt.figure(figsize=(8, 4))
    plt.bar([str(c) for c in cids], [counts[c] for c in cids])
    plt.xlabel('Cluster')
    plt.ylabel('Documents')
    plt.title('Cluster Sizes')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Chart saved: {save_path}')
        plt.close()
    else:
        plt.show()


def save_clustering(labels: np.ndarray, km_model, path_prefix: str):
    np.save(f'{path_prefix}_cluster_labels.npy', labels)
    with open(f'{path_prefix}_km_model.pkl', 'wb') as f:
        pickle.dump(km_model, f)
    print(f'Clustering saved: {len(labels):,} docs, {km_model.n_clusters} clusters')


def load_clustering(path_prefix: str):
    labels = np.load(f'{path_prefix}_cluster_labels.npy')
    with open(f'{path_prefix}_km_model.pkl', 'rb') as f:
        km_model = pickle.load(f)
    return labels, km_model
