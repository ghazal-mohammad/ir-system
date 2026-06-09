import json
import pickle
import os
from collections import defaultdict
from tqdm import tqdm


def build_inverted_index(processed_docs: dict) -> dict:
    """
    Build an inverted index from preprocessed documents.
    Structure: { term: { doc_id: term_freq, ... } }
    """
    index = defaultdict(dict)

    for doc_id, text in tqdm(processed_docs.items(), desc='Building index'):
        tokens = text.split()
        # count term frequency in this doc
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1
        # add to index
        for term, freq in term_freq.items():
            index[term][doc_id] = freq

    return dict(index)


def get_doc_lengths(processed_docs: dict) -> dict:
    """Returns number of tokens per document (needed for BM25)."""
    return {doc_id: len(text.split()) for doc_id, text in processed_docs.items()}


def get_avg_doc_length(doc_lengths: dict) -> float:
    if not doc_lengths:
        return 0.0
    return sum(doc_lengths.values()) / len(doc_lengths)


def save_index(index: dict, path: str):
    with open(path, 'wb') as f:
        pickle.dump(index, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f'Index saved: {path} ({size_mb:.1f} MB)')


def load_index(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_index_stats(index: dict, doc_lengths: dict) -> dict:
    return {
        'vocab_size': len(index),
        'num_docs': len(doc_lengths),
        'avg_doc_length': round(get_avg_doc_length(doc_lengths), 2),
        'total_tokens': sum(doc_lengths.values()),
    }
