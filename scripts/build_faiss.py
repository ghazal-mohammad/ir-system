"""
Build a FAISS index from saved embeddings (offline step, run once).
Makes embedding search fast on large collections without loading
the full matrix in RAM at query time.

Usage:
    python scripts/build_faiss.py --dataset msmarco
    python scripts/build_faiss.py --dataset ct2021
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import (
    load_embeddings, build_faiss_index, save_faiss_index)

DATA_DIR = os.environ.get('IR_DATA_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ct2021', 'msmarco'], required=True)
    parser.add_argument('--nlist', type=int, default=256)
    args = parser.parse_args()

    prefix = f'{DATA_DIR}/{args.dataset}'
    print(f'Loading embeddings: {prefix}_embeddings.npy')
    doc_ids, embeddings = load_embeddings(prefix, use_memmap=True)
    print(f'Shape: {embeddings.shape}')

    t0 = time.time()
    index = build_faiss_index(embeddings, nlist=args.nlist)
    save_faiss_index(index, prefix)
    print(f'Done in {time.time() - t0:.1f}s')
