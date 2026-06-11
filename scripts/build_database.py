"""
Build the raw-documents SQLite database for one (or both) datasets.

Raw text is stored in a DB so that at query time the original document
is read from the database by doc_id (graded requirement).

Usage:
    python scripts/build_database.py --dataset ct2021
    python scripts/build_database.py --dataset msmarco
    python scripts/build_database.py --dataset all
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import database_service as db

DATA_DIR = os.environ.get('IR_DATA_DIR', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

DATASETS = {
    'ct2021': 'clinicaltrials/2021/trec-ct-2021',
    'msmarco': 'msmarco-passage/trec-dl-2019',
}


def ct2021_raw_text(doc) -> str:
    # same field combination used during preprocessing
    return ' '.join(filter(None, [
        doc.title or '',
        str(doc.condition) if doc.condition else '',
        str(doc.intervention) if doc.intervention else '',
        str(doc.summary) if doc.summary else '',
    ]))


def docs_iterator(dataset_key: str):
    import ir_datasets
    dataset = ir_datasets.load(DATASETS[dataset_key])
    if dataset_key == 'ct2021':
        for doc in dataset.docs_iter():
            yield doc.doc_id, ct2021_raw_text(doc)
    else:
        for doc in dataset.docs_iter():
            yield doc.doc_id, doc.text


def build(dataset_key: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    db_path = db.get_db_path(DATA_DIR, dataset_key)
    print(f'Building DB for {dataset_key} -> {db_path}')

    t0 = time.time()
    conn = db.connect(db_path)
    db.insert_documents(conn, docs_iterator(dataset_key))
    n = db.count_documents(conn)
    conn.close()
    print(f'{dataset_key}: {n:,} docs stored in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ct2021', 'msmarco', 'all'],
                        default='all')
    args = parser.parse_args()

    keys = list(DATASETS) if args.dataset == 'all' else [args.dataset]
    for key in keys:
        build(key)
    print('Done.')
