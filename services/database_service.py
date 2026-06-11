import sqlite3
import os


# ── SQLite storage for raw documents ─────────────────────────────────────────
# Raw document text is stored in a database. At query time the original
# document is read from the DB by its doc_id (only top results are fetched,
# so retrieval stays fast and we never keep millions of raw texts in RAM).


def get_db_path(data_dir: str, dataset_name: str) -> str:
    return os.path.join(data_dir, f'{dataset_name}_docs.db')


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # speed-oriented pragmas (safe for a read-mostly store)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    return conn


def create_docs_table(conn: sqlite3.Connection):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            text   TEXT NOT NULL
        )
    ''')
    conn.commit()


def insert_documents(conn: sqlite3.Connection, docs_iter, batch_size: int = 50000):
    """
    Bulk-insert raw documents.
    docs_iter: iterable of (doc_id, text) tuples.
    Returns total number of inserted rows.
    """
    create_docs_table(conn)
    cur = conn.cursor()
    batch = []
    total = 0

    for doc_id, text in docs_iter:
        batch.append((str(doc_id), text))
        if len(batch) >= batch_size:
            cur.executemany(
                'INSERT OR REPLACE INTO documents (doc_id, text) VALUES (?, ?)',
                batch)
            conn.commit()
            total += len(batch)
            print(f'  inserted {total:,} docs...', end='\r')
            batch = []

    if batch:
        cur.executemany(
            'INSERT OR REPLACE INTO documents (doc_id, text) VALUES (?, ?)',
            batch)
        conn.commit()
        total += len(batch)

    print(f'Inserted {total:,} documents into DB')
    return total


def get_document(conn: sqlite3.Connection, doc_id: str) -> str:
    """Read one raw document from the DB by its id."""
    row = conn.execute(
        'SELECT text FROM documents WHERE doc_id = ?', (str(doc_id),)
    ).fetchone()
    return row[0] if row else ''


def get_documents(conn: sqlite3.Connection, doc_ids: list) -> dict:
    """
    Read multiple raw documents by id in one query.
    Used at query time to fetch the top-k results from the DB.
    Returns: { doc_id: text }
    """
    if not doc_ids:
        return {}
    ids = [str(d) for d in doc_ids]
    placeholders = ','.join('?' * len(ids))
    rows = conn.execute(
        f'SELECT doc_id, text FROM documents WHERE doc_id IN ({placeholders})',
        ids).fetchall()
    return {doc_id: text for doc_id, text in rows}


def count_documents(conn: sqlite3.Connection) -> int:
    row = conn.execute('SELECT COUNT(*) FROM documents').fetchone()
    return row[0] if row else 0


def db_exists_and_filled(db_path: str) -> bool:
    """Check the DB file exists and actually contains documents."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = connect(db_path)
        n = count_documents(conn)
        conn.close()
        return n > 0
    except sqlite3.Error:
        return False
