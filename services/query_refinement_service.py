import re
from collections import Counter
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


# ── Spell Correction ─────────────────────────────────────────────────────────

def build_vocab_from_index(inverted_index: dict) -> set:
    return set(inverted_index.keys())


def edit_distance(w1: str, w2: str) -> int:
    m, n = len(w1), len(w2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if w1[i - 1] == w2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def correct_token(token: str, vocab: set, max_dist: int = 2) -> str:
    if token in vocab:
        return token
    # filter by length proximity first (speed up)
    candidates = [w for w in vocab if abs(len(w) - len(token)) <= max_dist]
    best, best_dist = token, max_dist + 1
    for w in candidates:
        d = edit_distance(token, w)
        if d < best_dist:
            best, best_dist = w, d
    return best


def spell_correct_query(query_tokens: list, vocab: set) -> tuple:
    corrected = []
    changes = []
    for t in query_tokens:
        fixed = correct_token(t, vocab)
        corrected.append(fixed)
        if fixed != t:
            changes.append((t, fixed))
    return corrected, changes


# ── Synonym Expansion ────────────────────────────────────────────────────────

def get_synonyms(word: str, max_synonyms: int = 3) -> list:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ').lower()
            if name != word and len(name.split()) == 1:
                synonyms.add(name)
            if len(synonyms) >= max_synonyms:
                break
        if len(synonyms) >= max_synonyms:
            break
    return list(synonyms)


def expand_query(tokens: list, vocab: set, max_per_token: int = 2) -> list:
    expanded = list(tokens)
    for token in tokens:
        syns = get_synonyms(token, max_synonyms=max_per_token)
        for s in syns:
            if s in vocab and s not in expanded:
                expanded.append(s)
    return expanded


# ── Query Suggestions ────────────────────────────────────────────────────────

def suggest_queries(partial_query: str, query_history: list, top_k: int = 5) -> list:
    partial = partial_query.strip().lower()
    if not partial:
        return []
    matches = []
    for q in query_history:
        if q.lower().startswith(partial) and q.lower() != partial:
            matches.append(q)
    return list(dict.fromkeys(matches))[:top_k]


def suggest_related_terms(tokens: list, inverted_index: dict,
                           doc_lengths: dict, top_k: int = 5) -> list:
    co_occur = Counter()
    for token in tokens:
        if token not in inverted_index:
            continue
        candidate_docs = list(inverted_index[token].keys())[:200]
        for doc_id in candidate_docs:
            for term, posting in inverted_index.items():
                if doc_id in posting and term not in tokens:
                    co_occur[term] += posting[doc_id]
    return [term for term, _ in co_occur.most_common(top_k)]


# ── Full Refinement Pipeline ─────────────────────────────────────────────────

def refine_query(tokens: list, vocab: set, inverted_index: dict,
                 use_spell: bool = True, use_expand: bool = True) -> dict:
    result = {
        'original_tokens': tokens,
        'corrections': [],
        'expanded_tokens': tokens,
        'added_synonyms': [],
    }

    current = list(tokens)

    if use_spell:
        current, changes = spell_correct_query(current, vocab)
        result['corrections'] = changes
        result['corrected_tokens'] = current

    if use_expand:
        expanded = expand_query(current, vocab)
        added = [t for t in expanded if t not in current]
        result['expanded_tokens'] = expanded
        result['added_synonyms'] = added

    return result
