import re


MODEL_NAME = 'google/flan-t5-base'
_pipeline = None


def load_rag_model(max_new_tokens: int = 128):
    # 128 tokens keeps CPU generation well under the 20s response budget
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            'text2text-generation',
            model=MODEL_NAME,
            max_new_tokens=max_new_tokens,
            device=-1
        )
    return _pipeline


def build_context(query: str, retrieved_docs: list,
                  raw_docs: dict, max_docs: int = 5,
                  max_chars_per_doc: int = 400) -> str:
    context_parts = []
    for i, result in enumerate(retrieved_docs[:max_docs]):
        doc_id = result['doc_id']
        text = raw_docs.get(doc_id, '')
        if text:
            snippet = text[:max_chars_per_doc].strip()
            context_parts.append(f'[Document {i + 1}]: {snippet}')
    return '\n\n'.join(context_parts)


def build_prompt(query: str, context: str) -> str:
    return (
        f'Answer the following question based on the provided documents.\n\n'
        f'Question: {query}\n\n'
        f'Documents:\n{context}\n\n'
        f'Answer:'
    )


def generate_answer(query: str, retrieved_docs: list,
                    raw_docs: dict, pipeline=None,
                    max_docs: int = 5) -> dict:
    if pipeline is None:
        pipeline = load_rag_model()

    context = build_context(query, retrieved_docs, raw_docs, max_docs=max_docs)
    if not context:
        return {
            'answer': 'No relevant documents found to generate an answer.',
            'context': '',
            'sources': [],
        }

    prompt = build_prompt(query, context)
    output = pipeline(prompt)
    answer = output[0]['generated_text'].strip()

    sources = [r['doc_id'] for r in retrieved_docs[:max_docs]]
    return {
        'answer': answer,
        'context': context,
        'sources': sources,
    }


def rag_retrieve_and_generate(query: str, query_tokens: list,
                               retrieval_fn, retrieval_args: dict,
                               raw_docs: dict, pipeline=None,
                               top_k_retrieve: int = 10) -> dict:
    retrieved = retrieval_fn(query_tokens, **retrieval_args, top_k=top_k_retrieve)
    rag_result = generate_answer(query, retrieved, raw_docs, pipeline=pipeline)
    rag_result['retrieved_docs'] = retrieved
    return rag_result
