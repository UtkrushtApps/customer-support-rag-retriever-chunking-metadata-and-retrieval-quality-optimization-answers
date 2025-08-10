# rag_retrieval.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

def embed_query(query: str, model) -> List[float]:
    vec = model.encode([query], normalize_embeddings=True)
    return vec[0].tolist()

def retrieve(
        query: str, 
        chroma_client, 
        collection_name: str,  
        model, 
        k: int = 5,
        filters: Optional[dict] = None
    ) -> List[Dict]:
    collection = chroma_client.get_collection(collection_name)
    qvec = embed_query(query, model)
    results = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        where=filters if filters else None, # metadata filter
        include=["documents", "distances", "metadatas"]
    )
    output = []
    for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
        out = {
            'document': doc,
            'distance': dist,
            'metadata': meta
        }
        output.append(out)
    return output

def print_results(results: List[Dict]):
    for rank, item in enumerate(results, 1):
        print(f"Rank {rank}: dist={item['distance']:.4f}")
        print(f"Category: {item['metadata'].get('category')} | Priority: {item['metadata'].get('priority')} | Date: {item['metadata'].get('date')}")
        print(item['document'][:320] + ("..." if len(item['document']) > 320 else ""))
        print('---')

def rag_assemble(query: str, rag_results: List[Dict]) -> str:
    """Basic assembly by concatenating text chunks as RAG context."""
    context = '\n\n'.join([res['document'] for res in rag_results])
    return f"Question: {query}\n\nRelevant Support Context:\n{context}"

def recall_at_k(queries_gt: List[Dict], chroma_client, collection_name: str, model, k: int = 5) -> float:
    """
    queries_gt: [ { 'query': str, 'relevant_chunk_ids': [str, ...] }, ... ]
    """
    n_found = 0
    for row in queries_gt:
        true_ids = set(row['relevant_chunk_ids'])
        result = retrieve(row['query'], chroma_client, collection_name, model, k)
        returned_ids = {r['metadata']['chunk_id'] for r in result}
        if true_ids & returned_ids:
            n_found += 1
    return n_found / len(queries_gt)

def spotcheck_examples(queries: List[str], chroma_client, collection_name, model):
    for q in queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, chroma_client, collection_name, model)
        print_results(results)

if __name__ == "__main__":
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="chromadb.db.impl.sqlite",
        persist_directory="./chroma_data"
    ))
    collection_name = 'customersupport_rag_chunks'
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Spot-check a few queries
    test_queries = [
        "How do I reset my password?",
        "What are the refund policies?",
        "Priority escalation for critical issues?",
    ]
    spotcheck_examples(test_queries, chroma_client, collection_name, model)

    # Recall@5 eval for a small test set
    # e.g. [{ 'query': ..., 'relevant_chunk_ids': [...]}, ...]
    test_gt = [
        {
            'query': "How do I reset my password?",
            'relevant_chunk_ids': ["doc123_chunk0", "doc218_chunk1"]  # placeholder example
        },
        {
            'query': "Steps for product return",
            'relevant_chunk_ids': ["doc877_chunk2", "doc611_chunk0"]
        },
        {
            'query': "How to escalate high priority issues?",
            'relevant_chunk_ids': ["doc390_chunk1"]
        }
    ]
    recall_val = recall_at_k(test_gt, chroma_client, collection_name, model, k=5)
    print(f"\nRecall@5 on test set: {recall_val:.3f}")
