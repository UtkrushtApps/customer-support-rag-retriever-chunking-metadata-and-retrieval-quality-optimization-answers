# chunk_and_ingest.py
import os
import json
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime

# Chunking function: overlap, 2000-token chunks, 200-token overlap
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    tokens = text.split()  # crude whitespace tokenization
    if len(tokens) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap
    return chunks

def load_docs(doc_dir: str) -> List[Dict]:
    """
    Each doc is a JSON file with at least: {"id": str, "text": str, "category": str, "priority": str, "date": str (ISO)}
    """
    docs = []
    for fname in os.listdir(doc_dir):
        if fname.endswith('.json'):
            with open(os.path.join(doc_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                docs.append(data)
    return docs

def prepare_chunks(docs: List[Dict], chunk_size: int = 2000, overlap: int = 200) -> List[Dict]:
    all_chunks = []
    for doc in docs:
        doc_id = doc['id']
        category = doc.get('category', '')
        priority = doc.get('priority', '')
        date = doc.get('date', '')
        text = doc['text']
        chunks = chunk_text(text, chunk_size, overlap)
        for idx, chunk_text_ in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk{idx}"
            metadata = {
                "orig_id": doc_id,
                "chunk_id": chunk_id,
                "category": category,
                "priority": priority,
                "date": date,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            }
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text_,
                "metadata": metadata
            })
    return all_chunks

def batch_embed_texts(texts: List[str], model, batch_size: int = 32) -> List[List[float]]:
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Embedding'):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.extend(batch_emb)
    return embeddings

def upsert_chunks_to_chroma(chunks: List[Dict], chroma_client, collection_name: str):
    collection = chroma_client.get_or_create_collection(collection_name)
    ids = [chunk['id'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    documents = [chunk['text'] for chunk in chunks]
    batch_size = 128
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for i in tqdm(range(0, len(ids), batch_size), desc='Upserting'):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_embeds = model.encode(batch_docs, show_progress_bar=False, normalize_embeddings=True)
        # Idempotent upserts: delete possible preexisting ids first
        try:
            collection.delete(ids=batch_ids)
        except Exception:
            pass
        collection.add(
            ids=batch_ids,
            documents=batch_docs, 
            metadatas=batch_metas, 
            embeddings=[emb.tolist() for emb in batch_embeds]
        )

if __name__ == "__main__":
    # Assume docs are in docs_raw/ as .json
    doc_dir = 'docs_raw'
    collection_name = 'customersupport_rag_chunks'
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="chromadb.db.impl.sqlite",
        persist_directory="./chroma_data"
    ))
    print("Loading docs...")
    docs = load_docs(doc_dir)
    print("Chunking docs...")
    all_chunks = prepare_chunks(docs, chunk_size=2000, overlap=200)
    print(f"Total chunks: {len(all_chunks)}")
    print("Upserting chunks and embeddings to Chroma...")
    upsert_chunks_to_chroma(all_chunks, chroma_client, collection_name)
    print("Done.")
