# Solution Steps

1. Set up your Chroma DB server and ensure it is running, with the dataset pre-populated or new data available as JSON files in the docs_raw/ directory (each with id, text, category, priority, date).

2. Implement the optimal overlapping chunking logic in chunk_and_ingest.py, splitting each document into 2000-token chunks with a 200-token overlap using simple whitespace tokenization.

3. For each chunk, attach relevant metadata: original doc id, chunk id, category, priority, date, chunk index, and total chunks.

4. Embed each chunk's text using the all-MiniLM-L6-v2 SentenceTransformer, in batches, normalizing embeddings for cosine retrieval.

5. Implement an idempotent Chroma upsert by deleting pre-existing ids before inserting new chunks and their embeddings, also storing the metadata.

6. Run chunk_and_ingest.py to process all docs and populate Chroma with the improved chunk structure and metadata.

7. In rag_retrieval.py, implement query embedding, efficient top-k (k=5) cosine similarity search using Chroma's query API, and retrieve both the text and metadata.

8. Create a basic RAG 'assemble' function that returns the original question and concatenates the retrieved support chunks as context.

9. Implement utilities to spot-check retrieval output by querying and displaying the top results with their metadata.

10. Add a recall@5 evaluation function using a small test set of (query, ground-truth-chunk-ids) pairs to objectively measure retrieval improvements.

11. Run rag_retrieval.py to manually and quantitatively verify that recall@5 and relevant context have improved compared to the original non-overlapping/no-metadata version.

