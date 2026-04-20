"""
Service: Vector Store (ChromaDB + BM25 Hybrid RAG)
Combines dense (semantic) and sparse (keyword) search for better retrieval.
"""

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from app.config import settings
from app.services.llm_service import ask_llm

# Module-level cached instances
_client = None
_collection = None
_bm25 = None
_bm25_chunks = []  # stores original chunk text
_bm25_metadata = []  # stores metadata for each chunk


def get_collection():
    """Initialize ChromaDB client and collection."""
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)

        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )

        _collection = _client.get_or_create_collection(
            name="resumes",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def add_resumes_to_store(resumes: list):
    """
    Embed and store all resume texts in both ChromaDB (dense) and BM25 (sparse).
    """
    global _bm25, _bm25_chunks, _bm25_metadata

    collection = get_collection()

    # Clear existing ChromaDB data
    try:
        existing = collection.count()
        if existing > 0:
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
    except Exception:
        pass

    documents = []
    metadatas = []
    ids = []

    for i, resume in enumerate(resumes):
        raw_text = resume.get("raw_text", "")
        filename = resume.get("original_filename", resume.get("filename", f"resume_{i}"))

        if not raw_text.strip():
            continue

        chunks = split_text(raw_text, chunk_size=500, overlap=50)

        for j, chunk in enumerate(chunks):
            doc_id = f"{filename}_chunk_{j}"
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "chunk_index": j,
                "total_chunks": len(chunks),
            })
            ids.append(doc_id)

    if documents:
        # Store in ChromaDB (dense search)
        batch_size = 40
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

        # Build BM25 index (sparse search)
        _bm25_chunks = documents
        _bm25_metadata = metadatas
        tokenized = [chunk.lower().split() for chunk in documents]
        _bm25 = BM25Okapi(tokenized)

    return {
        "total_resumes": len(resumes),
        "total_chunks": len(documents),
        "search_modes": "hybrid (dense + sparse)",
        "status": "indexed",
    }


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


def hybrid_search(question: str, n_results: int = 5) -> list:
    """
    Hybrid search combining:
    1. Dense search (ChromaDB - semantic similarity)
    2. Sparse search (BM25 - keyword matching)
    
    Results are merged using Reciprocal Rank Fusion (RRF).
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    # --- Dense search (ChromaDB) ---
    dense_results = collection.query(
        query_texts=[question],
        n_results=min(n_results * 2, collection.count()),
    )

    dense_docs = []
    if dense_results["documents"] and dense_results["documents"][0]:
        for doc, meta in zip(dense_results["documents"][0], dense_results["metadatas"][0]):
            dense_docs.append({"text": doc, "metadata": meta})

    # --- Sparse search (BM25) ---
    sparse_docs = []
    if _bm25 is not None and _bm25_chunks:
        tokenized_query = question.lower().split()
        bm25_scores = _bm25.get_scores(tokenized_query)

        # Get top results
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        top_indices = top_indices[:n_results * 2]

        for idx in top_indices:
            if bm25_scores[idx] > 0:
                sparse_docs.append({
                    "text": _bm25_chunks[idx],
                    "metadata": _bm25_metadata[idx],
                })

    # --- Reciprocal Rank Fusion (RRF) ---
    # Merge results from both methods, giving weight to rank position
    k = 60  # RRF constant

    doc_scores = {}  # key: chunk text -> score

    # Score dense results
    for rank, doc in enumerate(dense_docs):
        key = doc["text"][:100]  # use first 100 chars as key
        rrf_score = 1 / (k + rank + 1)
        if key in doc_scores:
            doc_scores[key]["score"] += rrf_score
        else:
            doc_scores[key] = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": rrf_score,
                "source": "dense",
            }

    # Score sparse results
    for rank, doc in enumerate(sparse_docs):
        key = doc["text"][:100]
        rrf_score = 1 / (k + rank + 1)
        if key in doc_scores:
            doc_scores[key]["score"] += rrf_score
            doc_scores[key]["source"] = "both"  # found in both methods
        else:
            doc_scores[key] = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": rrf_score,
                "source": "sparse",
            }

    # Sort by combined RRF score and return top N
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return sorted_docs[:n_results]


def query_resumes(question: str, jd_text: str = "", chat_history: list = None, n_results: int = 5) -> str:
    """
    Hybrid RAG query:
    1. Hybrid search (dense + sparse) for relevant chunks
    2. Send chunks + JD + chat history + question to LLM
    3. Return LLM's answer
    """
    collection = get_collection()

    if collection.count() == 0:
        return "No resumes have been indexed yet. Please upload and analyze resumes first."

    # Step 1: Hybrid search
    results = hybrid_search(question, n_results)

    if not results:
        return "I couldn't find any relevant information in the uploaded resumes."

    # Step 2: Build context from retrieved chunks
    context_parts = []
    seen_files = set()

    for doc in results:
        filename = doc["metadata"].get("filename", "Unknown")
        search_source = doc.get("source", "unknown")
        if filename not in seen_files:
            context_parts.append(f"\n--- Resume: {filename} ---")
            seen_files.add(filename)
        context_parts.append(doc["text"])

    context = "\n".join(context_parts)

    # Step 3: Build chat history string
    history_str = ""
    if chat_history:
        recent = chat_history[-6:]
        history_parts = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "ResumeAI"
            history_parts.append(f"{role}: {msg['content']}")
        history_str = "\n".join(history_parts)

    # Step 4: Build prompt
    system_prompt = """You are ResumeAI, an intelligent assistant that answers questions about uploaded resumes.
You have access to resume data and the job description provided as context.
Answer the user's question based on the resume data and JD provided.
If the answer is not in the context, say so honestly.
Be specific — mention candidate names, skills, and details.
Use the chat history to understand follow-up questions.
Keep answers concise and helpful."""

    user_prompt = ""

    if jd_text:
        user_prompt += f"Job Description:\n{jd_text[:1000]}\n\n"

    user_prompt += f"Resume Data:\n{context[:2500]}\n\n"

    if history_str:
        user_prompt += f"Chat History:\n{history_str}\n\n"

    user_prompt += f"Latest Question: {question}\n\nAnswer:"

    try:
        answer = ask_llm(system_prompt, user_prompt, max_tokens=600)
        return answer.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"