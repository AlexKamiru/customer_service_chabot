"""
Retriever.py 

Retrieves the top-k most relevant chunks for a given query using FAISS.
Now returns RetrievedChunk objects(typed pydantic models).
"""

import faiss
import numpy as np
import pickle
import voyageai
from typing import List

from app.config import FAISS_INDEX_FILE, METADATA_FILE, TOP_K
from app.schemas import RetrievedChunk

# Voyage client
client = voyageai.Client()  # reads VOYAGE_API_KEY from env

# Don't load at import time — load lazily after vector store is built
index = None
metadata = None


def load_vector_store():
    global index, metadata
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)


def embed(texts: List[str]) -> np.ndarray:
    response = client.embed(texts, model="voyage-3-lite")
    return np.array(response.embeddings, dtype="float32")


def retrieve(query: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """
    Retrieve top_k most similar chunks for a user query.

    Args:
        query (str): User query string
        top_k (int): Number of top results to return

    Returns:
        List[RetrievedChunk]: Retrieved chunks with similarity scores
    """
    global index, metadata

    # Load index lazily — ensures we get the freshly built index
    if index is None:
        load_vector_store()

    # Step 1: Embed the query
    query_embedding = embed([query])

    # Step 2: Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results: List[RetrievedChunk] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            entry = metadata[idx]
            chunk = RetrievedChunk(
                text=entry["text"],
                source_file=entry["source_file"],
                chunk_id=entry["chunk_id"],
                score=float(dist)
            )
            results.append(chunk)

    return results