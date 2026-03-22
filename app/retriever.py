"""
Retriever.py 

Retrieves the top-k most relevant chunks for a given query using FAISS.
Now returns RetrievedChunk objects(typed pydantic models).
"""


import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List

from app.config import FAISS_INDEX_FILE, METADATA_FILE, TOP_K
from app.schemas import RetrievedChunk   # <- Use the schema

#load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

#Load FAISS index and metadata once
try:
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(METADATA_FILE, "rb") as f:
       metadata = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load vector store: {e}")

def retrieve(query: str, top_k:int = TOP_K) -> List[RetrievedChunk]:
    """
    Retrieve top_k most similar chunks for a user query.


    Returns: Args:
        query (str): User query string
        top_k (int): Number of top results to return

    Returns:
        List[RetrievedChunk]: Retrieved chunks with similarity scores
    """
    #step 1: Embed the query
    query_embedding= model.encode([query], convert_to_numpy= True).astype("float32")

    # step 2: Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results: List[RetrievedChunk] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            entry = metadata[idx]
            #Wrap result in RetrievedChunk
            chunk = RetrievedChunk(
                text=entry["text"],
                source_file=entry["source_file"],
                chunk_id=entry["chunk_id"],
                score=float(dist)  # lower distance = more similar
            )
            results.append(chunk)

    return results        