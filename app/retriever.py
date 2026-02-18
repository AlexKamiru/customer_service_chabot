import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from app.config import FAISS_INDEX_FILE, METADATA_FILE, TOP_K

#load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

#Load FAISS index and metadata once
index = faiss.read_index(FAISS_INDEX_FILE)
with open(METADATA_FILE, "rb") as f:
    metadata = pickle.load(f)


def retrieve(query, top_k = TOP_K):
    """
    Retrieve top_k most similar chunks for a user query.


    Returns:
        List of dicts: [{chunk_id, text, source_file, score}, ...]
    """
    #step 1: Embed the query
    query_embedding= model.encode([query], convert_to_numpy= True).astype("float32")

    # step 2: Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results=[]
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            entry = metadata[idx].copy()
            entry["score"]= float(dist) #lower distance = more similar
            results.append(entry)


    return results        