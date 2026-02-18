import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL,
    DATA_PATH,
    FAISS_INDEX_FILE,
    METADATA_FILE
)

# Load Sentence-Transformer model once (offline, free)
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast, free

def load_documents():
    """
    Load all .txt files from data folder.
    """
    documents = []
    
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_PATH, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append((filename, text))
    
    return documents


def chunk_text_by_paragraph(text):
    """
    Split text into paragraphs and remove empty chunks.
    """
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def create_embeddings(texts):
    """
    Generate embeddings using local Sentence-Transformer model.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.astype("float32")


def build_vector_store():
    """
    Build FAISS index and save metadata.
    """
    documents = load_documents()

    all_chunks = []
    metadata = []

    for filename, text in documents:
        paragraphs = chunk_text_by_paragraph(text)

        for i, paragraph in enumerate(paragraphs):
            all_chunks.append(paragraph)
            metadata.append({
                "source_file": filename,
                "chunk_id": i,
                "text": paragraph
            })

    print(f"Total chunks: {len(all_chunks)}")

    embeddings = create_embeddings(all_chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("Vector store created successfully.")