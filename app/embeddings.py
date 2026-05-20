import os
import faiss
import numpy as np
import pickle
import anthropic

from app.config import (
    DATA_PATH,
    FAISS_INDEX_FILE,
    METADATA_FILE
)

# anthropic client 
client = anthropic.Anthropic()

EMBEDDING_MODEL = "voyage-3-lite"
EMBEDDING_DIMENSION = 512

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
    Generate embeddings using Anthropic's Voyage API.
    """
    all_embeddings = []
    batch_size = 128

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model = EMBEDDING_MODEL,
            input = batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings,dtype="float32")    


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
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(embeddings)

    # Save FAISS index
    os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print("Vector store created successfully.")