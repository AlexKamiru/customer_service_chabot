"""
llm.py

generates grounded answers from retrieved chunks using the local Ollama model.
Returns structured citations with the answer.
"""
import requests
import os
from dotenv import load_dotenv
from app.prompts import RAG_PROMPT_TEMPLATE
from app.schemas import RetrievedChunk, RAGResponse, SourceReference
from typing import List

# ----------------------------
# Hugging Face Configuration.
# ----------------------------

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in environment variables")

headers = {
    "Authorization" : f"Bearer {HF_TOKEN}"
}

# ----------------------------
# Hugging Face Query Function
# ----------------------------

def query_hf(prompt: str) -> str:
    payload = {
        "inputs" : prompt,
        "parameters" : {
            "max_new_tokens" : 300,
            "temperature" : 0.7
        }
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        raise Exception( f"HuggingFace API error: {response.text}")
    
    result = response.json()

    # Handle different HF response formats
    if isinstance(result, list):
        return result[0].get("generated_text","")
    
    # fallback
    return str(result)

# --------------------------
# Main RAH Function
# --------------------------

def generate_answer(context_chunks: List[RetrievedChunk], question: str) -> RAGResponse:
    """
    Generate a grounded answer with structured citations. 
    """

    #1 Assign numeric IDs to chunks for citation
    numbered_chunks = list(enumerate(context_chunks, start=1))

    #2 Build context with explicit chunk IDs
    context_text = "\n\n".join(
        [
            f"[{i}] Source: {chunk.source_file} (chunk {chunk.chunk_id})\n{chunk.text}"
            for i, chunk in numbered_chunks
        ]
    )    
    
    #3 Build prompt
    prompt = RAG_PROMPT_TEMPLATE.format(
        context = context_text,
        question = question
     )
    
    #4 Generate response
    answer_text = query_hf(prompt)

    #5 Build structured sources
    sources: List[SourceReference] = [
        SourceReference(source_file=chunk.source_file, chunk_id=chunk.chunk_id)
        for _, chunk in numbered_chunks
    ]

    return RAGResponse(answer=answer_text, sources=sources)