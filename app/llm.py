"""
llm.py

generates grounded answers from retrieved chunks using the local Ollama model.
Returns structured citations with the answer.
"""
import os
from typing import List 

import anthropic
from dotenv import load_dotenv 

from app.prompts import RAG_PROMPT_TEMPLATE
from app.schemas import RetrievedChunk, RAGResponse, SourceReference


load_dotenv()

client = anthropic.Anthropic() #reads Anthropic api key from env automatically

MODEL_NAME = "claude-haiku-4-5-20251001" #fastest and cheapest Claude model

# ----------------------------
# Query Function
# ----------------------------

def query_claude(prompt: str) -> str:
    try:
        message = client.messages.create(
            model = MODEL_NAME,
            max_tokens = 500, 
            messages=[
                {"role": "user","content":prompt}
            ]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print("Claude ERROR:", str(e))
        return "The answer generation service is currently unavailable."

# --------------------------
# Main RAG Function
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
    answer_text = query_claude(prompt)

    #5 Build structured sources
    sources: List[SourceReference] = [
        SourceReference(source_file=chunk.source_file, chunk_id=chunk.chunk_id)
        for _, chunk in numbered_chunks
    ]

    return RAGResponse(answer=answer_text, sources=sources)