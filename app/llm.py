"""
llm.py

generates grounded answers from retrieved chunks using the local Ollama model.
Returns structured citations with the answer.
"""


import ollama
from ollama import Client

MODEL_NAME= "tinyllama"  #the local model

#use ngrok URL for tunneling
OLLAMA_HOST = " https://fibrocartilaginous-nicki-intricately.ngrok-free.dev"

client = Client(host= OLLAMA_HOST)

from app.prompts import RAG_PROMPT_TEMPLATE
from app.schemas import RetrievedChunk, RAGResponse, SourceReference
from typing import List

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
    response = client.chat(
        model = MODEL_NAME,
        messages= [{"role":"user", "content":prompt}]
    )

    answer_text =response["message"]["content"]

    #5 Build deterministic sources section
    sources: List[SourceReference] = [
        SourceReference(source_file=chunk.source_file, chunk_id=chunk.chunk_id)
        for _, chunk in numbered_chunks
    ]

    return RAGResponse(answer=answer_text, sources=sources)