# app/main.py

from fastapi import FastAPI, HTTPException
from app.schemas import RAGRequest, RAGResponse, SourceReference, RetrievedChunk
from app.retriever import retrieve
from app.llm import generate_answer
from app.logger import log_query
from typing import List

app = FastAPI(
    title="Customer Service Chatbot API",
    description="RAG-based chatbot that retrieves documents and generates answers with citations.",
    version="1.0"
)


@app.post("/chat", response_model=RAGResponse)
def chat(request: RAGRequest):

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # 1️ Retrieve top-k chunks
        retrieved_chunks: List[RetrievedChunk] = retrieve(question)

        # 2️ Handle empty retrieval gracefully
        if not retrieved_chunks:
            answer_text = "Sorry, I could not find relevant information."
            sources: List[SourceReference] = []
        else:
            # 3️ Generate grounded answer
            rag_response: RAGResponse = generate_answer(retrieved_chunks, question)
            answer_text = rag_response.answer
            sources = rag_response.sources

        # 4️ Log the interaction
        log_query(
            user_question=question,
            retrieved_chunks=retrieved_chunks,
            llm_response=answer_text,
            level="info"
        )

        # 5️ Return structured response
        return RAGResponse(answer=answer_text, sources=sources)

    except Exception as e:
        # 6️ Log unexpected errors
        log_query(
            user_question=question,
            retrieved_chunks=retrieved_chunks if 'retrieved_chunks' in locals() else None,
            llm_response=None,
            level="error"
        )
        raise HTTPException(status_code=500, detail=str(e))