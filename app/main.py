# app/main.py

from fastapi import FastAPI, HTTPException
from app.schemas import (
    RAGRequest,
    RAGResponse,
    SourceReference,
    RetrievedChunk,
)
from app.retriever import retrieve
from app.llm import generate_answer
from app.logger import log_query

app = FastAPI()


@app.post("/chat", response_model=RAGResponse)
def chat(request: RAGRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # ----------------------------------
        # 1. Retrieval (returns List[dict])
        # ----------------------------------
        raw_chunks = retrieve(question)

        # ----------------------------------
        # 2. Convert dict -> RetrievedChunk
        # ----------------------------------
        retrieved_chunks = [
            RetrievedChunk(
                source_file=chunk["source_file"],
                chunk_id=chunk["chunk_id"],
                score=chunk.get("score"),
                text=chunk.get("text", ""),
            )
            for chunk in raw_chunks
        ]

        # ----------------------------------
        # 3. Generate Answer
        # ----------------------------------
        answer = generate_answer(raw_chunks, question)

        # ----------------------------------
        # 4. Build API response sources
        # ----------------------------------
        sources = [
            SourceReference(
                source_file=chunk.source_file,
                chunk_id=chunk.chunk_id,
            )
            for chunk in retrieved_chunks
        ]

        # ----------------------------------
        # 5. Structured Logging
        # ----------------------------------
        log_query(
            user_question=question,
            retrieved_chunks=retrieved_chunks,
            llm_response=answer,
            level="info",
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as e:

        # Error logging (no chunks available)
        log_query(
            user_question=question,
            retrieved_chunks=None,
            llm_response=None,
            level="error",
        )

        raise HTTPException(status_code=500, detail=str(e))