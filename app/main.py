# app/main.py
import uuid
from fastapi import FastAPI, HTTPException
from app.schemas import RAGRequest, RAGResponse, SourceReference, RetrievedChunk
from app.retriever import retrieve
from app.llm import generate_answer
from app.logger import log_query
from typing import List
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Customer Service Chatbot API",
    description="RAG-based chatbot that retrieves documents and generates answers with citations.",
    version="1.0"
)

#Global EXception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions.
    log full traceback internally.
    Return sanitized error response externally.
    """

    error_trace = traceback.format_exc()

    import logging 
    logging.getLogger("rag_logger").error(
        f"Unhandled error at {request.url.path}\n{error_trace}"
    )

    log_query(
        user_question= "UNHANDLED_EXCEPTION",
        retrieved_chunks=None,
        llm_response=None,
        level="error"
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

#Health Check (Liveness Probe)
@app.get("/health")
def health_check():
    return{"status": "ok"}

#Readiness Check
@app.get("/ready")
def readiness_check():
    try:
        #lightweight test call
        _ = retrieve("test query")
        return{"status":"ready"}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"service not ready:{str(e)}")

#REQUEST ID 
def generate_request_id() -> str:
    '''Generate a unique request ID for tracing''' 
    return str(uuid.uuid4())   

#chat Endpoint
@app.post("/chat", response_model=RAGResponse)
def chat(request: RAGRequest):
    #Generate Request ID
    request_id = generate_request_id()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # 1️ Retrieve top-k chunks
    retrieved_chunks: List[RetrievedChunk] = retrieve(question)

    # 2️ Handle empty retrieval gracefully
    if not retrieved_chunks:
        answer_text = "Sorry, I could not find relevant information."
        sources: List[SourceReference] = []
    else:
        try:
              # 3️ Generate grounded answer
            rag_response: RAGResponse = generate_answer(retrieved_chunks, question)
            answer_text = rag_response.answer
            sources = rag_response.sources
        except Exception as e:
            #Fallback If LLM fails
            answer_text = "The answer generation service is currently unavailable. Showing retrieved context only:\n\n"

            for chunk in retrieved_chunks:
                answer_text += f"-{chunk.text[:150]}...\n"

            sources = [
                SourceReference(
                    source_file=chunk.source_file,
                    chunk_id=chunk.chunk_id
                )
                for chunk in retrieved_chunks
            ]       

    # 4️ Log with Request ID interaction
    log_query(
        user_question=question,
        retrieved_chunks=retrieved_chunks,
        llm_response=answer_text,
        level="info",
        request_id = request_id
        )

    return RAGResponse(answer=answer_text, sources=sources)

