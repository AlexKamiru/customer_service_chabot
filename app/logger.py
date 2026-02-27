# app/logger.py

import logging
import json
from datetime import datetime
from typing import List, Optional
from app.schemas import RetrievedChunk


# ----------------------------------
# Configure logger once
# ----------------------------------

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("chatbot.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# ----------------------------------
# Structured logging function
# ----------------------------------

def log_query(
    user_question: str,
    retrieved_chunks: Optional[List[RetrievedChunk]],
    llm_response: Optional[str],
    level: str = "info",
) -> None:
    """
    Logs a structured RAG interaction.
    Expects retrieved_chunks to be List[RetrievedChunk] — not dicts.
    """

    log_payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_question": user_question,
        "retrieved_chunks": [
            {
                "source_file": chunk.source_file,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "text_preview": chunk.text[:120] if chunk.text else ""
            }
            for chunk in (retrieved_chunks or [])
        ],
        "llm_response_preview": llm_response[:200] if llm_response else None,
    }

    if level.lower() == "error":
        logger.error(json.dumps(log_payload))
    else:
        logger.info(json.dumps(log_payload))