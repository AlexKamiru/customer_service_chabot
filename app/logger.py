# app/logger.py
import sys
import logging
import json
import os
from datetime import datetime
from typing import List, Optional

from app.schemas import RetrievedChunk

# ----------------------------------
# REsolve absolute log path
# ----------------------------------

LOG_FILE = os.path.join(os.getcwd(), "chatbot.log")

# ----------------------------------
# Configure logger once
# ----------------------------------

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    #file handler for local development
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    #console handler critical for render /container logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# ----------------------------------
# Structured logging function
# ----------------------------------

def log_query(
    user_question: str,
    retrieved_chunks: Optional[List[RetrievedChunk]],
    llm_response: Optional[str],
    level: str = "info",
    request_id: Optional[str] = None,
) -> None:
    """
    Logs a structured RAG interaction.
    Expects retrieved_chunks to be List[RetrievedChunk] — not dicts.
    """

    log_payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
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

    log_json = json.dumps(log_payload, ensure_ascii=False)

    if level.lower() == "error":
        logger.error(log_json)
    else:
        logger.info(log_json)