"""
schemas.py 

This module defines all structured data contracts used across the RAG system.

We use Pydantic models to:
- Enforce strict typing
- Validate data automatically
- Standardize communication between layers
- Enable FastAPI integration
"""

from pydantic import BaseModel, Field
from typing import List


class RetrievedChunk(BaseModel):
    """
    Represents a single chunk retrieved from the vector store.

    This replaces fragile dictionary structures like:
    {
        "text": "...",
        "source_file": "...",
        "chunk_id": 3,
        "score": 0.87
    }
    """

    text: str = Field(
        ...,
        description="The raw text content of the retrieved chunk"
    )

    source_file:str=Field(
        ...,
        description="The file from which this chunk originated (e.g., faqs.txt)."
    )

    chunk_id: int = Field(
        ...,
        description="The unique identifier of the chunk within its source file."
    )

    score: float = Field(
        ...,
        description="Similarity score returned by FAISS (higher = more relevant)."
    )



class RAGRequest(BaseModel):
    """
    Defines the expected structure of a user query sent to the API.
    """    

    question: str = Field(
        ...,
        min_length=3,
        description="The user's natural language question."
    )


class SourceReference(BaseModel):
    """
    Represents a structured citation reference returned in the response.
    """

    source_file: str = Field(
        ..., 
        description="Name of the source file."
    )

    chunk_id: int = Field(
        ...,
        description="Chunk identifier within the file."
    )


class RAGResponse(BaseModel):
    """
    Defines the full structured response returned by the API.
    """

    answer: str =Field(
        ...,
        description="The final LLM-generated grounded answer."
    )

    sources: List[SourceReference]= Field(
        ...,
        description="List of source references used to generate the answer."
    )