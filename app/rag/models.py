"""
Data models and schemas for the RAG backend.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for storing content and metadata."""
    content: str = Field(..., description="The main content of the document")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Document metadata")
    source_url: str = Field(..., description="URL where the document was sourced from")
    category: Optional[str] = Field(None, description="Document category")

class SearchResult(BaseModel):
    """Search result model containing documents and scores."""
    document: Document
    score: float = Field(..., description="Relevance score of the search result")
    
class VectorSearchQuery(BaseModel):
    """Search query model."""
    query: str = Field(..., description="The search query text")
    k: int = Field(default=5, description="Number of results to return")
    score_threshold: Optional[float] = Field(None, description="Minimum relevance score threshold") 

class TermSearchQuery(BaseModel):
    """Term search query model."""
    query: str = Field(..., description="The search query text")
    k: int = Field(default=10, description="Number of results to return")
    filter: Optional[str] = Field(None, description="Filter query")
    search_fields: Optional[List[str]] = Field(None, description="Search fields")