"""
Database operations for the RAG backend.
"""
from app.rag.document_store import load_documents, process_documents
from app.rag.models import Document, SearchResult, TermSearchQuery, VectorSearchQuery
from app.rag.vector_store import (
    initialize_vector_store,
    search_documents_semantic,
    search_documents_term_based,
)

__all__ = [
    'initialize_vector_store',
    'search_documents_semantic',
    'load_documents',
    'process_documents',
    'Document',
    'SearchResult',
    'VectorSearchQuery',
    'TermSearchQuery',
    'search_documents_term_based',
] 