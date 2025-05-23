"""
Vector store operations using Azure Search.
"""

import json
import os
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SimpleField,
)
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

from app.rag.models import Document, SearchResult, TermSearchQuery, VectorSearchQuery


def initialize_vector_store(embeddings: AzureOpenAIEmbeddings) -> AzureSearch:
    """Initialize Azure Search vector store with proper schema."""
    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=len(embeddings.embed_query("Text")),
            vector_search_profile_name="myHnswProfile",
        ),
        SearchField(
            name="category",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
        ),
        SearchField(
            name="source_url",
            type=SearchFieldDataType.String,
            filterable=True,
            searchable=True,
        ),
    ]
    return AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
        index_name="nestle-rag",
        embedding_function=embeddings.embed_query,
        fields=fields,
    )

async def search_documents_semantic(
    vector_store: AzureSearch,
    query: VectorSearchQuery
) -> List[SearchResult]:
    """Search documents using the vector store."""
    results = await vector_store.ahybrid_search_with_score(
        query=query.query,
        k=query.k,
    )
    
    return [
        SearchResult(
            document=Document(
                content=doc.page_content,
                metadata=doc.metadata,
                source_url=doc.metadata.get("source_url", ""),
                category=doc.metadata.get("category", ""),
            ),
            score=score
        )
        for doc, score in results
    ]

async def search_documents_term_based(query: TermSearchQuery):
    """Search documents using the full-text search, no vectors involved."""
    # use azure search client to perform term-based search since langchain doesn't support it
    async with SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"), 
        credential= AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")), 
        index_name="nestle-rag"
        ) as search_client:

        results = await search_client.search(
            search_text=query.query,
            filter=query.filter,
            search_fields=query.search_fields,
            top=query.k,
        )
        return [
            SearchResult(
                document=Document(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else json.loads(doc.get("metadata", "{}")),
                    source_url=doc.get("source_url", ""),
                    category=doc.get("category", ""),
                ),
                score=float(doc["@search.score"])
            )
            async for doc in results
        ]