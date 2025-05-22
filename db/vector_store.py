"""
Vector store operations using Azure Search.
"""

import os
from typing import List

from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SimpleField,
)
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

from .models import Document, SearchQuery, SearchResult


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

async def search_documents(
    vector_store: AzureSearch,
    query: SearchQuery
) -> List[SearchResult]:
    """Search documents using the vector store."""
    results = await vector_store.ahybrid_search(
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
            # score=0
        )
        for doc in results
    ] 