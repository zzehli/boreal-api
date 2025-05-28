import asyncio
import os
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.langgraph.prompts import (
    CHAT_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    QUERY_ANALYZER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
from app.model.response import ResponseWithCitation
from app.rag import (
    Document,
    TermSearchQuery,
    VectorSearchQuery,
    initialize_vector_store,
    search_documents_semantic,
    search_documents_term_based,
)

questions = [
    "What are some good gift ideas for Christmas?",
    "How many calories are in a kitkat bar?",
    "How many grams of protein in a kitkat 4-finger wafer bar?",
    "What's a healthy cake recipe?",
    "Is Nestle using sustainable practices in their production?",
    "What are the ingredients in Key Haagen-dazs Ice-cream Bars?",
    "How can I recycle Nestle packaging?",
    "What are the health benefits of drinking Milo?",
    "Where can I find Nestle products near me?",
    "What is Nestle's policy on palm oil sourcing?",
    "What are the different flavors of KitKat available?",
    "Is Nestle committed to reducing plastic waste?",
    "How can I contact Nestle customer service?"
]


load_dotenv()


llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    model="gpt-4o",
)

async def retrieve_context(input: str):
    embeddings = AzureOpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    vector_store = initialize_vector_store(embeddings)
    query = VectorSearchQuery(query=input, k=4, score_threshold=0)

    results = await search_documents_semantic(vector_store, query)
    return results

async def enrich_context(results: List[Document]):
    source_urls = list(set(result.document.metadata.get("source_url") for result in results if result.document.metadata.get("source_url")))
    if source_urls:
        filter_expression = " or ".join([f"source_url eq '{url}'" for url in source_urls])
        full_text_query = TermSearchQuery(query="*", filter=filter_expression)

    retrieved_docs = await search_documents_term_based(full_text_query)
    return retrieved_docs

async def generate(retrieved_docs: List[Document], question: str):
        # source_urls = list(set(result.document.metadata.get("source_url") for result in results if result.document.metadata.get("source_url")))
    
        # if source_urls:
        #     filter_expression = " or ".join([f"source_url eq '{url}'" for url in source_urls])
        #     full_text_query = TermSearchQuery(query="*", filter=filter_expression)
        
        # retrieved_docs = await search_documents_term_based(full_text_query)
        docs_content = "\n\n".join(f"[{i}] {doc.document.content}" for i, doc in enumerate(retrieved_docs))
        template = ChatPromptTemplate([
            ("system", GENERATION_SYSTEM_PROMPT),
            ("human", "Context: {context}\n Question: {question}\n Answer:"),
        ])

        input = template.invoke({"question": question, "context": docs_content})
        messages = [
            *input.to_messages()
        ]
        
        response = await llm.with_structured_output(ResponseWithCitation).ainvoke(messages)

        
        return response.response

if __name__ == "__main__":
    for question in questions[:1]:
        results = asyncio.run(retrieve_context(question))
        print("retrieved results: ", results)
        enriched_results = asyncio.run(enrich_context(results))
        print("enriched results: ", enriched_results)
        response = asyncio.run(generate(enriched_results, question))
        print("response: ", response)