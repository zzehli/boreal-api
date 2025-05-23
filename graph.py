import asyncio
import os
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import START, StateGraph

from rag import (
    Document,
    TermSearchQuery,
    VectorSearchQuery,
    initialize_vector_store,
    search_documents_semantic,
    search_documents_term_based,
)

load_dotenv()

llm = AzureChatOpenAI(
    api_version="2024-12-01-preview",
    model="gpt-4o",
)
class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    # structured_llm = llm.with_structured_output(Search)
    # print(f'question: {state["question"]}')
    query = VectorSearchQuery(query=state["question"], k=3, score_threshold=0)
    return {"query": query}


async def retrieve(state: State):
    embeddings = AzureOpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    vector_store = initialize_vector_store(embeddings)
    query = state["query"]
    results = await search_documents_semantic(vector_store, query)
    source_urls = [result.document.metadata.get("source_url") for result in results if result.document.metadata.get("source_url")]
    if source_urls:
        source_urls = source_urls[:1]
        filter_expression = " or ".join([f"source_url eq '{url}'" for url in source_urls])
        full_text_query = TermSearchQuery(query="*", filter=filter_expression)
    
    retrieved_docs = await search_documents_term_based(full_text_query)
    for doc in retrieved_docs:
        print("--------------------------------")
        print(doc)
    return {"context": retrieved_docs}


async def generate(state: State):
    docs_content = "\n\n".join(doc.document.content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = await llm.ainvoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")

graph = graph_builder.compile()
async def main():
    response = await graph.ainvoke({"question": "Give me a recipe for Cereal Clusters"})
    print(response["answer"])

if __name__ == "__main__":
    asyncio.run(main())