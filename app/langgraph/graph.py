import asyncio
import os
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain import hub
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from app.rag import (
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
    messages: Annotated[list, add_messages]

class RAGGraph:
    def __init__(self):
        self._graph: Optional[CompiledStateGraph] = None

    def _analyze_query(self, state: State):
        """Enrich the query, to be implemented."""
        query = VectorSearchQuery(query=state["question"], k=4, score_threshold=0)
        return {"query": query}


    async def _retrieve(self, state: State):
        embeddings = AzureOpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
        vector_store = initialize_vector_store(embeddings)
        query = state["query"]
        results = await search_documents_semantic(vector_store, query)
        source_urls = list(set(result.document.metadata.get("source_url") for result in results if result.document.metadata.get("source_url")))
        
        if source_urls:
            filter_expression = " or ".join([f"source_url eq '{url}'" for url in source_urls])
            full_text_query = TermSearchQuery(query="*", filter=filter_expression)
        
        retrieved_docs = await search_documents_term_based(full_text_query)
        for doc in retrieved_docs:
            print("--------------------------------")
            print(doc)
        return {"context": retrieved_docs}


    async def _generate(self, state: State):
        docs_content = "\n\n".join(doc.document.content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = await llm.ainvoke(messages)
        return {"answer": response.content}

    def _create_graph(self) -> Optional[CompiledStateGraph]:
        graph_builder = StateGraph(State).add_sequence([self._analyze_query, self._retrieve, self._generate])
        graph_builder.add_edge(START, "_analyze_query")
        memory = MemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    async def get_response(self, question: str, session_id: str):
        if self._graph is None:
            self._graph = self._create_graph()
        print(f"question: {question}")
        config = {"configurable": {"thread_id": session_id}}
        response = await self._graph.ainvoke({"question": question}, config=config)
        return response["answer"]
