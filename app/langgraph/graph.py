import os
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain import hub
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import Literal

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

class Route(BaseModel):
    step: Literal["rag", "chat"] = Field(
        None, description="The next step in the routing process"
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
    # messages: Annotated[list, add_messages]
    step: str

class RAGGraph:
    def __init__(self):
        self._graph: Optional[CompiledStateGraph] = None

    def _router(self, state: State):
        """Route the user to rag or chat."""
        router = llm.with_structured_output(Route)
        decision = router.invoke(
            [
                SystemMessage(content="""You are a customer agent for Nestle. Route the user input,
                              determine if this question needs retrieval (RAG) from the database of Nestle's 
                              website or if it can be answered by the chatbot based on the context provided (chat)."""),
                HumanMessage(content=state["question"])
            ]
        )
        print(f"decision: {decision}")
        return {"step": decision.step}
    
    def _router_decision(self, state: State):
        """Route the user to rag or chat."""
        if state["step"] == "rag":
            return "rag"
        else:
            return "chat"
    
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

    async def _chat(self, state: State):
        """Chat with the user."""
        response = llm.invoke(
            [
                SystemMessage(content="""You are a customer agent for Nestle. Aswered customer's question based on the context provided. Ask clarification questions if needed."""),
                HumanMessage(content=state["question"])
            ]
        )
        return {"answer": response.content}

    def _create_graph(self) -> Optional[CompiledStateGraph]:
        graph_builder = StateGraph(State).add_sequence([self._analyze_query, self._retrieve, self._generate])
        graph_builder.add_node("router", self._router)
        graph_builder.add_node("chat", self._chat)
        graph_builder.add_edge(START, "router")
        graph_builder.add_conditional_edges(
            "router",
            self._router_decision,
            {
                "rag": "_analyze_query",
                "chat": "chat",
            }
        )

        graph_builder.add_edge("chat", END)
        memory = InMemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    async def get_response(self, question: str, session_id: str):
        if self._graph is None:
            print("Creating graph")
            self._graph = self._create_graph()
        print(f"question: {question}")
        config = {"configurable": {"thread_id": session_id}}
        print(config)
        print(list(self._graph.get_state_history(config=config)))
        response = await self._graph.ainvoke({"question": question}, config=config)
        return response["answer"]
    
    def draw_graph(self):
        if self._graph is None:
            self._graph = self._create_graph()
        with open("graph.png", "wb") as f:
            f.write(self._graph.get_graph(xray=0).draw_mermaid_png())

if __name__ == "__main__":
    graph = RAGGraph()
    graph.draw_graph()