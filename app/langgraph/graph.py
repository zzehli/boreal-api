import os
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from typing_extensions import Literal

from app.langgraph.prompts import (
    CHAT_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    QUERY_ANALYZER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
)
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

class ReferenceItem(BaseModel):
    index: int = Field(description="The index of the document in the context")
    title: str = Field(description="The title or identifier of the document")
    url: str = Field(description="The url of the document")

class ResponseWithCitation(BaseModel):
    response: str = Field(description="The response to user's question")
    reference: List[ReferenceItem] = Field(description="A list of references from the context used to generate the response")

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]

# Define state for application
class State(TypedDict):
    question: str  # Current question text
    messages: Annotated[list, add_messages]  # For conversation history

    # Optional fields based on graph path
    query: Optional[Search]
    context: Optional[List[Document]]
    # answer: Optional[str]  # Can be derived from the last AIMessage
    step: Optional[Literal["rag", "chat"]]
    structured_response: Optional[ResponseWithCitation]  # Add this field

class RAGGraph:
    def __init__(self):
        self._graph: Optional[CompiledStateGraph] = None
        self._config: Optional[RunnableConfig] = None

    def _router(self, state: State):
        """Route the user to rag or chat."""
        router = llm.with_structured_output(Route)
        decision = router.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
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
    
    async def _analyze_query(self, state: State):
        """Enrich the query, to be implemented."""
        messages = [
                *state["messages"],
                SystemMessage(content=QUERY_ANALYZER_SYSTEM_PROMPT),
                HumanMessage(content=state["question"])
        ]


        response = await llm.ainvoke(messages)
        print(f"query analyzer response: {response}")
        query = VectorSearchQuery(query=response.content, k=4, score_threshold=0)
        return {"query": query, "question": response.content, "messages": [AIMessage(content=response.content)]}

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
        return {"context": retrieved_docs}


    async def _generate(self, state: State):
        docs_content = "\n\n".join(f"[{i}] {doc.document.content}" for i, doc in enumerate(state["context"]))
        template = ChatPromptTemplate([
            ("system", GENERATION_SYSTEM_PROMPT),
            ("human", "Context: {context}\n Question: {question}\n Answer:"),
        ])

        input = template.invoke({"question": state["question"], "context": docs_content})
        messages = [
            *state["messages"],
            *input.to_messages()
        ]
        
        response = await llm.with_structured_output(ResponseWithCitation).ainvoke(messages)

        # response.reference = [ReferenceItem(index=doc.document.metadata.get("index"), title=doc.document.metadata.get("title"), url=doc.document.metadata.get("source_url")) for doc in state["context"]]
        for ref in response.reference:
            ref.url = state["context"][ref.index].document.metadata.get("source_url")
        
        print("new response: ", response.model_dump_json())
        # Return both the structured response and a formatted message
        return {
            "messages": [AIMessage(content=response.response)],  # Just the response text
            "structured_response": response  # Keep the full structured response
        }

    async def _chat(self, state: State):
        """Chat with the user."""
        messages = [
            *state["messages"],
            SystemMessage(content=CHAT_SYSTEM_PROMPT),
            HumanMessage(content=state["question"])
        ]
        
        response = await llm.ainvoke(messages)
        return {"messages": [AIMessage(content=response.content)], "structured_response": None}

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
            self._graph = self._create_graph()
            self._config = {"configurable": {"thread_id": session_id}}
        response = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=question)], 
             "question": question},
            config=self._config)
        
        if response["messages"] and isinstance(response["messages"][-1], AIMessage):
            # Return both the message and structured data if available
            return {
                "message": response["messages"][-1].content,
                "references": response["structured_response"].reference if response["structured_response"] else []
            }
        return "No response generated"
    
    def draw_graph(self):
        if self._graph is None:
            self._graph = self._create_graph()
        with open("graph.png", "wb") as f:
            f.write(self._graph.get_graph(xray=0).draw_mermaid_png())

if __name__ == "__main__":
    graph = RAGGraph()
    graph.draw_graph()