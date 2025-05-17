import os
from typing import Literal

from langchain_core.vectorstores import InMemoryVectorStore

os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, List, TypedDict

load_dotenv()
embed_model_name = "text-embedding-3-large"

def main():
    llm = AzureChatOpenAI(
        api_version="2024-12-01-preview",
        model="gpt-4o",
    )
    # print(llm.invoke("What is the capital of France?"))
    
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        )
    
    vector_store = InMemoryVectorStore(embeddings)

    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    
    # Update metadata (illustration purposes)
    total_documents = len(all_splits)
    third = total_documents // 3

    for i, document in enumerate(all_splits):
        if i < third:
            document.metadata["section"] = "beginning"
        elif i < 2 * third:
            document.metadata["section"] = "middle"
        else:
            document.metadata["section"] = "end"


    _ = vector_store.add_documents(all_splits)

    
    # Define schema for search
    class Search(TypedDict):
        """Search query."""

        query: Annotated[str, ..., "Search query to run."]
        section: Annotated[
            Literal["beginning", "middle", "end"],
            ...,
            "Section to query.",
        ]

    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")

    # Define state for application
    class State(TypedDict):
        question: str
        query: Search
        context: List[Document]
        answer: str


    def analyze_query(state: State):
        structured_llm = llm.with_structured_output(Search)
        query = structured_llm.invoke(state["question"])
        return {"query": query}


    def retrieve(state: State):
        query = state["query"]
        retrieved_docs = vector_store.similarity_search(
            query["query"],
            filter=lambda doc: doc.metadata.get("section") == query["section"],
        )
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}


    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")

    graph = graph_builder.compile()
    response = graph.invoke({"question": "What is Task Decomposition?"})
    print(response["answer"])

if __name__ == "__main__":
    main()