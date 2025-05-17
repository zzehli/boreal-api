from langchain_core.vectorstores import InMemoryVectorStore
from typing import Literal
import os
os.environ['USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from typing_extensions import Annotated, List, TypedDict

from dotenv import load_dotenv
load_dotenv()
embed_model_name = "text-embedding-3-large"
print(os.environ["AZURE_INFERENCE_ENDPOINT"])
# https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/langchain#using-azure-openai-models
# https://github.com/langchain-ai/langchain-azure
def main():
    llm = AzureAIChatCompletionsModel(
        endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
        credential=os.environ["AZURE_INFERENCE_CREDENTIAL"],
        api_version="2025-01-01-preview",
        model_name="gpt-4o",
    )

    # print(llm.invoke("What is the capital of France?"))
    # llm = init_chat_model("gpt-4o-mini", model_provider="openai")

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
    print("line 60")

    embed_model = AzureAIEmbeddingsModel(
        model_name=embed_model_name,
    )
    vector_store = InMemoryVectorStore(embed_model)
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

    print("line 83")

    # Define state for application
    class State(TypedDict):
        question: str
        query: Search
        context: List[Document]
        answer: str


    def analyze_query(state: State):
        # set tool_choice is a problem
        llm_with_tools = llm.bind_tools([Search], tool_choice="auto")
        
        # Then use an output parser to extract the structured output
        from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
        output_parser = JsonOutputKeyToolsParser(key_name="Search", first_tool_only=True)
        
        # Create the chain
        structured_llm = llm_with_tools | output_parser
        
        print(structured_llm)
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

    print("line 114")

    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    print("line 117")

    graph = graph_builder.compile()
    response = graph.invoke({"question": "What is Task Decomposition?"})
    print(response["answer"])

if __name__ == "__main__":
    main()