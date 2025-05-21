from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from db import (
    SearchQuery,
    initialize_vector_store,
    load_documents,
    process_documents,
    search_documents,
)

load_dotenv()
embed_model_name = "text-embedding-3-large"

def main():
    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = initialize_vector_store(embeddings)

    # Load and process documents
    # documents = load_documents("./data/site")
    # processed_docs = process_documents(documents)
    # processed_docs = processed_docs[:10]
    # for i in range(len(processed_docs)):
    #     print("--------------------------------")
    #     print(processed_docs[i])
    # print("--------------------------------")

    # ls = vector_store.add_documents(processed_docs)
    # print(ls)

    # Search
    query = SearchQuery(query="Nestlé", k=3, score_threshold=0)
    results = search_documents(vector_store, query)
    for i in range(len(results)):
        print("--------------------------------")
        print(results[i])
    print("--------------------------------")
    print(len(results))


    # llm = AzureChatOpenAI(
    #     api_version="2024-12-01-preview",
    #     model="gpt-4o",
    # )
    # # print(llm.invoke("What is the capital of France?"))
    
    # embeddings = AzureOpenAIEmbeddings(
    #     model="text-embedding-3-large",
    #     )
    
    # fields = [
    #     SimpleField(
    #         name="id",
    #         type=SearchFieldDataType.String,
    #         key=True,
    #         filterable=True,
    #     ),
    #     SearchField(
    #         name="content",
    #         type=SearchFieldDataType.String,
    #         searchable=True,
    #     ),
    #     SearchField(
    #         name="content_vector",
    #         type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    #         searchable=True,
    #         vector_search_dimensions=len(embeddings.embed_query("Text")),
    #         vector_search_profile_name="pageContent",
    #     ),
    #     SearchField(
    #         name="category",
    #         type=SearchFieldDataType.String,
    #         searchable=True,
    #     ),
    #     SearchField(
    #         name="source_url",
    #         type=SearchFieldDataType.String,
    #         filterable=True,
    #         searchable=True,
    #     ),
    # ]
    # # vector_store = InMemoryVectorStore(embeddings)
    # vector_store: AzureSearch = AzureSearch(
    #     azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    #     azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    #     index_name="azue-search-nestle",
    #     embedding_function=embeddings.embed_query,
    #     fields=fields,
    # )
    # site_files = glob.glob("./data/site/*.md")
    # all_splits = []
    
    # for file_path in site_files[:10]:
    #     loader = TextLoader(file_path, encoding="utf-8")
    #     docs = loader.load()
    #     html_url = Utils.get_url_from_path(file_path)
    
    #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    #     all_splits = text_splitter.split_documents(docs)

    # for doc in all_splits:
    #     print('--------------------------------')
    #     print(doc.page_content)
    # Update metadata (illustration purposes)
    # total_documents = len(all_splits)
    # third = total_documents // 3

    # for i, document in enumerate(all_splits):
    #     if i < third:
    #         document.metadata["section"] = "beginning"
    #     elif i < 2 * third:
    #         document.metadata["section"] = "middle"
    #     else:
    #         document.metadata["section"] = "end"


    # vector_store.add_documents(all_splits)

    # docs = vector_store.similarity_search_with_relevance_scores(
    # query="how to make poached eggs",
    # k=3,
    # # score_threshold=0.10,
    # )
    # print(docs)

    # # Define schema for search
    # class Search(TypedDict):
    #     """Search query."""

    #     query: Annotated[str, ..., "Search query to run."]
    #     section: Annotated[
    #         Literal["beginning", "middle", "end"],
    #         ...,
    #         "Section to query.",
    #     ]

    # # Define prompt for question-answering
    # prompt = hub.pull("rlm/rag-prompt")

    # # Define state for application
    # class State(TypedDict):
    #     question: str
    #     query: Search
    #     context: List[Document]
    #     answer: str


    # def analyze_query(state: State):
    #     structured_llm = llm.with_structured_output(Search)
    #     query = structured_llm.invoke(state["question"])
    #     return {"query": query}


    # def retrieve(state: State):
    #     query = state["query"]
    #     retrieved_docs = vector_store.similarity_search(
    #         query["query"],
    #         filter=lambda doc: doc.metadata.get("section") == query["section"],
    #     )
    #     return {"context": retrieved_docs}


    # def generate(state: State):
    #     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    #     messages = prompt.invoke({"question": state["question"], "context": docs_content})
    #     response = llm.invoke(messages)
    #     return {"answer": response.content}


    # graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    # graph_builder.add_edge(START, "analyze_query")

    # graph = graph_builder.compile()
    # response = graph.invoke({"question": "What is Task Decomposition?"})
    # print(response["answer"])

if __name__ == "__main__":
    main()