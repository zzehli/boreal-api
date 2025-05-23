import asyncio
import os

from dotenv import load_dotenv

from app.langgraph.graph import RAGGraph

load_dotenv()

async def main():
    rag_graph = RAGGraph()
    response = await rag_graph.get_response("What your sustainability practices?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())