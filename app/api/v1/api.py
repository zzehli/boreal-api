from fastapi import APIRouter

from app.langgraph.graph import RAGGraph

graph = RAGGraph()

api_router = APIRouter()

@api_router.get("/")
async def read_root():
    response = await graph.get_response("What your sustainability practices?")
    return response