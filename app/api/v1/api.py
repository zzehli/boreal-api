import uuid

from fastapi import APIRouter, Cookie, Request, Response
from pydantic import BaseModel

from app.langgraph.graph import RAGGraph

graph = RAGGraph()

api_router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@api_router.post("/chat")
async def read_root(
    request: ChatRequest,
    response: Response, 
    chat_user_id: str = Cookie(default=None)):
    if not chat_user_id:
        chat_user_id = str(uuid.uuid4())
        response.set_cookie(key="chat_user_id", value=chat_user_id, max_age=60 * 60 * 24, samesite='strict')
    message = request.message
    response = await graph.get_response(message, chat_user_id)
    return response

@api_router.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Health status information.
    """
    return {"status": "healthy", "version": "1.0.0"}