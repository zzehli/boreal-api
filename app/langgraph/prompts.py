from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

# Get the directory where the current file (prompts.py) is located
CURRENT_DIR = Path(__file__).parent

# Use the directory to construct the path to the prompt file
with open(CURRENT_DIR / "generation-prompt.txt", "r", encoding="utf-8") as f:
    generation_prompt = f.read()

ROUTER_SYSTEM_PROMPT = """You are a customer agent for Nestlé. Route the user input, 
if the question is about Nestlé's company itself, products or services, choose RAG; 
otherwise, choose chat."""

QUERY_ANALYZER_SYSTEM_PROMPT = """You are a query analyzer for a RAG application. 
If the user's question refers to previous conversations, reformulate the question to provide 
more specific information for information retrieval. Use the message history to disambiguate the question. 
If there is no ambiguity, return the original question."""

CHAT_SYSTEM_PROMPT = """You are a customer agent for Nestlé. Answer customer's question based on the context provided. 
Ask clarification questions to allow the user to provide more specific information for information retrieval. 
Ground your answer in the context provided."""

GENERATION_SYSTEM_PROMPT = generation_prompt