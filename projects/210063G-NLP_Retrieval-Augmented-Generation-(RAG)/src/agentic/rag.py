import os
import asyncio
from typing_extensions import Literal

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc


from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# You need to import a language model
from langchain_openai import ChatOpenAI

# --- Initialize your LLM ---
llm = ChatOpenAI(model="gpt-4o-mini")

load_dotenv()  # Load environment variables from .env file

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    # rag = LightRAG(
    # working_dir=WORKING_DIR,
    # llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    # llm_model_name='gpt-oss:20b', # Your model name
    # # Use Ollama embedding function
    # embedding_func=EmbeddingFunc(
    #     embedding_dim=768,
    #     func=lambda texts: ollama_embed(
    #         texts,
    #         embed_model="embeddinggemma:300m"
    #             )
    #         ),
    #     )

    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag


rag = asyncio.run(initialize_rag())


def query_rag(query: str, mode="hybrid"):
    response = asyncio.run(rag.aquery(query, param=QueryParam(mode=mode)))
    return response


def retrieve_rag(query: str, mode: str = "hybrid") -> dict | None:
    result = asyncio.run(rag.aquery_data(query, param=QueryParam(mode=mode)))
    return result.get("data") if result else None
