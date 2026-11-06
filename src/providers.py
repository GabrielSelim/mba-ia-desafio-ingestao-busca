"""Fábrica de embeddings, LLM e vector store conforme o provedor configurado no .env."""
import os

from dotenv import load_dotenv
from langchain_postgres import PGVector

load_dotenv()

DEFAULT_COLLECTION = "documento_pdf"


def get_provider() -> str:
    """Retorna 'openai' ou 'gemini'. Sem LLM_PROVIDER, usa a primeira chave preenchida."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if provider:
        if provider not in ("openai", "gemini"):
            raise ValueError("LLM_PROVIDER deve ser 'openai' ou 'gemini'.")
        return provider
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    raise ValueError("Configure OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env.")


def _require(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise ValueError(f"Variável de ambiente {var} não configurada no .env.")
    return value


def get_embeddings():
    if get_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        _require("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    _require("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )


def get_llm():
    if get_provider() == "openai":
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
        # Modelos de raciocínio (gpt-5*, o*) só aceitam a temperatura padrão.
        if model.startswith(("gpt-5", "o")):
            return ChatOpenAI(model=model)
        return ChatOpenAI(model=model, temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"), temperature=0
    )


def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME") or DEFAULT_COLLECTION,
        connection=_require("DATABASE_URL"),
        use_jsonb=True,
    )
