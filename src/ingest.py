import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers import get_vector_store

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = os.getenv("PDF_PATH") or str(ROOT_DIR / "document.pdf")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

BATCH_SIZE = 20          # chunks por chamada de embedding
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 20  # espera cresce a cada tentativa (20s, 40s, ...)


def ingest_pdf():
    pdf_path = Path(PDF_PATH)
    if not pdf_path.is_absolute():
        pdf_path = ROOT_DIR / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    print(f"Lendo PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Nenhum texto extraído do PDF.")

    # Mantém só metadados preenchidos.
    for chunk in chunks:
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if v not in ("", None)}

    # IDs fixos: rodar a ingestão de novo atualiza os chunks em vez de duplicá-los.
    ids = [f"doc-{i}" for i in range(len(chunks))]

    store = get_vector_store()

    # Envia em lotes: os planos gratuitos limitam tokens de embedding por minuto.
    for start in range(0, len(chunks), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(chunks))
        _add_with_retry(store, chunks[start:end], ids[start:end])
        print(f"   {end}/{len(chunks)} chunks gravados")

    print(f"Ingestão concluída: {len(documents)} página(s), {len(chunks)} chunk(s) gravados no PGVector.")


def _add_with_retry(store, batch, batch_ids):
    """Grava um lote; se o provedor recusar por limite de taxa (429), espera e tenta de novo."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_documents(documents=batch, ids=batch_ids)
            return
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            is_rate_limit = "429" in message or "quota" in message or "rate limit" in message
            if not is_rate_limit or attempt == MAX_RETRIES:
                raise
            wait = RETRY_WAIT_SECONDS * attempt
            print(f"   Limite de requisições do provedor atingido; aguardando {wait}s (tentativa {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:  # noqa: BLE001
        print(f"Erro na ingestão: {exc}")
        sys.exit(1)
