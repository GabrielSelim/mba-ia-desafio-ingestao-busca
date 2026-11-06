import os
import sys
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
    store.add_documents(documents=chunks, ids=ids)

    print(f"Ingestão concluída: {len(documents)} página(s), {len(chunks)} chunk(s) gravados no PGVector.")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:  # noqa: BLE001
        print(f"Erro na ingestão: {exc}")
        sys.exit(1)
