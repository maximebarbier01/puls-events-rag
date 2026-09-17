"""Créez l'index vectoriel FAISS à partir du fichier data/interim/events_clean.parquet."""

from pathlib import Path

from app.vectorstore.build import (
    build_index,
    chunk_documents,
    get_embeddings,
    load_processed_events,
    save_index,
    to_documents,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data/interim/events_clean.parquet"
INDEX_PATH = PROJECT_ROOT / "index/events_faiss"


def main() -> None:
    df = load_processed_events(PROCESSED_PATH)
    print(f"Loaded {len(df)} processed events")

    documents = to_documents(df)
    chunks = chunk_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = build_index(chunks, embeddings)

    save_index(vectorstore, INDEX_PATH)
    print(f"Saved FAISS index to {INDEX_PATH}")


if __name__ == "__main__":
    main()
