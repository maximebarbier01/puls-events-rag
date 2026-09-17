"""Chunk, embed and index the cleaned events into a local FAISS vector store."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

METADATA_COLUMNS = [
    "uid",
    "title_fr",
    "firstdate_begin",
    "lastdate_end",
    "daterange_fr",
    "location_city",
    "location_name",
    "location_lat",
    "location_lon",
    "canonicalurl",
    "is_full",
    "keywords_fr",
]


def load_processed_events(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _row_metadata(row: pd.Series) -> dict:
    metadata = {}
    for col in METADATA_COLUMNS:
        value = row.get(col)
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        elif hasattr(value, "tolist"):  # numpy array (e.g. keywords_fr)
            value = value.tolist()
        metadata[col] = value
    return metadata


def to_documents(df: pd.DataFrame) -> list[Document]:
    return [
        Document(page_content=row["content"], metadata=_row_metadata(row))
        for _, row in df.iterrows()
    ]


def chunk_documents(
    documents: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_index(documents: list[Document], embeddings: Embeddings) -> FAISS:
    return FAISS.from_documents(documents, embeddings)


def save_index(vectorstore: FAISS, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))


def load_index(path: Path, embeddings: Embeddings) -> FAISS:
    # The index is generated locally by our own scripts (scripts/build_index.py), never
    # loaded from an untrusted external source, so unpickling the docstore is safe here.
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
