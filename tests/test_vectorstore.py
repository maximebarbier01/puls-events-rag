"""Unit tests for app.vectorstore.build — a fake deterministic Embeddings, no real model."""
import pandas as pd
import pytest

from app.vectorstore.build import (
    build_index,
    chunk_documents,
    load_index,
    save_index,
    to_documents,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "uid": "1",
                "content": "Concert de jazz au Centre Pompidou-Metz ce samedi.",
                "title_fr": "Concert de jazz",
                "firstdate_begin": pd.Timestamp("2026-09-20", tz="UTC"),
                "lastdate_end": pd.Timestamp("2026-09-20", tz="UTC"),
                "daterange_fr": "Samedi 20 septembre",
                "location_city": "Metz",
                "location_name": "Centre Pompidou-Metz",
                "location_lat": 49.11,
                "location_lon": 6.18,
                "canonicalurl": "https://openagenda.com/culture/events/1",
                "is_full": False,
                "keywords_fr": ["musique", "jazz"],
            },
            {
                "uid": "2",
                "content": "Exposition de peinture contemporaine à la galerie municipale.",
                "title_fr": "Exposition peinture",
                "firstdate_begin": pd.Timestamp("2026-10-01", tz="UTC"),
                "lastdate_end": pd.Timestamp("2026-10-15", tz="UTC"),
                "daterange_fr": "Du 1er au 15 octobre",
                "location_city": "Thionville",
                "location_name": "Galerie municipale",
                "location_lat": None,
                "location_lon": None,
                "canonicalurl": "https://openagenda.com/culture/events/2",
                "is_full": True,
                "keywords_fr": None,
            },
        ]
    )


def test_to_documents_builds_one_document_per_row_with_metadata(sample_df):
    documents = to_documents(sample_df)

    assert len(documents) == 2
    assert documents[0].page_content == sample_df.iloc[0]["content"]
    assert documents[0].metadata["uid"] == "1"
    assert documents[0].metadata["location_city"] == "Metz"
    assert documents[0].metadata["is_full"] is False
    # datetime metadata is serialized to a plain string, not left as a Timestamp
    assert isinstance(documents[0].metadata["firstdate_begin"], str)


def test_chunk_documents_keeps_short_text_as_a_single_chunk(sample_df):
    documents = to_documents(sample_df)
    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=150)

    assert len(chunks) == 2
    assert chunks[0].metadata["uid"] == "1"


def test_chunk_documents_splits_long_text_and_propagates_metadata():
    from langchain_core.documents import Document

    long_text = "Phrase à propos d'un festival culturel. " * 60  # well over 1000 chars
    doc = Document(page_content=long_text, metadata={"uid": "long-1"})

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.metadata["uid"] == "long-1" for c in chunks)
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_build_index_and_similarity_search_returns_relevant_documents(sample_df, fake_embeddings):
    documents = to_documents(sample_df)
    vectorstore = build_index(documents, fake_embeddings)

    results = vectorstore.similarity_search("Concert de jazz au Centre Pompidou-Metz ce samedi.", k=1)

    assert len(results) == 1
    assert results[0].metadata["uid"] == "1"


def test_save_and_load_index_round_trip(tmp_path, sample_df, fake_embeddings):
    documents = to_documents(sample_df)
    embeddings = fake_embeddings
    vectorstore = build_index(documents, embeddings)

    index_path = tmp_path / "events_faiss"
    save_index(vectorstore, index_path)

    assert (index_path / "index.faiss").exists()
    assert (index_path / "index.pkl").exists()

    reloaded = load_index(index_path, embeddings)
    results = reloaded.similarity_search("Exposition de peinture contemporaine.", k=1)

    assert len(results) == 1
    assert results[0].metadata["uid"] in {"1", "2"}
    assert reloaded.index.ntotal == vectorstore.index.ntotal
