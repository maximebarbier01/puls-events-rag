"""Unit tests for app.rag.chain — a fake chat model, no real Mistral API calls."""
import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag.chain import SYSTEM_PROMPT, answer_question, format_docs
from app.vectorstore.build import build_index, to_documents


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "uid": "1",
                "content": "Concert de jazz au Centre Pompidou-Metz ce samedi soir.",
                "title_fr": "Concert de jazz",
                "daterange_fr": "Samedi 20 septembre",
                "location_city": "Metz",
                "canonicalurl": "https://openagenda.com/culture/events/1",
            },
            {
                "uid": "2",
                "content": "Exposition de peinture contemporaine à la galerie municipale.",
                "title_fr": "Exposition peinture",
                "daterange_fr": "Du 1er au 15 octobre",
                "location_city": "Thionville",
                "canonicalurl": "https://openagenda.com/culture/events/2",
            },
        ]
    )


def test_format_docs_includes_title_dates_location_and_source():
    doc = Document(
        page_content="Un concert exceptionnel.",
        metadata={
            "title_fr": "Concert de jazz",
            "daterange_fr": "Samedi 20 septembre",
            "location_city": "Metz",
            "canonicalurl": "https://openagenda.com/culture/events/1",
        },
    )

    formatted = format_docs([doc])

    assert "Concert de jazz" in formatted
    assert "Samedi 20 septembre" in formatted
    assert "Metz" in formatted
    assert "Un concert exceptionnel." in formatted
    assert "https://openagenda.com/culture/events/1" in formatted


def test_answer_question_returns_llm_answer_and_retrieved_sources(sample_df, fake_embeddings):
    documents = to_documents(sample_df)
    vectorstore = build_index(documents, fake_embeddings)
    llm = FakeListChatModel(responses=["Il y a un concert de jazz samedi à Metz."])

    result = answer_question(vectorstore, llm, "Un concert à Metz ?", k=2)

    assert result.answer == "Il y a un concert de jazz samedi à Metz."
    assert len(result.sources) == 2
    assert {doc.metadata["uid"] for doc in result.sources} == {"1", "2"}


def test_system_prompt_instructs_the_model_to_refuse_when_context_does_not_match():
    """Garde-fou de non-régression : si quelqu'un modifie le prompt par erreur et
    retire la consigne anti-hallucination, ce test doit échouer. Ça ne vérifie pas
    que le vrai modèle obéit (ça, seul un appel réel peut le confirmer — voir
    tests/test_rag_integration.py), juste que l'instruction est toujours présente."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "correspond" in prompt_lower
    assert "dis-le clairement" in prompt_lower
    assert "invente" in prompt_lower


def test_answer_question_does_not_crash_on_unrelated_query(sample_df, fake_embeddings):
    documents = to_documents(sample_df)
    vectorstore = build_index(documents, fake_embeddings)
    llm = FakeListChatModel(
        responses=["Aucun événement du contexte ne correspond à votre question."]
    )

    result = answer_question(vectorstore, llm, "Quelle est la recette de la tarte flambée ?", k=2)

    assert result.answer
    assert len(result.sources) == 2
