"""Unit tests for app.rag.chain — a fake chat model, no real Mistral API calls."""
import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag.chain import SYSTEM_PROMPT, answer_question, detect_city, format_docs, retrieve
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

    result = answer_question(vectorstore, llm, "Un concert ce week-end ?", k=2)

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


def test_prompt_formats_without_error_and_keeps_context_and_question():
    """SYSTEM_PROMPT passe dans un ChatPromptTemplate : une accolade égarée (par
    exemple dans un exemple JSON) ferait planter chaque requête. On vérifie que le
    formatage réel fonctionne et que contexte et question arrivent bien au modèle."""
    from app.rag.chain import PROMPT

    messages = PROMPT.format_messages(context="Événement X — Metz", question="Quoi à Metz ?")

    assert messages[0].content == SYSTEM_PROMPT
    assert "Événement X — Metz" in messages[1].content
    assert "Quoi à Metz ?" in messages[1].content


def test_answer_question_does_not_crash_on_unrelated_query(sample_df, fake_embeddings):
    documents = to_documents(sample_df)
    vectorstore = build_index(documents, fake_embeddings)
    llm = FakeListChatModel(
        responses=["Aucun événement du contexte ne correspond à votre question."]
    )

    result = answer_question(vectorstore, llm, "Quelle est la recette de la tarte flambée ?", k=2)

    assert result.answer
    assert len(result.sources) == 2


CITIES = ["Metz", "Saint-Dizier", "Dizier", "Épinal", "Bar-le-Duc"]


def test_detect_city_finds_the_city_ignoring_case_and_accents():
    assert detect_city("Quels concerts à metz ?", CITIES) == "Metz"
    assert detect_city("Des spectacles à Epinal en octobre", CITIES) == "Épinal"


def test_detect_city_prefers_the_longest_name():
    assert detect_city("Que faire à Saint-Dizier ?", CITIES) == "Saint-Dizier"


def test_detect_city_matches_whole_words_only():
    assert detect_city("Une exposition metzquelque chose", CITIES) is None


def test_detect_city_returns_none_without_a_known_city():
    assert detect_city("Quelle est la recette du brownie ?", CITIES) is None
    assert detect_city("Quels concerts à Paris ?", CITIES) is None


def test_retrieve_restricts_results_to_the_city_in_the_question(fake_embeddings):
    df = pd.DataFrame(
        [
            {"uid": str(i), "content": f"Concert numéro {i}", "title_fr": f"Concert {i}", "location_city": city}
            for i, city in enumerate(["Metz", "Colmar", "Colmar", "Nancy"])
        ]
    )
    vectorstore = build_index(to_documents(df), fake_embeddings)

    results = retrieve(vectorstore, "Quels concerts à Colmar ?", k=3)

    assert len(results) == 2
    assert {doc.metadata["location_city"] for doc in results} == {"Colmar"}


def test_retrieve_without_city_searches_the_whole_index(fake_embeddings):
    df = pd.DataFrame(
        [
            {"uid": str(i), "content": f"Concert numéro {i}", "title_fr": f"Concert {i}", "location_city": city}
            for i, city in enumerate(["Metz", "Colmar", "Nancy"])
        ]
    )
    vectorstore = build_index(to_documents(df), fake_embeddings)

    results = retrieve(vectorstore, "Quels concerts ce week-end ?", k=3)

    assert len(results) == 3


def test_detect_city_does_not_read_the_grand_est_region_as_the_town_of_grand():
    assert detect_city("Quels concerts dans le Grand Est en octobre ?", ["Grand", "Metz"]) is None
    assert detect_city("Des concerts dans le grand-est ?", ["Grand"]) is None
    assert detect_city("Que faire à Grand ce week-end ?", ["Grand"]) == "Grand"


def test_retrieve_treats_case_variants_of_a_city_as_the_same_city(fake_embeddings):
    """Open Agenda contient « Strasbourg » et « STRASBOURG » : les deux doivent remonter."""
    df = pd.DataFrame(
        [
            {"uid": str(i), "content": f"Concert numéro {i}", "title_fr": f"Concert {i}", "location_city": city}
            for i, city in enumerate(["Strasbourg", "STRASBOURG", "Nancy"])
        ]
    )
    vectorstore = build_index(to_documents(df), fake_embeddings)

    results = retrieve(vectorstore, "Quels concerts à Strasbourg ?", k=5)

    assert {doc.metadata["location_city"] for doc in results} == {"Strasbourg", "STRASBOURG"}
