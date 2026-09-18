"""
Tests fonctionnels de l'API (app/main.py) — nom exact demandé par le sujet
(api_test.py). Aucun vrai modèle d'embedding, aucun vrai appel Mistral : le
lifespan de l'app n'est jamais déclenché (TestClient sans "with", vérifié plus bas),
et app.state est peuplé à la main avec de faux objets avant chaque test.
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.main import app
from app.vectorstore.build import build_index, to_documents


def _status(status_id: int, label_fr: str) -> str:
    return json.dumps({"id": status_id, "label": {"fr": label_fr}})


@pytest.fixture
def raw_events_file(tmp_path):
    """Un petit fichier parquet brut, avec le schéma attendu par preprocess()."""
    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title_fr": "Concert de jazz au Centre Pompidou-Metz",
                "description_fr": "Un concert.",
                "longdescription_fr": "<p>Un concert exceptionnel.</p>",
                "conditions_fr": "Gratuit",
                "keywords_fr": ["musique", "jazz"],
                "accessibility_label_fr": None,
                "daterange_fr": "Samedi 20 septembre",
                "firstdate_begin": pd.Timestamp.now(tz="UTC"),
                "lastdate_end": pd.Timestamp.now(tz="UTC"),
                "location_name": "Centre Pompidou-Metz",
                "location_city": "Metz",
                "location_lat": 49.11,
                "location_lon": 6.18,
                "canonicalurl": "https://openagenda.com/culture/events/1",
                "originagenda_title": "Centre Pompidou-Metz",
                "status": _status(1, "Programmé"),
            }
        ]
    )
    path = tmp_path / "raw.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def client(tmp_path, fake_embeddings, raw_events_file):
    """TestClient sans "with" : le lifespan (chargement du vrai modèle/index/LLM)
    n'est donc jamais déclenché. On pose nous-mêmes un état complet et léger sur
    app.state, comme le ferait le lifespan mais avec des objets factices.
    """
    initial_docs = to_documents(
        pd.DataFrame(
            [
                {
                    "uid": "1",
                    "content": "Concert de jazz au Centre Pompidou-Metz.",
                    "title_fr": "Concert de jazz",
                    "daterange_fr": "Samedi 20 septembre",
                    "location_city": "Metz",
                    "canonicalurl": "https://openagenda.com/culture/events/1",
                }
            ]
        )
    )

    app.state.embeddings = fake_embeddings
    app.state.vectorstore = build_index(initial_docs, fake_embeddings)
    app.state.llm = FakeListChatModel(responses=["Il y a un concert de jazz samedi à Metz."])
    app.state.raw_path = raw_events_file
    app.state.interim_path = tmp_path / "interim.parquet"
    app.state.index_path = tmp_path / "index_faiss"

    return TestClient(app)


def test_ask_with_valid_question_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "Un concert à Metz ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Il y a un concert de jazz samedi à Metz."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["title"] == "Concert de jazz"


def test_ask_with_empty_question_is_rejected(client):
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_without_question_field_is_rejected(client):
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_rebuild_without_token_configured_is_forbidden(client, monkeypatch):
    monkeypatch.delenv("REBUILD_TOKEN", raising=False)

    response = client.post("/rebuild")

    assert response.status_code == 403


def test_rebuild_with_wrong_token_is_unauthorized(client, monkeypatch):
    monkeypatch.setenv("REBUILD_TOKEN", "le-bon-jeton")

    response = client.post("/rebuild", headers={"X-Rebuild-Token": "mauvais-jeton"})

    assert response.status_code == 401


def test_rebuild_with_correct_token_updates_the_live_index(client, monkeypatch):
    monkeypatch.setenv("REBUILD_TOKEN", "le-bon-jeton")

    response = client.post("/rebuild", headers={"X-Rebuild-Token": "le-bon-jeton"})

    assert response.status_code == 200
    body = response.json()
    assert body["nb_documents"] == 1
    assert body["nb_chunks"] >= 1

    # L'index utilisé par /ask doit refléter immédiatement le rebuild, sans
    # redémarrer le serveur.
    ask_response = client.post("/ask", json={"question": "Un concert à Metz ?"})
    assert ask_response.status_code == 200
    assert ask_response.json()["sources"][0]["title"] == "Concert de jazz au Centre Pompidou-Metz"


def test_docs_page_is_served(client):
    response = client.get("/docs")

    assert response.status_code == 200


def test_health_reports_ok_when_vectorstore_and_llm_are_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vectorstore_loaded"] is True
    assert body["llm_configured"] is True


def test_metadata_reflects_current_index_size(client):
    response = client.get("/metadata")

    assert response.status_code == 200
    body = response.json()
    assert body["nb_chunks_indexed"] == 1  # l'unique événement de la fixture "client"
    assert body["embedding_model"]
    assert body["llm_model"]
    assert body["default_top_k"] == 5


def test_metadata_reads_the_live_vectorstore_not_a_stale_value(client, monkeypatch):
    """/rebuild remplace app.state.vectorstore par une toute nouvelle instance FAISS
    (voir app/api/routes.py::rebuild) : /metadata doit refléter cette nouvelle
    instance immédiatement, pas une valeur mise en cache à l'ancien objet."""
    monkeypatch.setenv("REBUILD_TOKEN", "le-bon-jeton")
    client.post("/rebuild", headers={"X-Rebuild-Token": "le-bon-jeton"})

    response = client.get("/metadata")

    assert response.json()["nb_chunks_indexed"] == 1
