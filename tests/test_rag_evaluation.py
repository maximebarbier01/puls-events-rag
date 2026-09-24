"""Tests de app.rag.evaluation — faux embeddings, aucun appel réseau ni modèle réel."""
import json
from datetime import datetime, timezone

import pandas as pd

from app.rag.evaluation import build_snapshot_vectorstore, load_qa_dataset

SNAPSHOT_DATE = datetime(2026, 9, 24, tzinfo=timezone.utc)


def _status(status_id: int) -> str:
    return json.dumps({"id": status_id, "label": {"fr": "x"}})


def _event(uid: str, title: str, end: str, source: str = "Centre Pompidou-Metz") -> dict:
    return {
        "uid": uid,
        "title_fr": title,
        "description_fr": "desc",
        "longdescription_fr": None,
        "conditions_fr": None,
        "keywords_fr": None,
        "accessibility_label_fr": None,
        "daterange_fr": "1er octobre",
        "firstdate_begin": "2026-10-01T18:00:00+00:00",
        "lastdate_end": end,
        "location_name": "Salle",
        "location_city": "Metz",
        "location_lat": 49.1,
        "location_lon": 6.2,
        "canonicalurl": f"https://openagenda.com/x/events/{uid}",
        "originagenda_title": source,
        "status": _status(1),
    }


def test_snapshot_vectorstore_uses_the_frozen_reference_date_not_today(tmp_path, fake_embeddings):
    # "avant" est terminé à la date du snapshot, "apres" est à venir : le résultat ne doit
    # dépendre que de SNAPSHOT_DATE, pas de la date à laquelle on lance le test.
    df = pd.DataFrame(
        [
            _event("avant", "Concert passé", "2026-09-20T20:00:00+00:00"),
            _event("apres", "Concert à venir", "2026-10-01T20:00:00+00:00"),
            _event("emploi", "Salon emploi", "2026-10-02T20:00:00+00:00", source="Mes événements France Travail"),
        ]
    )
    snapshot = tmp_path / "snapshot.parquet"
    df.to_parquet(snapshot)

    vectorstore = build_snapshot_vectorstore(snapshot, SNAPSHOT_DATE, fake_embeddings)

    uids = {doc.metadata["uid"] for doc in vectorstore.similarity_search("concert", k=10)}
    assert uids == {"apres"}


def test_load_qa_dataset_reads_question_and_reference(tmp_path):
    path = tmp_path / "qa.json"
    path.write_text(json.dumps([{"question": "Q ?", "reference_answer": "R"}]), encoding="utf-8")

    assert load_qa_dataset(path) == [{"question": "Q ?", "reference_answer": "R"}]
