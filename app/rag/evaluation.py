"""
Évaluation automatisée du système RAG avec Ragas.

Ragas a besoin d'un LLM "juge" pour noter les réponses (fidélité, pertinence...).
Par défaut il suppose OpenAI ; on lui fournit notre propre modèle Mistral et nos
propres embeddings HuggingFace via les wrappers LangChain de Ragas, pour rester
cohérent avec le reste du projet (pas de dépendance à un second fournisseur).
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from app.data.preprocessing import preprocess
from app.rag.chain import answer_question
from app.vectorstore.build import build_index, chunk_documents, to_documents

# Snapshot figé des données Open Agenda (Grand Est, événements à venir à cette date) sur
# lequel l'évaluation s'exécute TOUJOURS : les événements à venir périment chaque jour,
# alors que les réponses de référence de eval/qa_dataset.json portent sur des événements
# précis. Sans snapshot, les scores dériveraient de jour en jour sans que le système change.
SNAPSHOT_FILENAME = "openagenda_grand-est_2026-09-24.parquet"
SNAPSHOT_DATE = datetime(2026, 9, 24, tzinfo=timezone.utc)


def load_qa_dataset(path: Path) -> list[dict]:
    """Charger le jeu de questions/réponses de référence (eval/qa_dataset.json)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_snapshot_vectorstore(
    snapshot_path: Path, snapshot_date: datetime, embeddings: Embeddings
) -> VectorStore:
    """Reconstruire, en mémoire, l'index FAISS du snapshot : nettoyage avec la date de
    référence figée (pas la date du jour), chunking, vectorisation. Le pipeline est celui de
    production (mêmes fonctions), seule la date de référence est figée."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        df = preprocess(
            snapshot_path, Path(tmp_dir) / "events_clean.parquet", reference_date=snapshot_date
        )
    chunks = chunk_documents(to_documents(df))
    return build_index(chunks, embeddings)


def build_ragas_dataset(
    qa_pairs: list[dict], vectorstore: VectorStore, llm: BaseChatModel
) -> EvaluationDataset:
    """Rejouer chaque question sur le VRAI pipeline RAG (pas de raccourci) pour
    obtenir une réponse et des sources fraîches, et construire le dataset Ragas."""
    samples = []

    for pair in qa_pairs:
        result = answer_question(vectorstore, llm, pair["question"])

        samples.append(
            SingleTurnSample(
                user_input=pair["question"],
                response=result.answer,
                retrieved_contexts=[doc.page_content for doc in result.sources],
                reference=pair["reference_answer"],
            )
        )

    return EvaluationDataset(samples=samples)


def get_ragas_metrics() -> list:
    """Les 4 métriques demandées par le sujet : pertinence, fidélité au contexte,
    et couverture documentaire (precision + recall du contexte récupéré)."""
    return [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]


def run_evaluation(
    qa_pairs: list[dict],
    vectorstore: VectorStore,
    llm: BaseChatModel,
    embeddings: Embeddings,
) -> pd.DataFrame:
    """Construire le dataset Ragas et lancer l'évaluation. Retourne un DataFrame
    avec une ligne par question et une colonne par métrique."""
    dataset = build_ragas_dataset(qa_pairs, vectorstore, llm)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=get_ragas_metrics(),
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return result.to_pandas()
